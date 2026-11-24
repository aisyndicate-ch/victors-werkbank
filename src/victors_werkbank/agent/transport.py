"""Byte-bounded HTTP transport; IsolatedEndpoint enforces total wall deadlines.

A streamed request (`--stream`) reads a strict server-sent event stream and
assembles it into the non-streaming chat completion body; nothing is repaired.

An API key (`WERKBANK_API_KEY`) is optional: with one, every request to the
endpoint carries it as a bearer token; without one, no request changes. The
key lives in memory only and is never part of evidence or an exception text.
"""

import html
import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any, TypeGuard

from .measurements import (
    DEFAULT_SAMPLING,
    SAMPLING_KEYS,
    carries_token,
    has_token,
    is_error_chunk,
    validate_sampling,
)

# Also bounds one SSE line, one SSE event's data and an assembled stream body.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
# Raw bytes of one event stream after transfer decoding: data, framing,
# comments and blank lines alike.
MAX_STREAM_BYTES = 64 * 1024 * 1024
# Leading bytes of an HTTP error body kept as `EndpointHTTPError` evidence.
HTTP_BODY_BYTES = 8192
# Deepest error body `is_capacity_rejection` parses: a deeper one is never a
# capacity rejection, so the decision cannot depend on the interpreter stack.
CAPACITY_BODY_NESTING = 64
# Deepest tool arguments `ToolCall.arguments` parses: deeper ones are malformed
# on any interpreter stack, and no case copies or compares a value nested
# deeper.
TOOL_ARGUMENT_NESTING = 16
# Every read-only GET a probe may send: the model list and the engine probes
# of `serving.RUNTIMES`, below the base or at its root.
PROBE_PATHS = frozenset(
    {"models", "version", "props", "get_server_info", "api/version", "api/v1/models"}
)
# The one source of an API key: this environment variable. Unset or empty
# means no key.
API_KEY_VARIABLE = "WERKBANK_API_KEY"
MAX_API_KEY_CHARS = 4096
# A key shorter than this is a placeholder, not a secret (a local server is
# often started with a word such as "EMPTY"). It is sent like any key, but it
# is not sought in what a server answers or in a submission: a short word
# stands in ordinary error texts, where finding it would withhold a body that
# decides a status.
SECRET_KEY_CHARS = 8
# Fixed text: a refused key is never shown.
API_KEY_REFUSAL = f"{API_KEY_VARIABLE} is not a usable key"
_JSON_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\(.)", re.S)
_JSON_ESCAPED = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}


class EndpointError(ValueError):
    """Transport/response-envelope failure; raw evidence is retained when available."""

    def __init__(self, message: str, raw: Any = None):
        super().__init__(message)
        self.raw = raw


class ModelOutputError(EndpointError):
    """Malformed tool-call arguments; the loop answers the call with a tool error
    (`invalid_json_arguments`), so they never end an episode by themselves."""


class EndpointHTTPError(EndpointError):
    """A non-2xx HTTP status; `raw` is the transport's own status/body evidence.

    Only this error can be a request-capacity rejection: a refused 200 body is
    endpoint data and never carries trusted HTTP status evidence.
    """


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EndpointError("Duplicate JSON key")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise EndpointError("Non-finite JSON number")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise EndpointError("Non-finite JSON number")
        return number

    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (ValueError, TypeError, RecursionError) as error:
        raise EndpointError("Invalid JSON") from error


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments_raw: str

    def arguments(self) -> dict[str, Any]:
        # A fixed lexical depth before CPython's stack-dependent parser.
        if json_nesting(self.arguments_raw) > TOOL_ARGUMENT_NESTING:
            raise ModelOutputError("Tool argument nesting limit", self.arguments_raw)
        try:
            value = strict_json(self.arguments_raw)
        except EndpointError:
            raise ModelOutputError(
                "Invalid tool argument JSON", self.arguments_raw
            ) from None
        if not isinstance(value, dict):
            raise ModelOutputError(
                "Tool arguments must be an object", self.arguments_raw
            )
        return value


@dataclass(frozen=True)
class Reply:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    usage: dict[str, Any] | None
    raw: dict[str, Any]


def parse_reply(data: Any) -> Reply:
    try:
        if not isinstance(data, dict) or len(data["choices"]) != 1:
            raise EndpointError("Expected one response choice")
        choice = data["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise EndpointError("Expected assistant role")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise EndpointError("Unsupported content format")
        reason = choice["finish_reason"]
        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise EndpointError("Invalid usage metadata")
        # A truncated action is never executed, but remains raw token-limit evidence.
        if reason == "length":
            return Reply(content, (), reason, usage, data)
        calls = []
        native_calls = message.get("tool_calls")
        if native_calls is None:
            native_calls = []
        if not isinstance(native_calls, list):
            raise EndpointError("Invalid native tool call list")
        for item in native_calls:
            function = item["function"]
            values = [item["id"], function["name"], function["arguments"]]
            if item["type"] != "function" or not all(
                isinstance(v, str) for v in values
            ):
                raise EndpointError("Invalid native tool call")
            if not item["id"] or not function["name"]:
                raise EndpointError("Empty tool identity")
            calls.append(ToolCall(*values))
        if len({call.id for call in calls}) != len(calls):
            raise EndpointError("Duplicate tool call ID")
        if reason not in {"stop", "tool_calls", "length", "content_filter"}:
            raise EndpointError("Unsupported finish reason")
        if calls and reason != "tool_calls":
            raise EndpointError("Tool calls without matching finish reason")
        if reason == "tool_calls" and not calls:
            raise EndpointError("Empty native tool call result")
        return Reply(content, tuple(calls), reason, usage, data)
    except EndpointError as error:
        error.raw = data
        raise error from None
    except (KeyError, TypeError, IndexError):
        raise EndpointError("Malformed response envelope", data) from None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise EndpointError("Endpoint redirects are not supported")


def _encoded(value: Any) -> bytes:
    """The bytes the journal encodes `value` as (`loop.encode_event`)."""
    return json.dumps(value, allow_nan=False, ensure_ascii=False).encode(
        "utf-8", errors="backslashreplace"
    )


def _joined(parts: list[str]) -> str:
    """Streamed text as the journal reads it back: a UTF-16 surrogate pair
    split across chunks becomes the one character it encodes."""
    text = "".join(parts)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return json.loads(json.dumps(text))
    return text


def _refused(value: Any) -> bool:
    try:
        parse_reply(value)
    except EndpointError:
        return True
    return False


_CALL_KEYS = frozenset({"index", "id", "type", "function"})
_FUNCTION_KEYS = frozenset({"name", "arguments"})


def _first_delta(chunk: object) -> object:
    """The delta of a chunk's first choice, where it has one; an error object
    has none (an error event never sets the first token)."""
    if not isinstance(chunk, dict) or is_error_chunk(chunk):
        return None
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("delta")


class _StreamAssembler:
    """Folds streamed chat completion chunks into the non-streaming body shape.

    Top-level and choice metadata keep their first non-null value, `usage` its
    last one. Delta text fields and tool-call arguments concatenate in order;
    the role, the finish reason and each call's id, type and name keep their
    first non-null value, and a different later one is "Malformed stream". A
    key is in the body iff some chunk reported it, null if only null was. Only
    `parse_reply` decides whether the body is a reply; nothing is repaired.
    """

    def __init__(self) -> None:
        self.meta: dict[str, Any] = {}
        self.usage: tuple[Any, int] | None = None  # Last value, encoded size.
        self.chosen = False
        self.choice: dict[str, Any] = {}
        self.finish: dict[str, Any] = {"finish_reason": None}
        # The role (str or None), tool calls (call records or None) and text
        # fields (fragments or None), in the order deltas reported them.
        self.message: dict[str, Any] = {}
        # A loose lower bound of the body's encoded size (`_encoded`; one
        # character encodes to at least one byte; quotes, separators and
        # nulls are not counted), so a stream far above the limit is refused
        # early; the assembled body's exact size is checked at the end.
        self.size = 0

    def grow(self, size: int) -> None:
        self.size += size
        usage = 0 if self.usage is None else self.usage[1]
        if self.size + usage > MAX_RESPONSE_BYTES:
            raise EndpointError("Response exceeds byte limit")

    def keep(
        self, fields: dict[str, Any], key: str, value: Any, *, conflict: bool
    ) -> None:
        """Keep the first non-null value; with `conflict`, a different later
        non-null value is malformed (`True` differs from `1`)."""
        held = fields.get(key)
        if held is None:
            if key not in fields:
                self.grow(len(key))
            fields[key] = value
            if value is not None:
                self.grow(
                    len(value) if isinstance(value, str) else len(_encoded(value))
                )
        elif (
            conflict
            and value is not None
            and (type(value), value) != (type(held), held)
        ):
            raise EndpointError("Malformed stream")

    def text(self, fields: dict[str, Any], key: str, value: Any) -> None:
        """Append one text fragment (str or null) to `fields[key]`."""
        if value is not None and not isinstance(value, str):
            raise EndpointError("Malformed stream")
        if key not in fields:
            self.grow(len(key))
            fields[key] = None
        if value is not None:
            self.grow(len(value))
            if fields[key] is None:
                fields[key] = []
            fields[key].append(value)

    def add(self, chunk: Any) -> dict[str, Any] | None:
        """Fold one parsed chunk in; returns its delta, or None without a choice."""
        if not isinstance(chunk, dict):
            raise EndpointError("Malformed stream")
        if is_error_chunk(chunk):
            # Never an HTTP status, so never a capacity rejection. Evidence only
            # if no reply can be read from it and it carries no output, which
            # only deltas report.
            raw = chunk if _refused(chunk) and not carries_token(chunk) else None
            raise EndpointError("Endpoint stream error", raw)
        choices = chunk.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise EndpointError("Malformed stream")
        for key, value in chunk.items():
            if key == "usage":
                if value is not None or self.usage is None:
                    self.usage = (value, len(key) + len(_encoded(value)))
                    self.grow(0)
            elif key != "choices":
                self.keep(self.meta, key, value, conflict=False)
        if not choices:
            return None
        choice = choices[0]
        if (
            not isinstance(choice, dict)
            or type(choice.get("index")) is not int
            or choice["index"] != 0
            or not isinstance(choice.get("delta"), dict)
            or "message" in choice
        ):
            raise EndpointError("Malformed stream")
        self.chosen = True
        for key, value in choice.items():
            if key == "finish_reason":
                self.keep(self.finish, key, value, conflict=True)
            elif key not in ("index", "delta"):
                self.keep(self.choice, key, value, conflict=False)
        delta: dict[str, Any] = choice["delta"]
        for key, value in delta.items():
            if key == "role":
                if value is not None and not isinstance(value, str):
                    raise EndpointError("Malformed stream")
                self.keep(self.message, key, value, conflict=True)
            elif key == "tool_calls":
                self.calls(value)
            else:
                self.text(self.message, key, value)
        return delta

    def calls(self, value: Any) -> None:
        """Fold one delta's tool-call fragments in by their `index`."""
        if value is not None and not isinstance(value, list):
            raise EndpointError("Malformed stream")
        if "tool_calls" not in self.message:
            self.grow(len("tool_calls"))
            self.message["tool_calls"] = None
        if value is None:
            return
        if self.message["tool_calls"] is None:
            self.message["tool_calls"] = []
        calls: list[dict[str, Any]] = self.message["tool_calls"]
        for item in value:
            if not isinstance(item, dict) or not set(item) <= _CALL_KEYS:
                raise EndpointError("Malformed stream")
            index = item.get("index")
            # A new call takes the next index; a gap is malformed.
            if type(index) is not int or not 0 <= index <= len(calls):
                raise EndpointError("Malformed stream")
            if index == len(calls):
                self.grow(len("{}"))
                calls.append({})
            call = calls[index]
            for key in ("id", "type"):
                if key in item:
                    self.keep(call, key, item[key], conflict=True)
            if "function" not in item:
                continue
            function = item["function"]
            if function is not None and (
                not isinstance(function, dict) or not set(function) <= _FUNCTION_KEYS
            ):
                raise EndpointError("Malformed stream")
            if "function" not in call:
                self.grow(len("function"))
                call["function"] = None
            if function is None:
                continue
            if call["function"] is None:
                call["function"] = {}
            if "name" in function:
                self.keep(call["function"], "name", function["name"], conflict=True)
            if "arguments" in function:
                self.text(call["function"], "arguments", function["arguments"])

    def body(self) -> dict[str, Any]:
        """The assembled body; its tool calls have no stream `index`, as in a
        non-streaming body (they return to the endpoint in the history)."""
        choices = []
        if self.chosen:
            message: dict[str, Any] = {}
            for key, value in self.message.items():
                if key == "role" or value is None:
                    message[key] = value
                elif key == "tool_calls":
                    message[key] = [_assembled_call(call) for call in value]
                else:
                    message[key] = _joined(value)
            choices.append(
                {"index": 0, "message": message, **self.choice, **self.finish}
            )
        body = {**self.meta, "choices": choices}
        if self.usage is not None:
            body["usage"] = self.usage[0]
        return body


def _assembled_call(record: dict[str, Any]) -> dict[str, Any]:
    call = {key: record[key] for key in ("id", "type") if key in record}
    if "function" in record:
        function = record["function"]
        if function is not None:
            function = {
                key: function[key] for key in ("name", "arguments") if key in function
            }
            if function.get("arguments") is not None:
                function["arguments"] = _joined(function["arguments"])
        call["function"] = function
    return call


def check_api_key(key: object) -> None:
    """A key is 1 to `MAX_API_KEY_CHARS` characters of printable ASCII without
    a space (0x21 to 0x7E), so it is one header value and nothing else. The
    refusal is a fixed text that never holds the value."""
    if (
        not isinstance(key, str)
        or not 1 <= len(key) <= MAX_API_KEY_CHARS
        or any(not 0x21 <= ord(char) <= 0x7E for char in key)
    ):
        raise EndpointError(API_KEY_REFUSAL)


def _json_unescaped(match: re.Match[str]) -> str:
    if match[1] is not None:
        return chr(int(match[1], 16))
    return _JSON_ESCAPED.get(match[2], match[2])


def holds_key(data: bytes, key: str) -> bool:
    """Whether `data` holds the secret `key`: as it stands, or as a JSON
    string, a percent-encoded value or an HTML page writes it, whichever of
    its characters the writer chose to escape. A key shorter than
    `SECRET_KEY_CHARS` is no secret and is never sought."""
    if len(key) < SECRET_KEY_CHARS:
        return False
    text = data.decode("utf-8", errors="replace")
    if key in text:
        return True
    forms = [text]
    if "\\" in text:
        forms.append(_JSON_ESCAPE.sub(_json_unescaped, text))
    if "%" in text:
        forms += [urllib.parse.unquote(form) for form in forms]
    if "&" in text:
        forms += [html.unescape(form) for form in forms]
    return any(key in form for form in forms[1:])


def api_key_from_environment(
    environ: MutableMapping[str, str] | None = None, *, keep: bool = False
) -> str | None:
    """The API key of `WERKBANK_API_KEY`, or None where it is unset or empty.

    Unless `keep`, the variable is removed from `environ` (this process's
    environment by default) as it is read, also where the key is then
    refused, so no child process started afterwards inherits it. `keep` is
    for a process whose `werkbank run` children read the variable themselves.
    """
    source = os.environ if environ is None else environ
    key = source.get(API_KEY_VARIABLE) if keep else source.pop(API_KEY_VARIABLE, None)
    if not key:
        return None
    check_api_key(key)
    return key


def check_probe_path(path: object) -> None:
    """Probe paths are fixed literals from our code, never caller URLs."""
    if not isinstance(path, str) or path not in PROBE_PATHS:
        raise EndpointError("Invalid probe path")


class ChatEndpoint:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 30.0,
        api_key: str | None = None,
    ):
        url = urllib.parse.urlsplit(base_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise EndpointError("Expected HTTP(S) endpoint")
        if url.username or url.password or url.query or url.fragment:
            raise EndpointError(
                "Credentials, query and fragment are not endpoint paths"
            )
        # The base URL is the /v1 path; the chat path is appended to it.
        if url.path.rstrip("/").endswith("/chat/completions"):
            raise EndpointError(
                "The endpoint is the /v1 base URL, without /chat/completions"
            )
        if not model or not math.isfinite(timeout) or timeout <= 0:
            raise EndpointError("Explicit model and positive timeout required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        if api_key is not None:
            check_api_key(api_key)
        # With a key, every request to this endpoint carries it as a bearer
        # token; redirects are refused, so it reaches no other host.
        self._api_key = api_key
        # Do not inherit host proxy configuration for local model traffic.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        sampling: dict[str, Any] | None = None,
        on_send: Callable[[float], None] | None = None,
        stream: bool = False,
        on_first_token: Callable[[float], None] | None = None,
    ) -> Reply:
        """One request, streamed only if `stream`; it carries exactly the
        settings in `sampling`, none by default, so the server applies its
        own value for every setting not sent. Template variables go out as
        the top-level `chat_template_kwargs` verbatim: never translated for
        an engine and never resent without them when a server refuses them.

        `exchange` holds the monotonic (sent, received) of the network exchange
        alone, without request encoding or response parsing; `on_send` gets the
        send time first. A stream is received at its `data: [DONE]` event and
        never drained after it. `first_token` holds the monotonic moment the
        first event carrying output (`has_token`) was read, before its JSON was
        parsed, or None; `on_first_token` gets it once, as soon as it is known.
        """
        self.exchange: tuple[float, float] | None = None
        self.first_token: float | None = None
        if type(max_tokens) is not int or max_tokens < 1:
            raise EndpointError("Positive output budget required")
        if type(stream) is not bool:
            raise EndpointError("Streaming mode must be a boolean")
        try:
            settings = validate_sampling(
                dict(DEFAULT_SAMPLING) if sampling is None else sampling
            )
        except ValueError:
            raise EndpointError("Invalid sampling settings") from None
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            # Without it no streaming server reports usage, and tokens are
            # never estimated.
            **({"stream_options": {"include_usage": True}} if stream else {}),
            "max_tokens": max_tokens,
            **{key: settings[key] for key in SAMPLING_KEYS if key in settings},
        }
        if tools:
            payload["tools"] = tools
        headers = {"Content-Type": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"
        headers.update(self._authorization())
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, allow_nan=False, ensure_ascii=False).encode(
                "utf-8", errors="backslashreplace"
            ),
            headers=headers,
            method="POST",
        )
        sent = time.monotonic()
        if on_send is not None:
            on_send(sent)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if stream:
                    data = self._read_stream(response, sent, on_first_token)
                else:
                    body = response.read(MAX_RESPONSE_BYTES + 1)
            if not stream:
                self.exchange = (sent, time.monotonic())
                if len(body) > MAX_RESPONSE_BYTES:
                    raise EndpointError("Response exceeds byte limit")
                data = strict_json(body.decode("utf-8"))
            elif len(_encoded(data)) > MAX_RESPONSE_BYTES:
                # Measured as the journal encodes the assembled body.
                raise EndpointError("Response exceeds byte limit")
            return parse_reply(data)
        except urllib.error.HTTPError as error:
            # With a key, more bytes than are kept, so a key that straddles
            # the kept part's end is still seen whole, also where the server
            # escaped its characters (six bytes each as JSON writes them).
            extra = max(1, 6 * len(self._api_key or ""))
            try:
                body = error.read(HTTP_BODY_BYTES + extra)
            except (OSError, http.client.HTTPException):
                body = b""
            finally:
                error.close()
            self.exchange = (sent, time.monotonic())
            evidence = {
                "http_status": error.code,
                "body": body[:HTTP_BODY_BYTES].decode("utf-8", errors="replace"),
                "body_truncated": len(body) > HTTP_BODY_BYTES,
            }
            if self._api_key is not None and holds_key(body, self._api_key):
                # A server that echoes the request's key: no part of the body
                # is kept, and the evidence keeps its shape.
                evidence["body"], evidence["body_truncated"] = "", True
            # Evidence is local data, never interpolated into the exception text.
            raise EndpointHTTPError("Endpoint HTTP error", evidence) from None
        except EndpointError:
            # A refused redirect, body or stream: the exchange ended here,
            # unless a stream's `[DONE]` already ended it.
            if self.exchange is None:
                self.exchange = (sent, time.monotonic())
            raise
        except (
            urllib.error.URLError,
            TimeoutError,
            UnicodeError,
            OSError,
            http.client.HTTPException,
        ):
            # Also a stream cut inside a chunk (IncompleteRead) or a stream
            # line that is not UTF-8.
            if self.exchange is None:
                self.exchange = (sent, time.monotonic())
            # Never include response bodies, URLs or possible credentials in errors.
            raise EndpointError("Endpoint transport failed") from None

    def _authorization(self) -> dict[str, str]:
        """The bearer header of a request, or nothing without a key."""
        if self._api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    def _read_stream(
        self,
        response: Any,
        sent: float,
        on_first_token: Callable[[float], None] | None,
    ) -> dict[str, Any]:
        """Read one event stream up to `data: [DONE]`; returns the assembled body.

        Strict dialect: lines end in LF or CRLF;
        `data:` fields and comments count, `id:` and `retry:` fields are
        ignored (their bytes still count), any other field and any line
        without a colon is "Malformed stream".
        Without `[DONE]` a stream is incomplete, whatever it carried.
        """
        if response.headers.get_content_type() != "text/event-stream":
            # A server ignoring `stream` sends a body `parse_reply` may accept,
            # so it can never be evidence of a refused stream.
            raise EndpointError("Endpoint did not stream")
        assembler = _StreamAssembler()
        data: list[bytes] = []
        total = size = 0
        while True:
            line = response.readline(
                min(MAX_RESPONSE_BYTES, MAX_STREAM_BYTES - total) + 1
            )
            total += len(line)
            if total > MAX_STREAM_BYTES or len(line) > MAX_RESPONSE_BYTES:
                raise EndpointError("Response exceeds byte limit")
            if not line.endswith(b"\n"):
                raise EndpointError("Incomplete stream")
            line = line[:-1].removesuffix(b"\r")
            if b"\r" in line:
                # A CR-only line end, which the SSE standard allows.
                raise EndpointError("Malformed stream")
            if line:
                name, colon, value = line.partition(b":")
                if not colon:
                    # Only fields with a colon count.
                    raise EndpointError("Malformed stream")
                if name == b"data":
                    value = value.removeprefix(b" ")
                    # Joined data of one event, like one non-streaming body.
                    size += len(value) + bool(data)
                    if size > MAX_RESPONSE_BYTES:
                        raise EndpointError("Response exceeds byte limit")
                    data.append(value)
                elif name in (b"", b"id", b"retry"):
                    line.decode("utf-8")  # A comment (keepalive) or ignored.
                else:
                    raise EndpointError("Malformed stream")
                continue
            # A blank line dispatches the pending event.
            moment = time.monotonic()
            event = b"\n".join(data)
            data, size = [], 0
            if not event:
                continue
            if event == b"[DONE]":
                self.exchange = (sent, moment)
                return assembler.body()
            chunk = strict_json(event.decode("utf-8"))
            # Output arrived even if the chunk then fails assembly.
            if self.first_token is None and has_token(_first_delta(chunk)):
                self.first_token = moment
                if on_first_token is not None:
                    on_first_token(moment)
            assembler.add(chunk)

    def get_json(self, path: str, *, root: bool, max_bytes: int) -> Any:
        """One read-only GET below the base (or its root without /v1); no retry."""
        check_probe_path(path)
        if type(max_bytes) is not int or max_bytes < 1:
            raise EndpointError("Positive byte limit required")
        base = self.base_url
        if root:
            if not urllib.parse.urlsplit(base).path.rstrip("/").endswith("/v1"):
                raise EndpointError("Endpoint base is not a /v1 path")
            base = base.removesuffix("/v1")
        request = urllib.request.Request(
            base + "/" + path,
            headers={"Accept": "application/json", **self._authorization()},
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read(max_bytes + 1)
        except urllib.error.HTTPError as error:
            error.close()
            # Only the status is evidence; probe bodies are never retained.
            raise EndpointError(
                "Endpoint HTTP error", {"http_status": error.code}
            ) from None
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ):
            # Never include response bodies, URLs or possible credentials in errors.
            raise EndpointError("Endpoint transport failed") from None
        if len(body) > max_bytes:
            raise EndpointError("Response exceeds byte limit")
        try:
            return strict_json(body.decode("utf-8"))
        except (UnicodeError, EndpointError):
            raise EndpointError("Invalid JSON") from None


def json_nesting(text: str) -> int:
    """Bracket depth of JSON text outside its string literals, as deep as a
    parser of that text recurses."""
    depth = deepest = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif char in "]}":
            depth -= 1
    return deepest


def is_http_evidence(raw: Any) -> TypeGuard[dict[str, Any]]:
    """Exactly the evidence `ChatEndpoint.complete` builds for an HTTP error."""
    return (
        isinstance(raw, dict)
        and set(raw) == {"http_status", "body", "body_truncated"}
        and type(raw["http_status"]) is int
        and isinstance(raw["body"], str)
        and len(raw["body"]) <= HTTP_BODY_BYTES
        and type(raw["body_truncated"]) is bool
    )


def is_capacity_rejection(raw: Any) -> bool:
    """Recognize explicit context/output request-capacity rejection, not generic 400s.

    `raw` is `EndpointHTTPError` evidence; replayed evidence is checked data
    too, so a malformed status is no rejection.
    """
    status = raw.get("http_status") if isinstance(raw, dict) else None
    if type(status) is not int or status not in {400, 413}:
        return False
    if raw.get("body_truncated"):
        return False
    text = raw.get("body", "")
    if not isinstance(text, str) or json_nesting(text) > CAPACITY_BODY_NESTING:
        return False
    try:
        body = strict_json(text)
    except EndpointError:
        return False
    error = body.get("error") if isinstance(body, dict) else None
    if (
        not isinstance(error, dict)
        and isinstance(body, dict)
        and body.get("object") == "error"
    ):
        error = body
    if not isinstance(error, dict):
        return False
    if error.get("code") in ("context_length_exceeded", "max_context_length_exceeded"):
        return True
    if error.get("type") == "exceed_context_size_error":
        return True
    message = error.get("message")
    if (
        isinstance(message, str)
        and "max_tokens" in message.lower()
        and "too large" in message.lower()
    ):
        return True
    return (
        isinstance(message, str)
        and "maximum context length" in message.lower()
        and ("exceed" in message.lower() or "you requested" in message.lower())
    )
