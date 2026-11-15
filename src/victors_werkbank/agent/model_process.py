"""One HTTP request per disposable worker; parent enforces total wall deadline.

An API key reaches a worker as one of its spawn arguments (pickled to the
worker's pipe), never on a command line, in the environment or in a file.
"""

import copy
import math
import multiprocessing
import time
from multiprocessing.connection import Connection
from typing import Any, Callable

from .measurements import DEFAULT_SAMPLING, validate_sampling
from .transport import (
    ChatEndpoint,
    EndpointError,
    EndpointHTTPError,
    Reply,
    check_probe_path,
)

# Error classes a request worker reports by name; the parent re-raises them.
_ERRORS: dict[str, type[EndpointError]] = {
    "http": EndpointHTTPError,
    "endpoint": EndpointError,
}


def _error_kind(error: EndpointError) -> str:
    return "http" if isinstance(error, EndpointHTTPError) else "endpoint"


def _send(connection: Connection, message: Any, fallback: Any) -> None:
    """Send `message`, or the fixed `fallback` if it cannot be pickled (a
    nesting beyond the pickle limit); nothing once the parent stopped reading,
    at its deadline, so a worker prints no traceback. The parent classifies a
    missing reply as it always does."""
    try:
        connection.send(message)
    except OSError:
        return
    except Exception:
        try:
            connection.send(fallback)
        except Exception:
            return


class DeadlineExceeded(TimeoutError):
    """Parent wall deadline won; partial response evidence exists only if received."""

    def __init__(self, message: str, raw: Any = None):
        super().__init__(message)
        self.raw = raw


def _request(
    connection: Connection,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
    sampling: dict[str, Any],
    started: Any,
    stream: bool,
    first: Any,
    api_key: str | None = None,
) -> None:
    # The request window brackets the network exchange only, not encoding,
    # parsing, worker start, IPC or teardown (same system-wide monotonic clock
    # as the parent). `started` and `first` are shared, so the parent knows
    # the send time and a stream's first-token time without a reply.
    sent = time.monotonic()
    endpoint: ChatEndpoint | None = None

    def window() -> tuple[float, float]:
        measured = getattr(endpoint, "exchange", None)
        return measured if measured is not None else (sent, time.monotonic())

    def on_send(moment: float) -> None:
        started.value = moment

    def on_first_token(moment: float) -> None:
        first.value = moment

    try:
        endpoint = ChatEndpoint(base_url, model, timeout=timeout + 5.0, api_key=api_key)
        sent = time.monotonic()
        reply = endpoint.complete(
            messages,
            max_tokens=max_tokens,
            tools=tools,
            sampling=sampling,
            on_send=on_send,
            stream=stream,
            on_first_token=on_first_token,
        )
        message: tuple[bool, Any] = (True, (reply, window()))
    except EndpointError as error:
        message = (False, (_error_kind(error), str(error), error.raw, window()))
    except Exception:
        message = (False, ("endpoint", "Model worker failed", None, window()))
    try:
        _send(
            connection,
            message,
            (False, ("endpoint", "Model worker failed", None, window())),
        )
    finally:
        connection.close()


def _probe(
    connection: Connection,
    base_url: str,
    model: str,
    path: str,
    root: bool,
    max_bytes: int,
    timeout: float,
    api_key: str | None = None,
) -> None:
    try:
        value = ChatEndpoint(
            base_url, model, timeout=timeout + 5.0, api_key=api_key
        ).get_json(path, root=root, max_bytes=max_bytes)
        message: tuple[bool, Any] = (True, value)
    except EndpointError as error:
        message = (False, (str(error), error.raw))
    except Exception:
        message = (False, ("Endpoint probe worker failed", None))
    try:
        _send(connection, message, (False, ("Endpoint probe worker failed", None)))
    finally:
        connection.close()


def _exchange(
    target: Callable[..., None],
    args: tuple[Any, ...],
    timeout: float,
    *,
    deadline_message: str,
    failure_message: str,
    late_raw: Callable[[bool, Any], Any],
    marks: dict[str, float] | None = None,
) -> tuple[bool, Any]:
    """Run one disposable spawn worker; the parent alone owns the total deadline.

    `marks["stopped"]` records when a deadline or a dead worker was detected
    without a reply, before the worker's teardown.
    """
    deadline = time.monotonic() + timeout
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, *args))
    try:
        process.start()
        sender.close()
        if not receiver.poll(max(0, deadline - time.monotonic())):
            if marks is not None:
                marks["stopped"] = time.monotonic()
            raise DeadlineExceeded(deadline_message)
        try:
            success, reply = receiver.recv()
        except EOFError as error:
            if marks is not None:
                marks["stopped"] = time.monotonic()
            raise EndpointError(failure_message) from error
        if time.monotonic() > deadline:
            raise DeadlineExceeded(deadline_message, late_raw(success, reply))
        return success, reply
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            process.join(timeout=0.05)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            if not process.is_alive():
                process.close()


class IsolatedEndpoint:
    """No retry; HTTP redirects/proxies remain disabled by ChatEndpoint.

    With `streaming`, every request is streamed (`--stream`). Only a complete
    stream is a body: one still running at the deadline, or aborted, leaves no
    body and no tokens, but its first-token time.

    With `api_key`, every request and probe of this endpoint carries it as a
    bearer token. It is held for the workers only: nothing records it or
    whether one is set, and no text of this object shows it.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        sampling: dict[str, Any] | None = None,
        *,
        streaming: bool = False,
        api_key: str | None = None,
    ):
        if type(streaming) is not bool:
            raise ValueError("Streaming mode must be a boolean")
        # Validate before any worker or connection.
        ChatEndpoint(base_url, model, api_key=api_key)
        self.base_url = base_url
        self.model = model
        self._api_key = api_key
        # Monotonic (sent, received) of the last request's HTTP exchange as the
        # worker measured it; None when no worker reported one.
        self.request_window: tuple[float, float] | None = None
        # Monotonic moment the last request's stream delivered its first
        # output, whatever the outcome; None unstreamed or without output.
        self.first_token: float | None = None
        # Sent with every request (none by default: the server's own default
        # applies) and recorded by the runner as run provenance; a deep copy,
        # so no caller can change the template variables after admission
        # recorded them.
        self.sampling = validate_sampling(
            dict(DEFAULT_SAMPLING) if sampling is None else copy.deepcopy(sampling)
        )
        # Every request streamed or none; recorded by the runner as provenance.
        self.streaming = streaming

    def __repr__(self) -> str:
        # Fixed: neither the key nor the address can reach a log or a message
        # through this object's text.
        return f"<{type(self).__name__}>"

    @property
    def keyed(self) -> bool:
        """Whether requests carry an API key; for advice on a refused
        request only, never recorded."""
        return self._api_key is not None

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_tokens: int,
        timeout: float,
    ) -> Reply:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive finite model deadline required")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Positive integer token budget required")
        self.request_window = None
        self.first_token = None
        context = multiprocessing.get_context("spawn")
        started = context.Value("d", -1.0, lock=False)
        first = context.Value("d", -1.0, lock=False)
        marks: dict[str, float] = {}

        def late(ok: bool, value: Any) -> Any:
            self._record_window(value[1] if ok else value[3])
            return value[0].raw if ok else value[2]

        try:
            success, value = _exchange(
                _request,
                (
                    self.base_url,
                    self.model,
                    messages,
                    tools,
                    max_tokens,
                    timeout,
                    self.sampling,
                    started,
                    self.streaming,
                    first,
                    self._api_key,
                ),
                timeout,
                deadline_message="Model wall deadline exceeded",
                failure_message="Model worker failed",
                late_raw=late,
                marks=marks,
            )
        except (EndpointError, DeadlineExceeded):
            # No reply: from the worker's send (or, unsent, nothing) to the
            # moment the parent stopped waiting, never its start or teardown.
            stopped = marks.get("stopped")
            if self.request_window is None and stopped is not None:
                begin = started.value
                self._record_window((begin if begin >= 0 else stopped, stopped))
            raise
        finally:
            # Read after the worker's teardown, so it no longer changes; kept
            # whatever the outcome, even when the stream left no body.
            moment = first.value
            if math.isfinite(moment) and moment >= 0:
                self.first_token = moment
        if not success:
            kind, message, raw, window = value
            self._record_window(window)
            raise _ERRORS[kind](message, raw)
        reply, window = value
        if not isinstance(reply, Reply):
            raise EndpointError("Model worker contract failed")
        self._record_window(window)
        return reply

    def _record_window(self, window: Any) -> None:
        if (
            isinstance(window, tuple)
            and len(window) == 2
            and all(type(t) is float and math.isfinite(t) for t in window)
            and window[0] <= window[1]
        ):
            self.request_window = window

    def get_json(self, path: str, *, root: bool, max_bytes: int, timeout: float) -> Any:
        """One read-only GET in one disposable worker under a total deadline."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive finite probe deadline required")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Positive integer byte limit required")
        check_probe_path(path)
        success, reply = _exchange(
            _probe,
            (
                self.base_url,
                self.model,
                path,
                bool(root),
                max_bytes,
                timeout,
                self._api_key,
            ),
            timeout,
            deadline_message="Endpoint probe deadline exceeded",
            failure_message="Endpoint probe worker failed",
            late_raw=lambda ok, reply: reply if ok else reply[1],
        )
        if not success:
            message, raw = reply
            raise EndpointError(message, raw)
        return reply
