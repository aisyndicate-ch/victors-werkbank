"""`werkbank preflight`: whether a server can do what a run needs, before one.

It reads the endpoint fingerprint a run reads (one `GET {endpoint}/models`, at
most one engine probe and, for an endpoint on loopback, the local GPU read)
and sends at most two chat requests through the run's own endpoint, with the
run's sampling, streaming mode and per-response cap: one asking for a single
native tool call, and, only after it passed, one returning that call and its
result in the history exactly as the episode loop builds it. Each outcome is
named with the status a run records for it (the loop's classification) or,
where a run carries on, as a finding of its own. The chat template variables
a run would send go with both requests; how many replies returned reasoning
apart from their answer is information only, never a finding, and the level
is not checked. It writes no file, and its output holds fixed texts and
server-reported numbers and names only, never a response body, model output,
the endpoint URL or a credential: of an API key (`WERKBANK_API_KEY`) only
the advice differs, by whether one was sent.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from .agent.episode import RESPONSE_BYTES, harness_digest
from .agent.loop import Endpoint, json_bytes
from .agent.measurements import (
    SERVER_DEFAULT,
    identity_sampling,
    reasoning_seen,
    template_text,
)
from .agent.transport import (
    EndpointError,
    EndpointHTTPError,
    Reply,
    is_capacity_rejection,
)
from .domains.tools import tool
from .registry import CASE_SETS, CASES, case_set_name, context_recommendation
from .serving import NOT_LISTED, NOT_PROBED, collect_serving, context_tokens

# Model-visible contract values: the one prompt, tool and tool result sent.
TOOL_NAME = "record_check"
PROMPT = (
    "This is a connection check. Call the tool record_check once, with code"
    ' set to "ready". When its result arrives, reply with the word done and'
    " call no tool."
)
TOOLS = [
    tool(TOOL_NAME, "Record the connection check's code.", {"code": {"type": "string"}})
]
TOOL_RESULT = {"recorded": True}
# Each chat request's total deadline: long enough for a reasoning model's
# short answer at the slowest measured decode rate, short enough to wait for.
REQUEST_SECONDS = 300.0
MAX_REQUESTS = 2

CHECKS = ("tool_call", "token_count", "tool_result")
LABELS = {
    "tool_call": "Tool call (request 1)",
    "token_count": "Token count",
    "tool_result": "Tool result (request 2)",
}
# Servers the hints name, in the order every hint line lists them; the
# runtimes the fingerprint identifies map onto the first three.
SERVERS = {
    "vllm": "vLLM",
    "llama.cpp": "llama.cpp",
    "sglang": "SGLang",
    "ollama": "Ollama",
    "lmstudio": "LM Studio",
}
ANY = "any"
# Advice per finding under a topic: one line per server where the fix differs
# by server, else one line for any server. Fixed texts, not a tested claim
# for every server version.
HINTS: dict[str, tuple[str, dict[str, str]]] = {
    "tool_parser": (
        "Tool calls",
        {
            "vllm": "start vllm serve with --enable-auto-tool-choice and the"
            " --tool-call-parser made for the model",
            "llama.cpp": "start llama-server with --jinja, and with"
            " --chat-template-file where the model's own chat template has no"
            " tool support",
            "sglang": "start the server with the --tool-call-parser made for the model",
            "ollama": "use a model whose Ollama template supports tools",
            "lmstudio": "use a model that LM Studio lists with tool-use support",
        },
    ),
    "finish_reason": (
        "Finish reason",
        {
            ANY: "a native tool call must come with finish_reason tool_calls,"
            " and finish_reason tool_calls with a call; update the server or"
            " choose the tool-call parser made for the model",
        },
    ),
    "usage": (
        "Token count",
        {
            ANY: "every reply needs an integer usage.completion_tokens: point"
            " --endpoint at the server itself, not at a gateway that drops"
            " usage; with --stream the server must honour"
            " stream_options.include_usage",
        },
    ),
    "capacity": (
        "Request size",
        {
            ANY: "the per-response cap and the prompt must fit the context:"
            " lower --max-tokens-per-call or serve a larger context",
        },
    ),
    "context": (
        "Context",
        {
            "vllm": "set it with --max-model-len",
            "llama.cpp": "set it with --ctx-size; with --parallel N one request"
            " gets one slot's share of it",
            "sglang": "set it with --context-length",
            "ollama": "set it with the server's OLLAMA_CONTEXT_LENGTH or the"
            " model's num_ctx parameter",
            "lmstudio": "set it as the context length in the model's load settings",
        },
    ),
    "transport": (
        "Connection",
        {
            ANY: "check that the server is running and that --endpoint is its"
            " base URL ending in /v1; host proxy settings are ignored",
        },
    ),
    "redirect": (
        "Redirect",
        {
            ANY: "give the base URL the server answers on itself; redirects are"
            " never followed",
        },
    ),
    "credentials": (
        "Credentials",
        {
            ANY: "the server asks for an API key: set the environment variable"
            " WERKBANK_API_KEY to it, or serve without a key (werkbank start"
            " sends a key only to a server named with --endpoint)",
        },
    ),
    "credentials_refused": (
        "Credentials",
        {
            ANY: "the server refused the API key sent from WERKBANK_API_KEY: set"
            " it to the key this server was started with",
        },
    ),
    "model_name": (
        "Model name",
        {ANY: "--model must be the exact name the server's /v1/models lists"},
    ),
    "server_log": (
        "Server log",
        {ANY: "the server's own log names why it refused or failed the request"},
    ),
    "deadline": (
        "Deadline",
        {
            ANY: "the server may still be busy, also with an earlier request (a"
            " stopped client does not cancel a generation), or decode slowly;"
            " check again once it is idle",
        },
    ),
    "stream": (
        "Streaming",
        {
            ANY: "with --stream the server must answer with Content-Type"
            " text/event-stream and end the stream with data: [DONE]; leave out"
            " --stream if it cannot",
        },
    ),
    "length": (
        "Response length",
        {
            ANY: "the reply hit the per-response cap before a tool call: raise"
            " --max-tokens-per-call (a reasoning model needs room to think)",
        },
    ),
    "template": (
        "Chat template settings",
        {
            ANY: "the server refused chat_template_kwargs: leave out"
            " --chat-template-kwargs or --thinking, or give only variables the"
            " model's chat template reads",
        },
    ),
    "response": (
        "Response",
        {
            ANY: "--endpoint must be an OpenAI-compatible /v1 base URL that"
            " answers chat completions",
        },
    ),
}
# Fixed transport texts in plain words, with the hint they call for.
_ERRORS = {
    "Endpoint transport failed": (
        "the server could not be reached or broke off the connection",
        "transport",
    ),
    "Endpoint redirects are not supported": (
        "the server answered with a redirect, which is never followed",
        "redirect",
    ),
    "Endpoint did not stream": (
        "a streamed request was answered without an event stream",
        "stream",
    ),
    "Incomplete stream": ("the event stream ended before data: [DONE]", "stream"),
    "Malformed stream": ("the event stream is malformed", "stream"),
    "Endpoint stream error": ("the event stream carried an error", "stream"),
    "Tool calls without matching finish reason": (
        "the reply holds native tool calls without finish_reason tool_calls",
        "finish_reason",
    ),
    "Empty native tool call result": (
        "the reply has finish_reason tool_calls but no tool call",
        "finish_reason",
    ),
    "Unsupported finish reason": (
        "the reply has a finish_reason a run does not accept",
        "finish_reason",
    ),
    "Invalid native tool call": (
        "a native tool call does not have the documented shape",
        "tool_parser",
    ),
    "Empty tool identity": (
        "a native tool call has an empty ID or name",
        "tool_parser",
    ),
    "Duplicate tool call ID": (
        "two native tool calls share one ID",
        "tool_parser",
    ),
    "Model worker failed": ("the request's worker process failed", None),
}
# A reply status in plain words, with the hint it calls for.
_REPLY_TEXTS = {
    "response_byte_budget": (
        f"the reply is larger than the {RESPONSE_BYTES // (1024 * 1024)} MiB a"
        " run accepts",
        "response",
    ),
    "response_token_budget": (
        "the reply ended at the per-response cap (finish_reason length)",
        "length",
    ),
    "endpoint_refusal": (
        "the server refused the reply (finish_reason content_filter)",
        None,
    ),
}


@dataclass(frozen=True)
class Finding:
    """One check: passed (True), failed (False) or not checked (None); the
    status a run records for the same outcome, None where a run carries on or
    records another check's status for the same reply; the hints it calls
    for."""

    passed: bool | None
    text: str
    status: tuple[str, str] | None = None
    hints: tuple[str, ...] = ()


@dataclass
class Preflight:
    """The findings. For all registered cases and, by name, for each case
    set: the recommended context and the cases whose context need with the
    checked cap exceeds the reported context (none when it is unknown)."""

    serving: dict[str, Any]
    context_tokens: int | None
    recommended_context: int
    cases_needing_more_context: list[str]
    case_sets: dict[str, tuple[int, list[str]]]
    checks: dict[str, Finding] = field(default_factory=dict)
    requests: int = 0
    deadline: bool = False
    # The template variables sent, `server default` without any.
    thinking: str = SERVER_DEFAULT
    # Replies that returned reasoning apart from their answer, and replies;
    # information only.
    reasoning: tuple[int, int] = (0, 0)


def error_status(error: BaseException, responses: int) -> tuple[str, str]:
    """The status the episode loop's `classify` gives `error` raised by a
    model request in the agent phase after `responses` model responses;
    anything it does not classify is raised again."""
    if isinstance(error, TimeoutError):
        return "timeout", "agent_deadline"
    if isinstance(error, EndpointHTTPError):
        if not is_capacity_rejection(error.raw):
            return "invalid", "endpoint_contract_error"
        if responses == 0:
            return "unsupported", "endpoint_request_capacity"
        return "output_limit", "endpoint_capacity_rejection_after_responses"
    if isinstance(error, EndpointError):
        return "invalid", "endpoint_contract_error"
    if isinstance(error, (OSError, RuntimeError, ValueError)):
        return "invalid", "infrastructure_or_contract_error"
    raise error


def _tokens(reply: Reply) -> Any:
    return (reply.usage or {}).get("completion_tokens")


def counted(reply: Reply) -> bool:
    """The reply reports its generated tokens as a run counts them."""
    tokens = _tokens(reply)
    return type(tokens) is int and tokens >= 0


def _shape_status(reply: Reply) -> tuple[str, str] | None:
    if len(json_bytes(reply.raw)) > RESPONSE_BYTES:
        return "output_limit", "response_byte_budget"
    if reply.finish_reason == "length":
        return "output_limit", "response_token_budget"
    if reply.finish_reason == "content_filter":
        return "failed", "endpoint_refusal"
    return None


def reply_status(reply: Reply) -> tuple[str, str] | None:
    """The status the episode loop records for `reply` as a response within
    the episode's token budget and deadline (the budget is far above one
    per-response cap); None where the episode carries on."""
    if not counted(reply):
        return "unsupported", "completion_token_accounting_missing"
    return _shape_status(reply)


def _describe_error(
    error: BaseException, keyed: bool = False
) -> tuple[str, tuple[str, ...]]:
    """Plain words and hints for a failed request, from fixed texts and the
    HTTP status only; a body is matched, never shown. `keyed` says whether
    the request carried an API key, which decides the advice on HTTP 401 and
    403 only."""
    if isinstance(error, TimeoutError):
        return f"no complete reply within {REQUEST_SECONDS:.0f} seconds", ("deadline",)
    if isinstance(error, EndpointHTTPError):
        raw = error.raw if isinstance(error.raw, dict) else {}
        status = raw.get("http_status")
        text = f"the server answered HTTP status {status}"
        body = raw.get("body")
        if is_capacity_rejection(raw):
            return text + " and refused the request's size", ("capacity", "context")
        if status in (401, 403):
            return text, ("credentials_refused" if keyed else "credentials",)
        if status == 404:
            return text, ("model_name", "transport")
        if isinstance(body, str) and "chat_template_kwargs" in body:
            return text, ("template", "server_log")
        if isinstance(body, str) and "tool" in body.lower():
            return text, ("tool_parser", "server_log")
        return text, ("server_log",)
    if isinstance(error, EndpointError):
        message = str(error)
        known = _ERRORS.get(message)
        if known is not None:
            text, hint = known
            return text, (hint,) if hint else ()
        return f"the reply is not a chat completion a run accepts ({message})", (
            "response",
        )
    return f"the request could not be made ({type(error).__name__})", ()


def _run_text(status: tuple[str, str]) -> str:
    if status[1] == "agent_deadline":
        return "; an episode allows far longer, so readiness is not confirmed"
    stops = " and stops" if status[0] == "invalid" else ""
    return f"; a run records {status[0]}/{status[1]}{stops}"


def _failed_request(
    error: BaseException, responses: int, keyed: bool = False
) -> Finding:
    status = error_status(error, responses)
    text, hints = _describe_error(error, keyed)
    return Finding(False, f"failed: {text}{_run_text(status)}", status, hints)


def _failed_reply(reply: Reply) -> Finding | None:
    """A failed finding for a reply a run ends on for its size, length or
    refusal, or None. A run records a reply without an integer token count as
    that, before any of these, so the token count's finding then carries the
    run's status and this one names none."""
    shape = _shape_status(reply)
    if shape is None:
        return None
    text, hint = _REPLY_TEXTS[shape[1]]
    hints = (hint,) if hint else ()
    status = reply_status(reply)
    if status != shape:
        return Finding(False, f"failed: {text}", None, hints)
    return Finding(False, f"failed: {text}{_run_text(shape)}", shape, hints)


def _tool_call(reply: Reply) -> Finding:
    """Request 1's tool call: native, finish_reason tool_calls (the transport
    refuses any other pairing), the offered tool, JSON-object arguments."""
    failed = _failed_reply(reply)
    if failed is not None:
        return failed
    if not reply.tool_calls:
        return Finding(
            False,
            "failed: the reply holds no native tool call; in a run no tool call"
            " would reach a case",
            None,
            ("tool_parser",),
        )
    if any(call.name != TOOL_NAME for call in reply.tool_calls):
        return Finding(
            False,
            "failed: a native tool call names a tool that was not offered; in a"
            " run such calls only return tool errors",
            None,
            ("tool_parser",),
        )
    for call in reply.tool_calls:
        try:
            call.arguments()
        except EndpointError:
            return Finding(
                False,
                "failed: a tool call's arguments are not a JSON object; in a run"
                " such calls only return tool errors",
                None,
                ("tool_parser",),
            )
    return Finding(
        True,
        f"passed: a native call of {TOOL_NAME} with finish_reason tool_calls and"
        " JSON-object arguments",
    )


def history(reply: Reply) -> list[dict[str, Any]]:
    """Request 2's messages as the episode loop builds them: the prompt, the
    assistant message verbatim (native reasoning fields included) and one tool
    message per call."""
    return [
        {"role": "user", "content": PROMPT},
        reply.raw["choices"][0]["message"],
        *(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(TOOL_RESULT, allow_nan=False),
            }
            for call in reply.tool_calls
        ),
    ]


def _token_count(replies: list[Reply]) -> Finding:
    if not all(counted(reply) for reply in replies):
        status = ("unsupported", "completion_token_accounting_missing")
        return Finding(
            False,
            "failed: a reply has no integer usage.completion_tokens"
            + _run_text(status),
            status,
            ("usage",),
        )
    counts = " and ".join(str(_tokens(reply)) for reply in replies)
    return Finding(True, f"passed: integer usage.completion_tokens ({counts})")


def check(
    endpoint: Endpoint,
    *,
    max_tokens_per_call: int,
    serving_note: dict[str, Any] | None = None,
    hardware_label: str | None = None,
) -> Preflight:
    """Read the fingerprint, then send at most two chat requests, each with
    `max_tokens_per_call` as max_tokens, under `REQUEST_SECONDS`; no retry.
    The context lines use the cases' needs with that cap."""
    sampling = getattr(endpoint, "sampling", None)
    keyed = getattr(endpoint, "keyed", False) is True
    serving = collect_serving(
        endpoint,
        endpoint.model,
        serving_note,
        configuration={
            "sampling": identity_sampling(sampling or {}),
            "max_tokens_per_call": max_tokens_per_call,
            "harness_sha256": harness_digest(),
            "streaming": getattr(endpoint, "streaming", False),
        },
        hardware_label=hardware_label,
    )
    context = context_tokens(serving)
    recommended, short = context_recommendation(
        tuple(CASES), context, max_tokens_per_call=max_tokens_per_call
    )
    case_sets = {
        name: context_recommendation(
            case_ids, context, max_tokens_per_call=max_tokens_per_call
        )
        for name, case_ids in CASE_SETS.items()
    }
    result = Preflight(serving, context, recommended, short, case_sets)
    result.thinking = template_text(sampling or {})
    checks = result.checks
    replies: list[Reply] = []
    messages: list[dict[str, Any]] = [{"role": "user", "content": PROMPT}]
    for responses in range(MAX_REQUESTS):
        result.requests += 1
        name = "tool_call" if responses == 0 else "tool_result"
        try:
            reply = endpoint.complete(
                messages,
                tools=TOOLS,
                max_tokens=max_tokens_per_call,
                timeout=REQUEST_SECONDS,
            )
        except Exception as error:
            checks[name] = _failed_request(error, responses, keyed)
            result.deadline = isinstance(error, TimeoutError)
            break
        replies.append(reply)
        if responses == 0:
            checks[name] = _tool_call(reply)
        else:
            checks[name] = _failed_reply(reply) or Finding(
                True,
                "passed: the server accepted the tool call and its result in the"
                " history",
            )
        if not (checks[name].passed and counted(reply)):
            break
        messages = history(reply)
    seen = [reasoning_seen(reply.raw) for reply in replies]
    result.reasoning = (seen.count(True), sum(item is not None for item in seen))
    checks["token_count"] = (
        _token_count(replies)
        if replies
        else Finding(None, "not checked: no reply arrived")
    )
    checks.setdefault(
        "tool_result",
        Finding(
            None, "not checked: sent only after a passed tool call and token count"
        ),
    )
    result.checks = {name: checks[name] for name in CHECKS}
    return result


def exit_code(result: Preflight) -> int:
    """0: every check passed; 2: a run would stop (a run records `invalid`);
    3: cases would be lost or readiness is not confirmed. The context and the
    model list never change it."""
    findings = result.checks.values()
    if any(f.status is not None and f.status[0] == "invalid" for f in findings):
        return 2
    return 0 if all(f.passed for f in findings) else 3


def _names(case_ids: list[str]) -> str:
    if len(case_ids) < 2:
        return "".join(case_ids)
    return f"{', '.join(case_ids[:-1])} and {case_ids[-1]}"


def _server_line(serving: dict[str, Any]) -> str:
    engine = serving["engine"]
    runtime = engine["runtime"]
    if runtime is None:
        return "Server: not identified"
    name = SERVERS.get(runtime, runtime)
    if engine["reason"] is not None:
        return f"Server: {name} (engine probe: {engine['reason']})"
    version = engine["version"]
    return f"Server: {name} {version}" if version else f"Server: {name}"


def _models_line(serving: dict[str, Any]) -> str:
    reason = serving["models"]["reason"]
    if reason is None:
        return "Model list: lists the model"
    if reason == NOT_LISTED:
        return "Model list: the model is not listed"
    if reason == NOT_PROBED:
        return "Model list: not read"
    return f"Model list: not read ({reason})"


ALL_CASES = "all cases"


def _scopes(result: Preflight) -> list[tuple[str, int, list[str]]]:
    """Each case set, then all registered cases unless a case set holds them
    all: its name, recommended context and the cases that need more than the
    reported context."""
    scopes = [(name, *coverage) for name, coverage in result.case_sets.items()]
    if case_set_name(tuple(CASES)) is None:
        scopes.append(
            (ALL_CASES, result.recommended_context, result.cases_needing_more_context)
        )
    return scopes


def _context_line(result: Preflight) -> str:
    """The reported context and, for each case set and for all registered
    cases, whether it covers every need with the checked cap or which cases
    need more, so a tester sees which selection the server can run as it
    is. A need is a measured one, so a covered selection says so: a model
    that carries more reasoning per step can need more."""
    scopes = _scopes(result)
    if result.context_tokens is None:
        each = _names([f"{need:,} for {name}" for name, need, _ in scopes])
        return (
            f"Context: not reported (recommended in tokens: {each}; a serving"
            " note's context_profile states it)"
        )
    parts = []
    covered = [name for name, _, short in scopes if not short]
    if covered:
        parts.append(
            f"covers the measured need for {_names(covered)} (a model"
            " that carries more reasoning per step can need more)"
        )
    for name, recommended, short in scopes:
        if short:
            verb = "needs" if len(short) == 1 else "need"
            parts.append(
                f"for {name} {_names(short)} {verb} more (recommended: {recommended:,})"
            )
    if result.cases_needing_more_context:
        parts.append("a case that needs more may end with a capacity rejection")
    return f"Context: {result.context_tokens:,} tokens; " + "; ".join(parts)


def hint_keys(result: Preflight) -> list[str]:
    """The hints the findings call for, each once, in the order found."""
    keys: list[str] = []
    for finding in result.checks.values():
        if finding.passed is False:
            keys.extend(finding.hints)
    if result.serving["models"]["reason"] == NOT_LISTED:
        keys.append("model_name")
    if result.context_tokens is None or result.cases_needing_more_context:
        keys.append("context")
    return list(dict.fromkeys(keys))


def hint_lines(result: Preflight) -> list[str]:
    """One line per hint: for a per-server hint only the identified server's
    line, or every server's where none is identified."""
    runtime = result.serving["engine"]["runtime"]
    lines = []
    for key in hint_keys(result):
        topic, texts = HINTS[key]
        if ANY in texts:
            lines.append(f"- {topic}: {texts[ANY]}")
            continue
        servers = [runtime] if runtime in texts else list(SERVERS)
        lines.extend(
            f"- {topic} ({SERVERS[server]}): {texts[server]}" for server in servers
        )
    return lines


REASONING_REPLIES = (
    "reasoning seen in {seen} of {replies} replies (information only; the level"
    " is not checked)"
)


def _thinking_line(result: Preflight) -> str:
    """The template variables sent and, once a reply arrived, how many
    replies returned reasoning apart from their answer."""
    line = f"Thinking: {result.thinking}"
    seen, replies = result.reasoning
    if replies:
        line += "; " + REASONING_REPLIES.format(seen=seen, replies=replies)
    return line


def render(result: Preflight) -> list[str]:
    """The plain-text summary: what was sent, the fingerprint, each check,
    the thinking line, the verdict and the hints."""
    code = exit_code(result)
    sent = "request" if result.requests == 1 else "requests"
    lines = [
        f"Preflight: {result.requests} chat {sent} sent (at most {MAX_REQUESTS})",
        _server_line(result.serving),
        _models_line(result.serving),
        _context_line(result),
        *(f"{LABELS[name]}: {f.text}" for name, f in result.checks.items()),
        _thinking_line(result),
    ]
    if code == 0:
        lines.append("Ready: yes")
    elif code == 2:
        lines.append("Ready: no, a run would stop (exit 2)")
    else:
        lines.append(
            "Ready: no, cases would be lost or readiness is not confirmed (exit 3)"
        )
    hints = hint_lines(result)
    if hints:
        lines.append("Hints (advice, not tested against every server version):")
        lines.extend(hints)
    if result.deadline:
        lines.append(
            "Note: the server may still be generating the unanswered request;"
            " wait until it is idle before a run"
        )
    return lines
