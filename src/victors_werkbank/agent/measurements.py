"""Sampling settings and per-request/per-case measurement records.

The agent writes them; the offline validation and the replay grader check them
with the same rules here. Times are seconds on the journal's clock (monotonic,
since the run started), so each request lies between the journal events around
it. Token counts are endpoint-reported and never estimated: a missing or
malformed value is `null` with a fixed reason. Each request records the moment
of its first token (`first_token_seconds`), null unless the run streams and a
token arrived before the request ended; the run's streaming mode tells the two
apart, so the record carries no reason for it. Validators take the mode as
`streaming`. The prompt tokens the endpoint reports as served from its prefix
cache (`cached_tokens`) are a token field like the others, recorded when the
run contract says so; validators take that as `cached`.
"""

import json
import math
import re
from collections.abc import Iterable
from typing import Any

# Default sampling: no setting is sent, so the server applies its own default
# (vLLM and Ollama: the model's recommended settings). Every setting is opt-in
# and sent only when set; a sampling record holds exactly the settings sent.
DEFAULT_SAMPLING: dict[str, Any] = {}
# The sampler settings proper; "server default" in every display means none
# of them is sent.
SAMPLER_KEYS = ("temperature", "top_p", "top_k")
# Chat template variables (a model's thinking switch, for example), sent as the
# request's top-level `chat_template_kwargs` exactly as given and only when
# given: never translated per engine and never resent without them. Whether
# the server's chat template reads them is the server's matter.
TEMPLATE_KWARGS = "chat_template_kwargs"
# Appended last, so the body order of the settings before it is unchanged.
SAMPLING_KEYS = (*SAMPLER_KEYS, "seed", TEMPLATE_KWARGS)
# Sampling keys that identify a configuration; the seed is provenance only, so
# seeded repetitions of one configuration pool, and the template variables
# sent are part of the setup like the sampler settings.
IDENTITY_SAMPLING_KEYS = (*SAMPLER_KEYS, TEMPLATE_KWARGS)
MAX_TEMPERATURE = 2
MAX_TOP_K = 1_000_000
# Portable seed range: llama.cpp, vLLM and SGLang all accept 32-bit values.
MAX_SEED = 2**32 - 1
# Runs a pool needs for scores, whatever its sampling: one run is one draw of
# the model and, for a case that draws its data per run, of one instance.
MIN_REPETITIONS = 3
# The template variables a run may send: a flat object of named switches and
# levels, no structure, so that no value can replace the messages or tools,
# add a sentence to a template that writes it into the prompt, or carry a
# path, address or host name into a submission. A word reading true, false or
# null is refused, as a template reads any non-empty word as true.
MAX_TEMPLATE_KEYS = 8
MAX_TEMPLATE_INTEGER = 2**31 - 1
# The canonical JSON (`template_json`) of the variables, in bytes.
MAX_TEMPLATE_BYTES = 1024
_TEMPLATE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_TEMPLATE_WORD = re.compile(r"[A-Za-z0-9_-]{1,64}")
_TEMPLATE_LITERALS = frozenset({"true", "false", "null"})
# The arguments engines and `apply_chat_template` take beside the template
# variables; none of them is a template variable a run may set.
RESERVED_TEMPLATE_KEYS = frozenset(
    {
        "messages",
        "conversation",
        "tools",
        "documents",
        "chat_template",
        "add_generation_prompt",
        "continue_final_message",
        "tokenize",
        "padding",
        "truncation",
        "max_length",
        "return_tensors",
        "return_dict",
        "return_assistant_tokens_mask",
        "tokenizer_kwargs",
    }
)
TEMPLATE_RULE = (
    f"a JSON object of 1 to {MAX_TEMPLATE_KEYS} chat template variables, each"
    " named with letters, digits and _ (at most 64, not starting with a digit,"
    " not an engine argument such as messages or tools) and set to true, false,"
    f" an integer from -{MAX_TEMPLATE_INTEGER:,} to {MAX_TEMPLATE_INTEGER:,} or"
    " one word of 1 to 64 letters, digits, _ and - (not true, false or null); no"
    f" null, fraction, list or object; at most {MAX_TEMPLATE_BYTES:,} bytes as"
    " canonical JSON"
)

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens")
# Prompt tokens served from the endpoint's prefix cache, as
# `prompt_tokens_details` reports them; measurement data only.
CACHED_TOKENS = "cached_tokens"
# Largest recorded count; a larger reported value is malformed, not recorded.
MAX_TOKEN_COUNT = 2**53 - 1
NOT_REPORTED = "not_reported"
MALFORMED = "invalid"
# An integer above MAX_TOKEN_COUNT: not recorded, but counted like any
# non-negative integer by the loop, so the case total exceeds the records.
OUT_OF_RANGE = "out_of_range"
NO_RESPONSE = "no_response"
NULL_REASONS = frozenset({NOT_REPORTED, MALFORMED, OUT_OF_RANGE, NO_RESPONSE})
# How a request ended: a parsed response, an endpoint error (no tokens), the
# wall deadline (tokens only from a late body), or any other worker failure.
OUTCOMES = frozenset({"response", "endpoint_error", "deadline", "failed"})
TIMING_KEYS = ("model_seconds", "tool_seconds", "verifier_seconds", "other_seconds")
# Documented reconciliation tolerance: per-case components sum to the case
# wall time, and model time to its requests, within float rounding of this size.
TIMING_TOLERANCE_SECONDS = 0.001


def _finite(value: object) -> bool:
    try:
        return type(value) in {int, float} and math.isfinite(value)  # type: ignore[arg-type]
    except OverflowError:
        return False


def template_json(value: dict[str, Any]) -> str:
    """The canonical JSON of template variables: sorted keys, ASCII."""
    return json.dumps(value, sort_keys=True)


def _template_value(value: object) -> bool:
    if type(value) is bool:
        return True
    if type(value) is int:
        return abs(value) <= MAX_TEMPLATE_INTEGER
    return (
        type(value) is str
        and _TEMPLATE_WORD.fullmatch(value) is not None
        and value.lower() not in _TEMPLATE_LITERALS
    )


def validate_template_kwargs(value: object) -> dict[str, Any]:
    """Raise ValueError (`TEMPLATE_RULE`) unless value is a set of template
    variables a run may send; returns it. Nothing is normalised: a value
    outside the rule is refused wherever it is read."""
    if (
        type(value) is not dict
        or not 1 <= len(value) <= MAX_TEMPLATE_KEYS
        or not all(
            type(name) is str
            and _TEMPLATE_NAME.fullmatch(name) is not None
            and name not in RESERVED_TEMPLATE_KEYS
            and _template_value(item)
            for name, item in value.items()
        )
        or len(template_json(value).encode()) > MAX_TEMPLATE_BYTES
    ):
        raise ValueError(TEMPLATE_RULE)
    return value


def validate_sampling(value: object) -> dict[str, Any]:
    """Raise ValueError unless value is a sampling record; returns it. Every
    setting is optional: one that is absent is not sent, and the server
    applies its own value for it."""
    if not isinstance(value, dict) or not set(value) <= set(SAMPLING_KEYS):
        raise ValueError("Invalid sampling settings")
    if TEMPLATE_KWARGS in value:
        try:
            validate_template_kwargs(value[TEMPLATE_KWARGS])
        except ValueError:
            raise ValueError("Invalid sampling settings") from None
    temperature = value.get("temperature", 0)
    top_p = value.get("top_p", 1)
    top_k = value.get("top_k", 1)
    seed = value.get("seed", 0)
    if (
        not _finite(temperature)
        or not 0 <= temperature <= MAX_TEMPERATURE
        or not _finite(top_p)
        or not 0 < top_p <= 1
        or type(top_k) is not int
        or not 1 <= top_k <= MAX_TOP_K
        or type(seed) is not int
        or not 0 <= seed <= MAX_SEED
    ):
        raise ValueError("Invalid sampling settings")
    return value


def identity_sampling(value: dict[str, Any]) -> dict[str, Any]:
    """The sampling settings that are part of the configuration identity."""
    return {key: value[key] for key in IDENTITY_SAMPLING_KEYS if key in value}


def server_default(value: dict[str, Any]) -> bool:
    """No sampler setting is sent (a seed or template variables alone set
    none): the server's default sampling applies."""
    return not any(key in value for key in SAMPLER_KEYS)


SERVER_DEFAULT = "server default"
SERVER_DEFAULT_SEED = "server default, seed {seed}"


def sampling_text(value: dict[str, Any]) -> str:
    """Display text of sampling settings: `server default` (with the seed when
    one is sent) or the settings sent as canonical JSON; the template
    variables are shown apart (`template_text`)."""
    if not server_default(value):
        return json.dumps(
            {key: item for key, item in value.items() if key != TEMPLATE_KWARGS},
            sort_keys=True,
        )
    seed = value.get("seed")
    return SERVER_DEFAULT if seed is None else SERVER_DEFAULT_SEED.format(seed=seed)


def template_text(value: dict[str, Any]) -> str:
    """Display text of the template variables a sampling record sends:
    `server default` without any, else their canonical JSON."""
    if TEMPLATE_KWARGS not in value:
        return SERVER_DEFAULT
    return template_json(value[TEMPLATE_KWARGS])


def reasoning_seen(body: object) -> bool | None:
    """Whether a journaled response returned reasoning apart from its answer:
    a non-empty `reasoning_content` or `reasoning` text in its message, or
    reported reasoning tokens above 0; None for a body that is no response
    (an HTTP rejection, an omission marker). Information only: it never
    changes a status, and a server that returns its thinking inside the
    answer text counts as not seen."""
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict) and any(
        isinstance(message.get(key), str) and message[key] != ""
        for key in ("reasoning_content", "reasoning")
    ):
        return True
    usage = body.get("usage") if isinstance(body, dict) else None
    details = (
        usage.get(DETAILS["reasoning_tokens"]) if isinstance(usage, dict) else None
    )
    tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
    return type(tokens) is int and tokens > 0


def _count(container: object, key: str) -> tuple[int | None, str | None]:
    if not isinstance(container, dict) or key not in container:
        return None, NOT_REPORTED
    value = container[key]
    if type(value) is not int or value < 0:
        return None, MALFORMED
    if value > MAX_TOKEN_COUNT:
        return None, OUT_OF_RANGE
    return value, None


def token_fields(cached: bool = False) -> tuple[str, ...]:
    """The token fields of a request record (`cached`: with `cached_tokens`)."""
    return (*TOKEN_FIELDS, CACHED_TOKENS) if cached else TOKEN_FIELDS


# Where a detail count is reported, if not at the top level of `usage`.
DETAILS = {
    "reasoning_tokens": "completion_tokens_details",
    CACHED_TOKENS: "prompt_tokens_details",
}


def token_counts(usage: object, *, cached: bool = False) -> dict[str, Any]:
    """Endpoint-reported counts of one response; null fields carry a reason.

    Reasoning tokens come from `completion_tokens_details`, cached prompt
    tokens (with `cached`) from `prompt_tokens_details`, where reported.
    """
    values: dict[str, Any] = {}
    missing: dict[str, str] = {}
    for field in token_fields(cached):
        details: object = usage
        if field in DETAILS:
            details = usage.get(DETAILS[field]) if isinstance(usage, dict) else None
        value: int | None
        reason: str | None
        if field in DETAILS and details is not None and not isinstance(details, dict):
            value, reason = None, MALFORMED
        else:
            value, reason = _count(details, field)
        values[field] = value
        if reason is not None:
            missing[field] = reason
    return {**values, "missing": missing}


def request_record(
    sent: float,
    received: float,
    outcome: str,
    body: object = None,
    first_token: float | None = None,
    *,
    cached: bool = False,
) -> dict[str, Any]:
    """One request's measurement (with `cached`, its cached prompt tokens);
    `body` is the response it returned, or a late body
    that arrived after the deadline, whose usage counts its tokens.
    `first_token` is the moment its first streamed token arrived, None where
    the run does not stream or none arrived."""
    fields = token_fields(cached)
    if isinstance(body, dict) and outcome in {"response", "deadline"}:
        counts = token_counts(body.get("usage"), cached=cached)
    else:
        counts = {field: None for field in fields}
        counts["missing"] = {field: NO_RESPONSE for field in fields}
    return {
        "sent_seconds": sent,
        "received_seconds": received,
        FIRST_TOKEN: first_token,
        "outcome": outcome,
        **counts,
    }


def timing_record(
    elapsed: float, requests: list[dict[str, Any]], tool: float, verifier: float
) -> dict[str, float]:
    """Per-case split of the wall time; `other_seconds` is harness overhead."""
    model = sum(r["received_seconds"] - r["sent_seconds"] for r in requests)
    return {
        "model_seconds": model,
        "tool_seconds": tool,
        "verifier_seconds": verifier,
        "other_seconds": elapsed - model - tool - verifier,
    }


def validate_request(
    value: object, *, streaming: bool = False, cached: bool = False
) -> dict[str, Any]:
    """Raise ValueError unless value is a request record of a run with this
    streaming mode and, with `cached`, cached prompt tokens; returns it.

    A first token lies within its request; any outcome may have one, as a
    token can precede an endpoint error, the deadline or a worker failure.
    """
    keys = REQUEST_KEYS
    fields = token_fields(cached)
    if cached:
        keys = keys | {CACHED_TOKENS}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Invalid request measurement")
    sent, received = value["sent_seconds"], value["received_seconds"]
    if (
        not _finite(sent)
        or not _finite(received)
        or not 0 <= sent <= received
        or value["outcome"] not in OUTCOMES
    ):
        raise ValueError("Invalid request measurement")
    first = value.get(FIRST_TOKEN)
    if first is not None and (
        not streaming or not _finite(first) or not sent <= first <= received
    ):
        raise ValueError("Invalid request measurement")
    missing = value["missing"]
    if not isinstance(missing, dict) or not set(missing) <= set(fields):
        raise ValueError("Invalid request measurement")
    for field in fields:
        count = value[field]
        if count is None:
            if missing.get(field) not in NULL_REASONS:
                raise ValueError("Invalid request measurement")
        elif (
            type(count) is not int
            or not 0 <= count <= MAX_TOKEN_COUNT
            or field in missing
        ):
            raise ValueError("Invalid request measurement")
    no_body = value["outcome"] in {"endpoint_error", "failed"}
    if no_body and any(missing.get(f) != NO_RESPONSE for f in fields):
        raise ValueError("Invalid request measurement")
    if value["outcome"] == "response" and NO_RESPONSE in missing.values():
        raise ValueError("Invalid request measurement")
    return value


def validate_measurements(
    result: dict[str, Any], *, streaming: bool = False, cached: bool = False
) -> None:
    """Shape of `requests` and `timing` and their reconciliation with the
    result's wall time; raises ValueError. `streaming` is the run's mode.

    A streamed request's window also holds the worker parsing the chunks it
    read while the stream ran; model time is still the sum of the windows.
    """
    requests = result.get("requests")
    timing = result.get("timing")
    if not isinstance(requests, list) or not isinstance(timing, dict):
        raise ValueError("Missing measurement data")
    for request in requests:
        validate_request(request, streaming=streaming, cached=cached)
    if set(timing) != set(TIMING_KEYS) or not all(
        _finite(timing[key]) for key in TIMING_KEYS
    ):
        raise ValueError("Invalid case timing")
    tolerance = TIMING_TOLERANCE_SECONDS
    model = sum(r["received_seconds"] - r["sent_seconds"] for r in requests)
    elapsed = result["elapsed_seconds"]
    # The loop counts every non-negative integer: an out-of-range one adds
    # more than MAX_TOKEN_COUNT that no record holds; nothing else is hidden.
    recorded = sum(r["completion_tokens"] or 0 for r in requests)
    hidden = [r["missing"].get("completion_tokens") for r in requests].count(
        OUT_OF_RANGE
    )
    total = result["output_tokens"]
    if hidden:
        consistent = total >= recorded + (MAX_TOKEN_COUNT + 1) * hidden
    else:
        consistent = total == recorded
    if not consistent:
        raise ValueError("Request tokens differ from the case total")
    if (
        any(timing[key] < 0 for key in TIMING_KEYS[:3])
        or timing["other_seconds"] < -tolerance
        or abs(timing["model_seconds"] - model) > tolerance
        or abs(sum(timing[key] for key in TIMING_KEYS) - elapsed) > tolerance
    ):
        raise ValueError("Case timing does not reconcile with its wall time")


# Results that replace any recorded one: a failed removal of the case's
# temporary directory, which the runner classifies as an infrastructure error
# after `classify` recorded the episode's end, and the cleanup deadline, which
# `finalize` applies last.
CLEANUP_DEADLINE = ("timeout", "cleanup_deadline")
REMOVAL_FAILED = ("invalid", "infrastructure_or_contract_error")
RELABELS = frozenset({REMOVAL_FAILED, CLEANUP_DEADLINE})
# Results `classify` records with the error record it journals in a response's
# place, or a relabel. A `rejected_raw` record holds a body `parse_reply`
# refused, a contract error (the endpoint raises no model action error; tool
# arguments raise one inside the loop); a `rejected_http` record the transport's
# HTTP error evidence, a capacity rejection exactly when `is_capacity_rejection`
# says so (which reason, the responses before it decide: report validation
# checks that on its own), else a contract error; a `late_raw` record a late
# body after the deadline.
CONTRACT_RESULTS = RELABELS | {("invalid", "endpoint_contract_error")}
CAPACITY_RESULTS = RELABELS | {
    ("unsupported", "endpoint_request_capacity"),
    ("output_limit", "endpoint_capacity_rejection_after_responses"),
}
LATE_RESULTS = RELABELS | {("timeout", "agent_deadline")}
# An omitted record cannot say which record it replaced, only the request
# outcome does: a rejected record ends an endpoint error, a late one a deadline.
OUTCOME_RESULTS = {
    "endpoint_error": CONTRACT_RESULTS | CAPACITY_RESULTS,
    "deadline": LATE_RESULTS,
}
# Request outcomes of the record each marker in a response's place replaces:
# an oversized counted response or late body; a rejected or late record (the
# emergency path); any record (the evidence quota).
MARKER_OUTCOMES = {
    "response_byte_budget": frozenset({"response", "deadline"}),
    "segment_byte_budget": frozenset({"endpoint_error", "deadline"}),
    "case_journal_byte_budget": frozenset({"response", "endpoint_error", "deadline"}),
}
# The only marker the runner writes in a request's place.
REQUEST_MARKERS = frozenset({"case_journal_byte_budget"})


def is_marker(payload: object, reasons: Iterable[str] = MARKER_OUTCOMES) -> bool:
    """Exactly the omission marker shape the sink writes, with one of `reasons`;
    no endpoint body has it (a late one is wrapped)."""
    return (
        isinstance(payload, dict)
        and set(payload) == {"omitted", "bytes", "sha256"}
        and isinstance(payload["omitted"], str)
        and payload["omitted"] in reasons
        and type(payload["bytes"]) is int
        and payload["bytes"] >= 0
        and isinstance(payload["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", payload["sha256"]) is not None
    )


def _counts(request: dict[str, Any], cached: bool) -> dict[str, Any]:
    return {key: request[key] for key in (*token_fields(cached), "missing")}


def _parses(body: object) -> bool:
    """Whether `parse_reply` accepts a body, as it did every counted response."""
    # Imported here: the transport imports this module.
    from .transport import EndpointError, parse_reply

    try:
        parse_reply(body)
    except EndpointError:
        return False
    return True


def _is_http_evidence(value: object) -> bool:
    # Imported here: the transport imports this module.
    from .transport import is_http_evidence

    return is_http_evidence(value)


def _http_results(evidence: object) -> frozenset[tuple[str, str]] | None:
    """Results `classify` records with a `rejected_http` record of `evidence`;
    None for evidence the transport cannot write."""
    # Imported here: the transport imports this module.
    from .transport import is_capacity_rejection, is_http_evidence

    if not is_http_evidence(evidence):
        return None
    return CAPACITY_RESULTS if is_capacity_rejection(evidence) else CONTRACT_RESULTS


def check_journal(
    events: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    streaming: bool = False,
    cached: bool = False,
) -> None:
    """Bind one episode's measurements to its journal (same clock); `events`
    are the episode's, ending with its `case_end`, whose payload is `result`.

    Each request was sent after its `model_request` event and answered before
    the next event; quota-omitted requests were never sent, and only a sent
    request's record follows it. A request the episode continued after (a
    further request, a tool call or a submission) was answered. A journaled
    body fixes its outcome and token counts: a rejected-response record is an
    endpoint error (a `rejected_raw` one with a body `parse_reply` refuses, a
    `rejected_http` one with the transport's evidence shape), a `late_raw`
    record of a JSON object a late body after the deadline, any other body a
    response `parse_reply` accepts, with the counts its usage reports; a marker
    an outcome its reason allows. Without a journaled body only an answered
    request has counts. Only a timeout (or a relabel) ends a request at the
    deadline, and an error record ends the episode with a result `classify`
    records for it or a relabel, an omitted record with one for the record its
    request outcome names. Tool time fits between each tool call and the next
    event, verifier time between the submission and the next event; without a
    submission there is none.

    In a streamed run (`streaming` True) a request whose record holds the
    transport's HTTP error read no stream, so it has no first token, and one
    whose journaled body (a response, a late body or a refused body) carries
    a token has one; `validate_request` settles the rest of the mode.
    """
    requests = result["requests"]
    end = (result["status"], result["reason"])
    positions = [
        index
        for index, event in enumerate(events)
        if event["kind"] == "model_request"
        and not is_marker(event["payload"], REQUEST_MARKERS)
    ]
    if len(positions) != len(requests):
        raise ValueError("Request measurements differ from the journal")
    replies = {index + 1 for index in positions}
    if any(
        event["kind"] == "model_response" and index not in replies
        for index, event in enumerate(events)
    ):
        raise ValueError("Request measurements differ from the journal")
    previous = 0.0
    # Completion tokens the loop counted, exactly where a journaled body holds
    # an out-of-range count, else only bounded below (see validate_measurements).
    known = unknown = 0
    bounds = [*positions[1:], len(events)] if positions else []
    for index, bound, request in zip(positions, bounds, requests, strict=True):
        known += request["completion_tokens"] or 0
        hidden = request["missing"].get("completion_tokens") == OUT_OF_RANGE
        unknown += hidden
        if index + 1 >= len(events):
            raise ValueError("Request measurements differ from the journal")
        sent, received = request["sent_seconds"], request["received_seconds"]
        after = events[index + 1]
        if not (
            previous <= sent
            and events[index]["elapsed_seconds"] <= sent
            and received <= after["elapsed_seconds"]
        ):
            raise ValueError("Request time lies outside its journal events")
        previous = received
        outcome = request["outcome"]
        answered = bound < len(events) or any(
            e["kind"] in {"tool_call", "submission"} for e in events[index:bound]
        )
        if answered and outcome != "response":
            raise ValueError("Request measurements differ from the journal")
        if outcome == "deadline" and end[0] != "timeout" and end not in RELABELS:
            raise ValueError("Request outcome differs from the result")
        payload = after["payload"]
        if after["kind"] != "model_response":
            # No body was journaled: only an answered request has counts.
            if outcome != "response" and any(
                request["missing"].get(field) != NO_RESPONSE
                for field in token_fields(cached)
            ):
                raise ValueError("Request tokens without a journaled body")
            continue
        results: frozenset[tuple[str, str]] | None = None
        if is_marker(payload):
            # The marker cannot settle the counts, only that the request
            # journaled a record its reason names (a failed request journals
            # none).
            if outcome not in MARKER_OUTCOMES[payload["omitted"]]:
                raise ValueError("Request outcome differs from the journal")
            expected, body = outcome, None
            results = OUTCOME_RESULTS.get(outcome)
        elif set(payload) == {"rejected_raw"}:
            if payload["rejected_raw"] is None or _parses(payload["rejected_raw"]):
                raise ValueError("Request outcome differs from the journal")
            expected, body, results = "endpoint_error", None, CONTRACT_RESULTS
        elif set(payload) == {"rejected_http"}:
            results = _http_results(payload["rejected_http"])
            if results is None:
                raise ValueError("Request outcome differs from the journal")
            expected, body = "endpoint_error", None
        elif set(payload) == {"late_raw"}:
            # `classify` wraps a late body only when it is a JSON object.
            if not isinstance(payload["late_raw"], dict):
                raise ValueError("Request outcome differs from the journal")
            expected, body, results = "deadline", payload["late_raw"], LATE_RESULTS
        elif _parses(payload):
            expected, body = "response", payload
        else:
            raise ValueError("Request outcome differs from the journal")
        if outcome != expected:
            raise ValueError("Request outcome differs from the journal")
        if results is not None and end not in results:
            raise ValueError("Request outcome differs from the result")
        if streaming:
            # The first tokens a record allows: none after an HTTP error status
            # (no stream was read, its evidence late or not); for a body the
            # stream assembled (a response, a late or a refused one) one
            # exactly when it carries output, as its deltas did; after an
            # error chunk (an error event never sets one) or behind a marker
            # either.
            held = payload["rejected_raw"] if set(payload) == {"rejected_raw"} else body
            if set(payload) == {"rejected_http"} or _is_http_evidence(held):
                allowed = {False}
            elif held is None or is_error_chunk(held):
                allowed = {False, True}
            else:
                allowed = {carries_token(held)}
            if (request[FIRST_TOKEN] is not None) not in allowed:
                raise ValueError("First token differs from the journal")
        usage = body.get("usage") if isinstance(body, dict) else None
        if body is not None and _counts(request, cached) != token_counts(
            usage, cached=cached
        ):
            raise ValueError("Request tokens differ from the journaled usage")
        if hidden and isinstance(usage, dict):
            known += usage["completion_tokens"]
            unknown -= 1
    total = result["output_tokens"]
    if unknown:
        consistent = total >= known + (MAX_TOKEN_COUNT + 1) * unknown
    else:
        consistent = total == known
    if not consistent:
        raise ValueError("Request tokens differ from the case total")
    _check_phase_times(events, result["timing"])


def _check_phase_times(events: list[dict[str, Any]], timing: dict[str, Any]) -> None:
    def span(kind: str) -> float:
        return sum(
            events[i + 1]["elapsed_seconds"] - event["elapsed_seconds"]
            for i, event in enumerate(events[:-1])
            if event["kind"] == kind
        )

    tolerance = TIMING_TOLERANCE_SECONDS
    if (
        timing["tool_seconds"] > span("tool_call") + tolerance
        or timing["verifier_seconds"] > span("submission") + tolerance
    ):
        raise ValueError("Case timing lies outside its journal events")


# Streaming mode (`--stream`): the moment of a request's first token, on the
# journal clock; null where the run does not stream or no token arrived.
FIRST_TOKEN = "first_token_seconds"


def has_token(fields: object) -> bool:
    """Whether a streamed delta, or an assembled message, carries output: a
    non-empty text field other than the role, or a tool call."""
    return isinstance(fields, dict) and any(
        (key != "role" and isinstance(value, str) and value != "")
        or (key == "tool_calls" and isinstance(value, list) and bool(value))
        for key, value in fields.items()
    )


def is_error_chunk(body: object) -> bool:
    """Whether a streamed chunk is an error object (a non-null `error`, or
    `object` "error"); an assembled stream body never is one."""
    return isinstance(body, dict) and (
        body.get("error") is not None or body.get("object") == "error"
    )


def carries_token(body: object) -> bool:
    """Whether a chat completion body's one message carries output; an
    assembled stream does exactly when one of its deltas did."""
    choices = body.get("choices") if isinstance(body, dict) else None
    return (
        isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict)
        and has_token(choices[0].get("message"))
    )


# Request record keys without `cached_tokens`.
REQUEST_KEYS = frozenset(
    {
        "sent_seconds",
        "received_seconds",
        "outcome",
        *TOKEN_FIELDS,
        "missing",
        FIRST_TOKEN,
    }
)
