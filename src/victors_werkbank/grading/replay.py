"""Deterministic offline regrading of one recorded host-simulated episode.

`replay` rebuilds a fresh episode and re-runs the shared loop's contract over a
complete journaled segment (`case_start` through `case_end`) recorded by the
identical package build: `case_start` must be exactly what the agent of this
package writes, its digests included. Every request must carry exactly the
history and token cap the loop would have sent, every stop rule is re-applied,
every journaled tool call is re-applied through the shared episode loop and
must reproduce its journaled `tool_result`, and the per-case evidence quota,
its omission markers and the case journal's bytes are recomputed exactly from
the payloads the runner held. A segment that ended before its submission may
stop wherever the loop can stop, never after a point where it must, and its
`case_end` must be a result the runner records at that stop; a byte limit can
end it only where the replayed bytes exceed it. For a submitted segment it
recomputes the submission and runs the verifier; the result must equal the
journaled `verification` and `case_end` records, so a segment whose
verification was not recorded is refused. It never guesses, repairs or falls
back: any inconsistency raises `Ungradable` with a short machine reason.
Code cases replay only with an executor: their journal is a
Harbor agent phase whose executions ran in the task container. Its preflight and
every result of a call that executed are taken as journaled once their shape is
one the executor and the case record: a `public_test` result in the single-run
shape, or a structured result under the case's `ResultContract`, whose values
the case derives from the arguments and its host state alone (counters,
argument echoes) are recomputed and must be equal; everything else replays as
above. What an execution printed is evidence of the agent phase, not re-run. A submitted
segment ends with `agent_end` instead of a verification, and the submission is
verified here, once, with `executor`. Without an executor they are refused.
Wall-clock deadlines are re-checked only where the journal's
clock decides them: an event's time is taken after the deadline check before
it, so it cannot prove that check passed, but it bounds every later check from
below. No event the loop writes only after a passed agent-deadline check (a
request, a tool call, the submission) may follow an event recorded past that
deadline before the check, a stop after such an event records only the
deadline (or a relabel), no verification follows a submission past the wall
budget, and a last event past the wall budget proves the cleanup relabel.
Otherwise a deadline result is accepted wherever the runner can record one,
once the recorded duration reaches its budget within the journal's clock.
"""

import hashlib
import json
import math
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent import episode as agent_episode
from ..agent.loop import (
    EVENT_OVERHEAD_BYTES,
    LIFECYCLE_KINDS,
    NO_PROGRESS,
    EpisodeState,
    Progress,
    answer_tool_call,
    encode_event,
    graded_status,
    json_bytes,
    omission,
    render_prompt,
    result_record,
    security_record,
    verification_payload,
)
from ..agent.measurements import (
    CLEANUP_DEADLINE,
    RELABELS,
    REMOVAL_FAILED,
    check_journal,
    identity_sampling,
    validate_measurements,
    validate_sampling,
)
from ..agent.transport import (
    MAX_RESPONSE_BYTES,
    EndpointError,
    Reply,
    ToolCall,
    is_capacity_rejection,
    is_http_evidence,
    json_nesting,
    parse_reply,
)
from ..core_profile import design_profile_sha256
from ..domains.catalog import CATALOG, Episode, EpisodeContext
from ..domains.execution import Execution, Executor
from ..environment import validate_collected
from ..journal import JournalError, validate_events
from ..registry import BUDGET_PROFILE, CASE_LIMITATIONS, CASES, lifecycle
from ..serving import SERVING_FORMAT, is_local_path, validate_serving
from ..snapshots import SNAPSHOT_LIMITS, snapshot_name

# Evidence semantics this grader reproduces (the Harbor agent's run contract);
# other recorded values are refused.
CONTRACTS = agent_episode.CONTRACTS
REASONING_HISTORY_POLICY = agent_episode.REASONING_HISTORY_POLICY
# Every key the agent writes into `case_start`; its environment is taken at
# the run's start, before any sandbox probe.
START_KEYS = frozenset(
    {
        "branch_id",
        "run_contract_version",
        "executor_semantics",
        "case_limitations",
        "executor_code_sha256",
        "design_profile_sha256",
        "wall_seconds",
        "output_tokens",
        "verifier_reserve_seconds",
        "turns",
        "max_tokens_per_call",
        "budget_profile",
        "model_id",
        "sampling",
        "streaming",
        "report_format",
        "reasoning_history_policy",
        "environment",
        "serving",
    }
)
UNPROBED = {"version": None, "reason": "not probed"}
OMISSION_REASONS = frozenset(
    {"case_journal_byte_budget", "segment_byte_budget", "response_byte_budget"}
)
# Trust label the runner records for each event kind of a host-simulated case.
EVENT_TRUST = {
    "case_start": "trusted_runner",
    "model_request": "trusted_runner",
    "model_response": "untrusted_model",
    "tool_call": "untrusted_model",
    "tool_result": "untrusted_tool",
    "submission": "trusted_runner",
    "verification": "trusted_verifier",
    "case_end": "trusted_runner",
    "agent_end": "trusted_runner",
}
# Omission reasons the runner writes for each kind; only a response is replaced
# for its own size or on the emergency path.
MARKER_REASONS = {
    "model_request": frozenset({"case_journal_byte_budget"}),
    "model_response": OMISSION_REASONS,
    "tool_call": frozenset({"case_journal_byte_budget"}),
    "tool_result": frozenset({"case_journal_byte_budget"}),
}
# Seconds by which rounding of the runner's monotonic clock arithmetic may
# shift a recorded duration across a deadline, in either direction.
CLOCK_TOLERANCE = 0.001
# Outcomes the runner records where the loop stops before a submission:
# `EpisodeState`'s default, the no-progress stop (`agent.loop.NO_PROGRESS`),
# the agent deadline, a failed build, and a failed journal append. An append
# exceeds the segment budget only where its line can cross the case journal's
# ceiling (see `_Ledger`); only a payload nested deep enough can also fail to
# re-encode (an `EndpointError` or a `RecursionError`). An I/O failure poisons
# the journal, so no `case_end` follows it.
DEFAULT_OUTCOME = ("failed", "turn_limit")
DEADLINE = ("timeout", "agent_deadline")
ACCOUNTING = ("unsupported", "completion_token_accounting_missing")
ENDPOINT_ERROR = ("invalid", "endpoint_contract_error")
INFRASTRUCTURE = ("invalid", "infrastructure_or_contract_error")
# A code episode whose container executor failed its preflight.
SANDBOX_UNAVAILABLE = ("unsupported", "sandbox_unavailable")
VERIFICATION_DEADLINE = ("timeout", "verification_deadline")
PREFLIGHT_KEYS = frozenset({"isolated", "executor", "uid", "python"})
# Execution reasons a code case's public result can carry.
EXECUTION_REASONS = frozenset(
    {"completed", "execution_failed", "timeout", "output_limit"}
)
SEGMENT = ("output_limit", "segment_byte_budget")
# Nesting below which a payload re-encodes and replays on any interpreter
# stack. Tool arguments never reach it: deeper than `TOOL_ARGUMENT_NESTING`
# they are malformed in the runner and here alike.
SHALLOW_NESTING = 64
# Bound on a parsed body's re-encoding: at most four times the body (`1e15,`
# becomes `1000000000000000.0, `), and the endpoint reads no larger body. An
# assembled stream is held to the same limit as journaled, and each event's
# data (an error chunk journaled as is) to the limit as read.
REENCODED_RESPONSE_BYTES = 5 * MAX_RESPONSE_BYTES
# Bytes of the largest marker that replaces a response above `RESPONSE_BYTES`.
RESPONSE_MARKER_BYTES = len(
    encode_event(
        {
            "omitted": "response_byte_budget",
            "bytes": REENCODED_RESPONSE_BYTES,
            "sha256": "0" * 64,
        }
    )
)
# Bytes the `{"late_raw": body}` wrapper adds to a late body's encoding.
LATE_RECORD_OVERHEAD = len(encode_event({"late_raw": {}})) - len(encode_event({}))
# Results `classify` gives a rejected response, whose tokens are never counted,
# and with a late body after the deadline or a relabel, every path that
# journals a response through the emergency path (as `report.py` validates
# them).
REJECTED_ENDS = (
    ("invalid", "endpoint_contract_error"),
    ("unsupported", "endpoint_request_capacity"),
    ("output_limit", "endpoint_capacity_rejection_after_responses"),
)
ERROR_RESPONSE_ENDS = REJECTED_ENDS + (("timeout", "agent_deadline"), *RELABELS)
# Reasons with which a response whose tokens were counted can be missing: its
# append failed for the segment budget, or the journal could not re-encode it
# (a nesting limit); a relabel may replace any of them.
MISSING_RESPONSE_REASONS = (
    "segment_byte_budget",
    "endpoint_contract_error",
    "infrastructure_or_contract_error",
    "cleanup_deadline",
)
# Results `finalize` never relabels for an exhausted evidence quota.
QUOTA_PRIORITY_STATUSES = ("invalid", "timeout", "unsupported")
QUOTA_PRIORITY_REASONS = ("endpoint_capacity_rejection_after_responses",)


class Ungradable(ValueError):
    """Recorded evidence cannot be regraded deterministically."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Grade:
    """Regraded verifier result.

    `verification` equals the journaled `verification` payload
    (`{checks, strict_success, **extra}`); `status` and `reason` are the
    check-derived classification after the episode's `outcome()` override.
    """

    verification: dict[str, Any]
    status: str
    reason: str


@dataclass(frozen=True)
class Verified:
    """A code episode's submission, verified here once: the verification
    payload (None where the verifier raised) and the episode's result record."""

    verification: dict[str, Any] | None
    result: dict[str, Any]
    verifier_seconds: float
    # The exception type where the verifier raised; never its text.
    failure: str | None = None


class _Executions:
    """A code episode's executor during replay. In the agent phase an
    execution is the journaled one, so `run` only notes that the case executed
    (see `_recorded_execution`); after `recording` ends it runs for real."""

    def __init__(self, real: Executor) -> None:
        self.real = real
        self.recording = True
        self.executed = False

    def run(
        self, source: Path, helper: str, data: bytes = b"", *, timeout: float = 5.0
    ) -> Execution:
        if not self.recording:
            return self.real.run(source, helper, data, timeout=timeout)
        self.executed = True
        return Execution(b"", b"", 0, "completed", 0.0)

    def probe(self, empty_source: Path) -> Execution:
        raise Ungradable("probe_replayed")

    def take(self) -> bool:
        executed, self.executed = self.executed, False
        return executed


def _recorded_execution(
    recorded: Any, computed: dict[str, Any], episode: Episode, call: ToolCall
) -> dict[str, Any]:
    """The journaled `tool_result` of a call that executed in the agent phase.

    Under the case's `ResultContract` for the tool, the journaled result must
    pass the contract's check and equal the replayed result on every computed
    key. Otherwise it must be the replayed payload with a public result of the
    case's shape (the keys the case writes, an executor reason, at most 1 KiB of
    each stream and a `passed` flag, where the case has one, that the stdout
    proves by the case's rule, `Episode.public_passed`)."""
    if not isinstance(recorded, dict) or _is_omission("tool_result", recorded):
        raise Ungradable("execution_not_recorded")
    result = recorded.get("result")
    shape = computed["result"]
    contract = episode.result_contract(call.name)
    if contract is not None:
        if not isinstance(result, dict) or not isinstance(shape, dict):
            raise Ungradable("tool_result_mismatch")
        try:
            arguments = episode.arguments(call)
            acceptable = contract.valid(result, shape, arguments) is True
        except Exception:
            acceptable = False
        if (
            not acceptable
            or any(
                (key in result) != (key in shape)
                or (key in shape and not _same(shape[key], result[key]))
                for key in contract.computed
            )
            or not _same({**computed, "result": result}, recorded)
        ):
            raise Ungradable("tool_result_mismatch")
        return {**computed, "result": result}
    if (
        not isinstance(result, dict)
        or set(result) != set(shape)
        or not _same({**computed, "result": result}, recorded)
        or result["reason"] not in EXECUTION_REASONS
    ):
        raise Ungradable("tool_result_mismatch")
    for stream in ("stdout", "stderr"):
        if (
            not isinstance(result[stream], str)
            or len(result[stream]) > 1024
            or type(result[f"{stream}_truncated"]) is not bool
        ):
            raise Ungradable("tool_result_mismatch")
    if "passed" in shape and result["passed"] is not episode.public_passed(result):
        raise Ungradable("tool_result_mismatch")
    return {**computed, "result": result}


def _check_preflight(payload: Any) -> None:
    """The container executor's journaled preflight, as its probe prints it."""
    if not isinstance(payload, dict) or set(payload) != {"executor_preflight"}:
        raise Ungradable("preflight_mismatch")
    preflight = payload["executor_preflight"]
    if (
        not isinstance(preflight, dict)
        or set(preflight) != PREFLIGHT_KEYS
        or preflight["isolated"] is not True
        or preflight["executor"] != "container"
        or type(preflight["uid"]) is not int
        or preflight["uid"] <= 0
        or not isinstance(preflight["python"], str)
    ):
        raise Ungradable("preflight_mismatch")


@dataclass(frozen=True)
class NotGraded:
    """The episode recorded no submission, so there is nothing to grade.

    `security` holds the replayed counters, which equal the journaled
    `case_end` security record (`None` for cases without counters).
    """

    reason: str = "no_submission"
    security: dict[str, int] | None = None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _journaled(value: Any) -> Any:
    """The value as an event payload stores it (same encoding as the journal)."""
    return json.loads(
        json.dumps(value, allow_nan=False, ensure_ascii=False)
        .encode("utf-8", errors="backslashreplace")
        .decode("utf-8")
    )


def _same(value: Any, recorded: Any) -> bool:
    try:
        return _canonical(_journaled(value)) == _canonical(recorded)
    except (TypeError, ValueError, RecursionError):
        return False


def _duration(value: Any) -> bool:
    """A difference of the runner's monotonic clock: a finite float that is not
    negative (not even a negative zero)."""
    return (
        type(value) is float and math.isfinite(value) and math.copysign(1.0, value) > 0
    )


def _is_omission(kind: str, payload: dict[str, Any]) -> bool:
    """Only the runner-authored marker shape counts; raw model keys do not.

    A marker needs a reason the runner writes for this event kind, a
    non-negative byte count and a SHA-256 hex digest; a malformed one is
    corrupted evidence. A late endpoint body is journaled wrapped as
    `{"late_raw": body}`, so no endpoint body takes a marker's shape.
    """
    if (
        kind not in MARKER_REASONS
        or set(payload) != {"omitted", "bytes", "sha256"}
        or not isinstance(payload["omitted"], str)
        or payload["omitted"] not in OMISSION_REASONS
    ):
        return False
    size, digest = payload["bytes"], payload["sha256"]
    if (
        payload["omitted"] in MARKER_REASONS[kind]
        and type(size) is int
        and size >= 0
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        return True
    raise Ungradable("malformed_omission")


def _size_bounds(payload: Any) -> tuple[int, int]:
    """Bounds on the bytes the runner's sink encoded for a journaled payload.

    The journal reads a split UTF-16 surrogate pair back as one astral
    character, 8 bytes shorter than the sink encoded the pair.
    """
    encoded = encode_event(payload)
    astral = sum(ord(char) > 0xFFFF for char in encoded.decode("utf-8"))
    return len(encoded), len(encoded) + 8 * astral


def _describes(marker: dict[str, Any], reason: str, payload: Any) -> bool:
    """Whether `marker` is the runner's omission marker for `payload`.

    Replay rebuilds the payload as the runner held it (a split surrogate pair
    in a tool result stays split), so its size and digest must match.
    """
    return _same(omission(reason, encode_event(payload)), marker)


def _late_tokens(raw: dict[str, Any]) -> int:
    """Completion tokens `classify` counts from a late body after the deadline."""
    usage = raw.get("usage")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return tokens if type(tokens) is int and tokens >= 0 else 0


def _call(payload: dict[str, Any]) -> ToolCall:
    if set(payload) != {"id", "name", "arguments_raw"} or not all(
        isinstance(payload[key], str) for key in payload
    ):
        raise Ungradable("malformed_tool_call")
    return ToolCall(payload["id"], payload["name"], payload["arguments_raw"])


def _check_structure(
    case_id: str, branch_id: str, events: list[dict[str, Any]], *, code: bool = False
) -> int | None:
    """Validate one segment's shape; returns the submission index, if any.

    A `code` segment may hold the executor preflight as its second event, never
    a verification, and ends with `agent_end` right after its submission unless
    a relabel ended it with `case_end`."""
    if not isinstance(events, list):
        raise Ungradable("invalid_events")
    if not events:
        raise Ungradable("no_events")
    first = events[0]
    start = first.get("sequence") if isinstance(first, dict) else None
    if type(start) is not int or start < 0:
        raise Ungradable("invalid_events")
    try:
        validate_events(events, start_sequence=start)
        for event in events:
            encode_event(event["payload"])
        # The runner's run identity is a UUID4; its clock gives float times.
        run_id = uuid.UUID(events[0]["run_id"])
    except (JournalError, ValueError, TypeError, RecursionError):
        raise Ungradable("invalid_events") from None
    if (
        run_id.version != 4
        or str(run_id) != events[0]["run_id"]
        or not all(_duration(event["elapsed_seconds"]) for event in events)
    ):
        raise Ungradable("invalid_events")
    for position, event in enumerate(events):
        trust = EVENT_TRUST.get(event["kind"])
        if event["kind"] == "agent_end" and not code:
            trust = None
        if code and position == 1 and event["kind"] == "tool_result":
            # The executor preflight, if this is one (`_check_preflight`).
            trust = event["trust"] if event["trust"] == "trusted_runner" else trust
        if event["trust"] != trust:
            raise Ungradable("trust_mismatch")
    if any(event["case_id"] != case_id for event in events):
        raise Ungradable("foreign_event")
    if (
        events[0]["kind"] != "case_start"
        or events[0]["payload"].get("branch_id") != branch_id
    ):
        raise Ungradable("case_start_mismatch")
    kinds = [event["kind"] for event in events]
    if kinds.count("case_start") != 1:
        raise Ungradable("multiple_case_starts")
    ends = ("case_end", "agent_end") if code else ("case_end",)
    if any(kind in ends for kind in kinds[:-1]):
        raise Ungradable("case_end_not_last")
    if kinds[-1] not in ends:
        raise Ungradable("missing_case_end")
    submissions = [i for i, kind in enumerate(kinds) if kind == "submission"]
    if len(submissions) > 1:
        raise Ungradable("multiple_submissions")
    if code:
        if "verification" in kinds:
            raise Ungradable("verification_in_agent_phase")
        if not submissions:
            if kinds[-1] == "agent_end":
                raise Ungradable("agent_end_without_submission")
            return None
        index = submissions[0]
        if index != len(kinds) - 2:
            raise Ungradable("events_after_submission")
        if any(_is_omission(e["kind"], e["payload"]) for e in events[:index]):
            raise Ungradable("omission_before_submission")
        return index
    if kinds.count("verification") > 1:
        raise Ungradable("multiple_verifications")
    if not submissions:
        if "verification" in kinds:
            raise Ungradable("verification_without_submission")
        return None
    index = submissions[0]
    if "verification" in kinds[:index]:
        raise Ungradable("verification_before_submission")
    if not set(kinds[index + 1 :]) <= {"verification", "case_end"}:
        raise Ungradable("events_after_submission")
    if "verification" not in kinds[index + 1 :]:
        raise Ungradable("verification_missing")
    if any(_is_omission(event["kind"], event["payload"]) for event in events[:index]):
        raise Ungradable("omission_before_submission")
    return index


def _check_contract(
    case_id: str, start: dict[str, Any], *, code: bool = False
) -> tuple[int, int, int]:
    """Recorded semantics and budgets must be the ones this grader reproduces."""
    contract = CASES[case_id]
    version = start.get("run_contract_version")
    run_contract = CONTRACTS.get(version) if isinstance(version, str) else None
    if run_contract is None:
        raise Ungradable("executor_semantics_mismatch")
    semantics = {
        "executor_semantics": agent_episode.container_semantics(case_id)
        if code
        else contract.executor_semantics,
        "case_limitations": CASE_LIMITATIONS.get(case_id, []),
        "reasoning_history_policy": REASONING_HISTORY_POLICY,
    }
    if any(
        key not in start or not _same(value, start[key])
        for key, value in semantics.items()
    ):
        raise Ungradable("executor_semantics_mismatch")
    resolved = lifecycle(case_id, start["branch_id"])
    cap, turns = resolved.output_tokens, resolved.turns
    budgets = {
        "wall_seconds": resolved.wall_seconds,
        "output_tokens": cap,
        "verifier_reserve_seconds": resolved.verifier_reserve_seconds,
        "turns": turns,
    }
    per_call = start.get("max_tokens_per_call")
    if (
        any(key not in start or not _same(v, start[key]) for key, v in budgets.items())
        or type(per_call) is not int
        or not 1 <= per_call <= 65536
    ):
        raise Ungradable("budget_mismatch")
    _check_provenance(start, run_contract)
    return cap, turns, per_call


def _cached(events: list[dict[str, Any]]) -> bool:
    """Whether the episode's run contract records cached prompt tokens; its
    `case_start` was checked before the loop replayed."""
    return CONTRACTS[events[0]["payload"]["run_contract_version"]].cached_tokens


def _check_provenance(start: dict[str, Any], run_contract: Any) -> None:
    """The rest of `case_start` must be what the agent of this package writes:
    its constants and digests, a served model name, a boolean streaming mode,
    a valid serving identity with that mode in its configuration and an
    environment `collect_environment` writes before any sandbox probe."""
    expected = {
        "budget_profile": BUDGET_PROFILE,
        "report_format": run_contract.report_format,
        "executor_code_sha256": agent_episode.harness_digest(code_only=True),
        "design_profile_sha256": design_profile_sha256(),
    }
    environment: Any = start.get("environment")
    model = start.get("model_id")
    try:
        validate_collected(environment)
        validate_sampling(start.get("sampling"))
        validate_serving(start.get("serving"), model)
        entry = start["serving"]["models"]["entry"]
        configuration = {
            "sampling": identity_sampling(start["sampling"]),
            "max_tokens_per_call": start["max_tokens_per_call"],
            "harness_sha256": agent_episode.harness_digest(),
            "streaming": start["streaming"],
        }
        unprobed = (
            type(start["streaming"]) is bool
            and _same(UNPROBED, environment["sandbox_python"])
            and (entry is None or entry["id"] == model)
            and start["serving"]["format"] == SERVING_FORMAT
            and _same(configuration, start["serving"]["configuration"])
        )
    except (ValueError, TypeError, KeyError):
        unprobed = False
    if (
        set(start) != START_KEYS
        or not unprobed
        or not isinstance(model, str)
        or not model
        or is_local_path(model)
        or any(not _same(value, start[key]) for key, value in expected.items())
    ):
        raise Ungradable("provenance_mismatch")


def _check_quota(events: list[dict[str, Any]]) -> None:
    """Bound the per-case evidence quota the runner's sink charged.

    Every admitted event must fit `agent_episode.CASE_JOURNAL_BYTES`, and the first
    case-journal marker must be the event that could cross it; only markers
    follow. Emergency segment markers are never charged. A journaled payload
    only bounds the charge
    (see `_size_bounds`); `_Ledger` charges the replayed payloads exactly.
    """
    low = high = 0
    exhausted = False
    for event in events:
        kind, payload = event["kind"], event["payload"]
        if kind in LIFECYCLE_KINDS:
            continue
        if _is_omission(kind, payload):
            reason = payload["omitted"]
            if reason == "segment_byte_budget":
                continue
            if reason == "case_journal_byte_budget":
                if (
                    exhausted
                    or high + payload["bytes"] + EVENT_OVERHEAD_BYTES
                    > agent_episode.CASE_JOURNAL_BYTES
                ):
                    exhausted = True
                    continue
                raise Ungradable("evidence_quota_mismatch")
        if exhausted:
            raise Ungradable("evidence_after_omission")
        size_low, size_high = _size_bounds(payload)
        low += size_low + EVENT_OVERHEAD_BYTES
        high += size_high + EVENT_OVERHEAD_BYTES
        if low > agent_episode.CASE_JOURNAL_BYTES:
            raise Ungradable("evidence_quota_mismatch")


def _check_deadlines(
    events: list[dict[str, Any]], final_answer: bool, deadline: float, wall: float
) -> None:
    """Events the loop writes only after a passed agent-deadline check must
    not follow an event recorded past that deadline before the check, and the
    verification never follows a submission recorded past the wall budget.

    The loop checks before each request, tool call and the submission, so the
    previous event bounds that check's clock from below. A final-answer
    episode's forbidden call follows only the check after its response
    arrived, which its request bounds: a call after a late response is genuine.
    The runner checks the wall budget after the submission, before verifying.
    """
    for index, event in enumerate(events[2:], 2):
        kind = event["kind"]
        if kind == "verification":
            if _late(events, index - 1, wall):
                raise Ungradable("event_after_deadline")
        elif kind in {"model_request", "tool_call", "submission"}:
            back = 2 if final_answer and kind == "tool_call" else 1
            if _late(events, index - back, deadline):
                raise Ungradable("event_after_deadline")


def _late(events: list[dict[str, Any]], index: int, deadline: float) -> bool:
    """Whether event `index` lies past `deadline` on the case's journal clock;
    a deadline check read after it then failed."""
    elapsed = events[index]["elapsed_seconds"] - events[0]["elapsed_seconds"]
    return elapsed > deadline + CLOCK_TOLERANCE


def _ceilings() -> tuple[int, int, int]:
    """Ordinary, emergency and terminal ceilings of a case journal, as
    the run allocates it from the agent's limits."""
    limit = agent_episode.CASE_JOURNAL_BYTES + agent_episode.CASE_JOURNAL_RESERVE
    terminal = limit - agent_episode.CASE_END_RESERVE
    return terminal - agent_episode.CASE_ERROR_RESERVE, terminal, limit


def _encoded_size(payload: Any) -> int | None:
    """Bytes the sink encodes for a replayed payload; `None` where it raises
    instead (an integer beyond the digit limit, a non-finite float, or nesting
    deeper than the stack)."""
    try:
        return len(encode_event(payload))
    except (ValueError, RecursionError):
        return None


def _elapsed_slack(low: float, high: float) -> int:
    """Bytes a journal time from `low` to `high` can add over a recorded `0`.

    A float's repr has at most 17 significant digits, positional from 1e-4
    below 1e16, where the smallest value has the longest form.
    """
    if high == 0:
        longest = len("0.0")
    elif low < 1e-4 or high >= 1e16:
        longest = len("1.7976931348623157e+308")
    else:
        longest = len(format(low, "#.17g"))
    return longest - len("0")


class _Ledger:
    """The evidence quota the sink charged and the bytes the journal wrote.

    The loop admits each whole event it matches with the payload the runner
    held, so both are exact: only a tool result or a submission can be
    journaled shorter than the runner wrote it (a split surrogate pair reads
    back as one character), and replay recomputes both. The run allocates
    every case journal from the agent's limits, so an append fails on the
    segment budget only where its line can cross the journal's ceiling.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.lines = [len(encode_event(event)) + 1 for event in events]
        self.extra: dict[int, int] = {}
        self.charged = 0
        self.exhausted = False
        # The response marker written on the error reserve, if any.
        self.emergency: int | None = None

    def wrote(self, index: int, payload: Any) -> int:
        """The journal wrote event `index` whole from `payload`; its size."""
        size = len(encode_event(payload))
        self.extra[index] = size - len(encode_event(self.events[index]["payload"]))
        return size

    def admit(self, index: int, payload: Any) -> None:
        """The sink charged event `index`, written whole from `payload`."""
        self.charged += self.wrote(index, payload) + EVENT_OVERHEAD_BYTES
        if self.exhausted or self.charged > agent_episode.CASE_JOURNAL_BYTES:
            raise Ungradable("evidence_quota_mismatch")

    def crosses(self, marker: dict[str, Any]) -> bool:
        """Whether the quota required this case-journal marker; it then stays
        exhausted."""
        if marker["omitted"] != "case_journal_byte_budget":
            return False
        if (
            self.charged + marker["bytes"] + EVENT_OVERHEAD_BYTES
            > agent_episode.CASE_JOURNAL_BYTES
        ):
            self.exhausted = True
        return self.exhausted

    def occupied(self, index: int) -> int:
        """Bytes the journal held before event `index`."""
        extra = sum(size for at, size in self.extra.items() if at < index)
        return sum(self.lines[:index]) + extra

    def check(self) -> None:
        """Every recorded line fits: the emergency response marker may use the
        error reserve, `case_end` the terminal one."""
        ordinary, emergency, terminal = _ceilings()
        written = 0
        for index, event in enumerate(self.events):
            written += self.lines[index] + self.extra.get(index, 0)
            ceiling = ordinary
            if event["kind"] in {"case_end", "agent_end"}:
                ceiling = terminal
            elif index == self.emergency:
                ceiling = emergency
            if written > ceiling:
                raise Ungradable("segment_occupancy_mismatch")

    def overflows(
        self, index: int, kind: str, size: int | None, *, exact: bool = True
    ) -> bool:
        """Whether a `kind` append that would have been event `index` can
        cross the ceiling. `size` is its payload's encoded bytes (at most that
        many unless `exact`; `None` if unknown). The sink first replaces a
        payload the quota cannot take with its marker. The journal's clock
        bounds the append's time by the previous event's and `case_end`'s."""
        if size is None:
            return True
        sizes = [size]
        if kind not in LIFECYCLE_KINDS:
            room = (
                agent_episode.CASE_JOURNAL_BYTES - EVENT_OVERHEAD_BYTES - self.charged
            )
            admitted = not self.exhausted and room >= (size if exact else 0)
            sizes = [min(size, room)] if admitted else []
            if self.exhausted or size > room:
                marker = {"omitted": "case_journal_byte_budget", "bytes": size}
                sizes.append(len(encode_event({**marker, "sha256": "0" * 64})))
        event = self.events[index]
        envelope = {
            "schema_version": event["schema_version"],
            "run_id": event["run_id"],
            "case_id": event["case_id"],
            "sequence": event["sequence"],
            "elapsed_seconds": 0,
            "kind": kind,
            "payload": None,
            "trust": EVENT_TRUST[kind],
        }
        times = (self.events[index - 1], self.events[-1])
        slack = _elapsed_slack(*(each["elapsed_seconds"] for each in times))
        line = len(encode_event(envelope)) - len(b"null") + 1 + slack
        occupied, ceiling = self.occupied(index), _ceilings()[0]
        return any(occupied + line + each > ceiling for each in sizes)


def _next(
    events: list[dict[str, Any]], index: int, stop: int, kind: str, missing: str
) -> Any:
    if index >= stop or events[index]["kind"] != kind:
        raise Ungradable(missing)
    return events[index]["payload"]


def _ends(index: int, stop: int, reason: str = "event_order_violation") -> None:
    """Nothing but the segment's end may follow the event before `index`."""
    if index != stop:
        raise Ungradable(reason)


@dataclass(frozen=True)
class _Stop:
    """Where a replayed loop ended.

    `reply` is the final reply of a submitted segment; `used` the completion
    tokens the loop counted; `hidden` where counted tokens may be missing
    (after a request whose response is missing, or in an omitted response);
    `outcomes` the (status, reason) pairs the runner can record at this point
    before `finalize`; `measured` the outcomes the last request's record can
    name at an omitted response (empty where the journal settles it).
    """

    reply: Reply | None
    used: int
    hidden: str | None = None
    outcomes: tuple[tuple[str, str], ...] = ()
    measured: tuple[str, ...] = ()


def _replay_loop(
    episode: Episode,
    events: list[dict[str, Any]],
    stop: int,
    *,
    submitted: bool,
    ledger: _Ledger,
    cap: int,
    turns: int,
    per_call: int,
    timeout: float,
    deadline: float,
    executions: _Executions | None = None,
    start: int = 1,
) -> _Stop:
    """Re-run the contract of `agent.loop.run_agent` over `events[start:stop]`.

    `stop` is the submission index, or the `case_end` index of a segment that
    ended before its submission. A submitted segment must reach a final
    response with no stop rule fired. Any other segment may end wherever the
    loop can stop (a stop rule, a deadline, an endpoint or tool error, or a
    journal quota), but every recorded event must be the one the loop writes at
    that point and nothing may follow a point where the loop must stop.
    `ledger` charges each event as the sink did and decides where an append can
    fail on the segment budget. At a stop after an event journaled past the
    agent `deadline`, only the results before the loop's deadline check remain.
    A code episode's `executions` take each execution as journaled; `start` is
    the loop's first event, after the executor preflight where there is one
    (a code episode without it stopped at the preflight).
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": render_prompt(
                episode.prompt, episode.tools, episode.public_test_notice
            ),
        }
    ]
    logged = used = responses = 0
    progress = Progress(episode.tools)
    index = start
    while True:
        if index == stop:
            if submitted:
                raise Ungradable("event_order_violation")
            # Checked where the loop checks it, before its turn limit.
            if progress.stopped:
                return _Stop(None, used, outcomes=(NO_PROGRESS,))
            if responses >= turns:
                return _Stop(None, used, outcomes=(DEFAULT_OUTCOME,))
            outcomes = [DEADLINE]
            if index == start:  # Building the episode failed.
                outcomes += [INFRASTRUCTURE, ("timeout", "setup_deadline")]
                if executions is not None and start == 1:
                    outcomes.append(SANDBOX_UNAVAILABLE)
            if _late(events, stop - 1, deadline):
                # The loop stops at its deadline check before any of these.
                return _Stop(None, used, outcomes=tuple(outcomes))
            if used >= cap:
                outcomes.append(("output_limit", "token_budget"))
            elif len(json_bytes(messages)) > agent_episode.HISTORY_BYTES:
                outcomes.append(("output_limit", "context_byte_budget"))
            else:
                # A request nests only journaled data, so only its size can
                # fail its append.
                request = {
                    "messages_delta": messages[logged:],
                    "max_tokens": min(cap - used, per_call),
                }
                size = _encoded_size(request)
                if ledger.overflows(stop, "model_request", size):
                    outcomes.append(SEGMENT)
            return _Stop(None, used, outcomes=tuple(outcomes))
        if (
            progress.stopped
            or responses >= turns
            or used >= cap
            or len(json_bytes(messages)) > agent_episode.HISTORY_BYTES
        ):
            raise Ungradable("loop_contract_violation")
        request = _next(events, index, stop, "model_request", "event_order_violation")
        index += 1
        request_cap = min(cap - used, per_call)
        expected = {"messages_delta": messages[logged:], "max_tokens": request_cap}
        if _is_omission("model_request", request):
            if not _describes(request, request["omitted"], expected):
                raise Ungradable("model_request_mismatch")
            if not ledger.crosses(request):
                raise Ungradable("evidence_quota_mismatch")
            _ends(index, stop)
            return _Stop(None, used, outcomes=(DEFAULT_OUTCOME,))
        if not _same(expected, request):
            raise Ungradable("model_request_mismatch")
        ledger.admit(index - 1, expected)
        logged = len(messages)
        if index == stop and not submitted:
            # The endpoint failed without a body, or its counted response could
            # not be journaled (which hides its tokens). A larger response is
            # journaled as its marker instead.
            reason = events[stop]["payload"].get("reason")
            hidden = "request" if reason in MISSING_RESPONSE_REASONS else None
            missing = [DEADLINE, ENDPOINT_ERROR, INFRASTRUCTURE]
            size = _response_bytes()
            if ledger.overflows(stop, "model_response", size, exact=False):
                missing.append(SEGMENT)
            return _Stop(None, used, hidden, tuple(missing))
        payload = _next(events, index, stop, "model_response", "event_order_violation")
        index += 1
        responses += 1
        if _is_omission("model_response", payload):
            prior = responses - 1
            return _omitted_response(events, stop, index, payload, ledger, used, prior)
        ledger.admit(index - 1, payload)
        try:
            reply = parse_reply(payload)
        except EndpointError:
            # Only `classify` records an unparseable response, and the loop
            # ends there: a rejected body, or a late body after the deadline.
            if submitted:
                raise Ungradable("unparseable_model_response") from None
            _ends(index, stop)
            if set(payload) in ({"rejected_raw"}, {"rejected_http"}):
                rejected = _rejected(responses - 1, payload)
                if not rejected:
                    # Not even the cleanup deadline relabels a record the
                    # runner cannot write.
                    raise Ungradable("unparseable_model_response") from None
                return _Stop(None, used, outcomes=rejected)
            late = payload.get("late_raw") if set(payload) == {"late_raw"} else None
            if not isinstance(late, dict):
                raise Ungradable("unparseable_model_response") from None
            return _Stop(None, used + _late_tokens(late), outcomes=(DEADLINE,))
        if len(json_bytes(reply.raw)) > agent_episode.RESPONSE_BYTES:
            raise Ungradable("loop_contract_violation")
        tokens = (reply.usage or {}).get("completion_tokens")
        if type(tokens) is not int or tokens < 0:
            if submitted:
                raise Ungradable("loop_contract_violation")
            _ends(index, stop, "loop_contract_violation")
            return _Stop(None, used, outcomes=(ACCOUNTING, DEADLINE))
        used += tokens
        if used > cap or reply.finish_reason in {"length", "content_filter"}:
            if submitted:
                raise Ungradable("loop_contract_violation")
            _ends(index, stop, "loop_contract_violation")
            if used > cap or reply.finish_reason == "length":
                budget = (
                    "response_token_budget"
                    if reply.finish_reason == "length"
                    and request_cap < cap - used + tokens
                    else "token_budget"
                )
                return _Stop(None, used, outcomes=(("output_limit", budget), DEADLINE))
            return _Stop(
                None, used, outcomes=(("failed", "endpoint_refusal"), DEADLINE)
            )
        messages.append(reply.raw["choices"][0]["message"])
        if not reply.tool_calls:
            _ends(index, stop)
            if submitted:
                return _Stop(reply, used)
            # The submission was late, or recording it failed.
            late = _late(events, stop - 1, deadline)
            failed = _submission_failures(episode, reply, ledger, stop, late)
            return _Stop(None, used, outcomes=(DEADLINE, *failed))
        if submitted and episode.final_answer:
            raise Ungradable("loop_contract_violation")
        if index == stop:
            if submitted:
                raise Ungradable("unmatched_tool_call")
            # A deadline, or the journal could not take the first (or the
            # forbidden) call, which follows the check after the response.
            first = _call_payload(reply.tool_calls[0])
            back = 2 if episode.final_answer else 1
            late = _late(events, stop - back, deadline)
            failed = _call_failures(ledger, stop, first, late)
            return _Stop(None, used, outcomes=failed)
        if episode.final_answer:
            # A forbidden call is recorded once and ends the loop.
            if events[index]["kind"] != "tool_call":
                raise Ungradable("tool_call_mismatch")
            _match_call(ledger, index, reply.tool_calls[0])
            episode.forbid_tool()
            _ends(index + 1, stop)
            return _Stop(None, used, outcomes=(("failed", "tool_forbidden"),))
        # Calls past the per-response quota replay as the loop's refusals.
        answered: list[tuple[int, ToolCall, Any, str]] = []
        for position, call in enumerate(reply.tool_calls):
            if index == stop and not submitted:
                # A deadline, or the journal could not take the next call.
                late = _late(events, stop - 1, deadline)
                failed = _call_failures(ledger, stop, _call_payload(call), late)
                return _Stop(None, used, outcomes=failed)
            _next(events, index, stop, "tool_call", "unmatched_tool_call")
            index += 1
            if _match_call(ledger, index - 1, call):
                _ends(index, stop)
                return _Stop(None, used, outcomes=(DEFAULT_OUTCOME,))
            if index == stop and not submitted:
                # The call raised, or its result could not be journaled; the
                # counters include the attempt either way.
                unrecorded = _unrecorded_result(
                    episode, call, position, timeout, ledger, stop, executions
                )
                return _Stop(None, used, outcomes=unrecorded)
            recorded = _next(events, index, stop, "tool_result", "unmatched_tool_call")
            index += 1
            try:
                result = answer_tool_call(episode, call, position, timeout=timeout)
            except Exception:
                raise Ungradable("tool_replay_error") from None
            if executions is not None and executions.take():
                result = _recorded_execution(recorded, result, episode, call)
            try:
                content = json.dumps(result["result"], allow_nan=False)
            except Exception:
                raise Ungradable("tool_replay_error") from None
            if _is_omission("tool_result", recorded):
                if not _describes(recorded, recorded["omitted"], result):
                    raise Ungradable("tool_result_mismatch")
                if not ledger.crosses(recorded):
                    raise Ungradable("evidence_quota_mismatch")
                # The quota stays exhausted: the next call is omitted and ends
                # the loop, unless a deadline comes first.
                rest = reply.tool_calls[position + 1 :]
                if index == stop:
                    if not rest:
                        return _Stop(None, used, outcomes=(DEFAULT_OUTCOME,))
                    late = _late(events, stop - 1, deadline)
                    failed = _call_failures(ledger, stop, _call_payload(rest[0]), late)
                    return _Stop(None, used, outcomes=failed)
                marker = _next(
                    events, index, stop, "tool_call", "event_order_violation"
                )
                if not rest or not _describes(
                    marker, "case_journal_byte_budget", _call_payload(rest[0])
                ):
                    raise Ungradable("evidence_after_omission")
                _ends(index + 1, stop)
                return _Stop(None, used, outcomes=(DEFAULT_OUTCOME,))
            if not _same(result, recorded):
                raise Ungradable("tool_result_mismatch")
            ledger.admit(index - 1, result)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )
            answered.append((position, call, result["result"], content))
        # Every call and result of the response is journaled: the loop counted
        # it here, where every earlier return above stopped it first.
        progress.after_response(episode, answered)


def _match_call(ledger: _Ledger, index: int, call: ToolCall) -> bool:
    """Check event `index` against `call`; whether the quota omitted it."""
    payload = ledger.events[index]["payload"]
    if _is_omission("tool_call", payload):
        if not _describes(payload, "case_journal_byte_budget", _call_payload(call)):
            raise Ungradable("tool_call_mismatch")
        if not ledger.crosses(payload):
            raise Ungradable("evidence_quota_mismatch")
        return True
    if _call(payload) != call:
        raise Ungradable("tool_call_mismatch")
    ledger.admit(index, _call_payload(call))
    return False


def _append_failures(
    ledger: _Ledger, stop: int, kind: str, size: int | None, deep: bool
) -> tuple[tuple[str, str], ...]:
    """Results of a `kind` append that failed at the stop: the segment budget
    where its line can cross the ceiling, and for a deeply nested payload
    re-encoding."""
    failed = [SEGMENT] if ledger.overflows(stop, kind, size) else []
    return tuple(failed + ([ENDPOINT_ERROR, INFRASTRUCTURE] if deep else []))


def _encode_failures(
    ledger: _Ledger, stop: int, kind: str, payload: Any, deep: bool
) -> tuple[tuple[str, str], ...]:
    """Results where the sink took `payload` for a `kind` event at the stop and
    failed: it raises before appending a payload it cannot encode, which
    `classify` records as an infrastructure error; otherwise its append failed
    (see `_append_failures`). A deep payload may have encoded when the episode
    was recorded."""
    size = _encoded_size(payload)
    if size is not None:
        return _append_failures(ledger, stop, kind, size, deep)
    appended = _append_failures(ledger, stop, kind, None, deep) if deep else ()
    return tuple(dict.fromkeys((INFRASTRUCTURE, *appended)))


def _call_failures(
    ledger: _Ledger, stop: int, payload: dict[str, Any], late: bool
) -> tuple[tuple[str, str], ...]:
    """Results where the next tool call is missing: the deadline before it, or,
    unless the check before it must have failed (`late`), its (flat) record's
    failed append."""
    if late:
        return (DEADLINE,)
    size = _encoded_size(payload)
    return (DEADLINE, *_append_failures(ledger, stop, "tool_call", size, False))


def _submission_failures(
    episode: Episode, reply: Reply, ledger: _Ledger, stop: int, late: bool
) -> tuple[tuple[str, str], ...]:
    """Results the runner records when the submission after `reply` fails.

    A raising `submit_final` is classified before the deadline check; past it
    (`late`) nothing else runs. A raising `submission()` is classified before
    anything is persisted. `write_snapshot` refuses only bytes above the case's limit, as
    the planned run budget reserves the rest. Once the snapshot file exists, any
    failed `submission` append re-raises and leaves no `case_end`. Without a
    snapshot only that append can fail (see `_append_failures`).
    """
    try:
        if episode.final_answer:
            episode.submit_final(reply.content)
        if late:
            return ()
        metadata, data = episode.submission()
    except Exception:
        return (INFRASTRUCTURE,)
    if data is None:
        deep = _nesting(metadata) > SHALLOW_NESTING
        return _encode_failures(ledger, stop, "submission", metadata, deep)
    if len(data) > SNAPSHOT_LIMITS[ledger.events[stop]["case_id"]]:
        return (("output_limit", "snapshot_byte_budget"),)
    return ()


def _rejected(prior: int, record: Any = ...) -> tuple[tuple[str, str], ...]:
    """Results `classify` records for a rejected response after `prior` ones.

    Only an endpoint error is journaled so (the endpoint raises no model action
    error; tool arguments do, inside the loop): `{"rejected_raw": body}` for a
    body `parse_reply` refused, which is always a contract error, or
    `{"rejected_http": evidence}` for the transport's HTTP error evidence. That
    is a capacity rejection exactly when `is_capacity_rejection` says so, which
    never parses a body nested past the interpreter stack; an omitted record
    cannot settle it. A record the runner cannot write has no result.
    """
    capacity = (
        ("unsupported", "endpoint_request_capacity")
        if prior == 0
        else ("output_limit", "endpoint_capacity_rejection_after_responses")
    )
    if record is ...:
        return (capacity, ENDPOINT_ERROR)
    if not isinstance(record, dict) or len(record) != 1:
        return ()
    if "rejected_raw" in record:
        raw = record["rejected_raw"]
        if raw is None:
            return ()
        try:
            parse_reply(raw)
        except EndpointError:
            pass
        except Exception:
            # The runner's parse refused it; a parse that fails differently
            # here still leaves the one result `classify` records.
            return (ENDPOINT_ERROR,)
        else:
            return ()
        return (ENDPOINT_ERROR,)
    raw = record.get("rejected_http")
    if not is_http_evidence(raw):
        return ()
    return (capacity if is_capacity_rejection(raw) else ENDPOINT_ERROR,)


def _unrecorded_result(
    episode: Episode,
    call: ToolCall,
    position: int,
    timeout: float,
    ledger: _Ledger,
    stop: int,
    executions: _Executions | None = None,
) -> tuple[tuple[str, str], ...]:
    """Results of a recorded call whose result is not: it raised, or the result's
    append failed (see `_append_failures`). Answers the call, so the security
    counters include the attempt. Arguments nested past the fixed limit are
    malformed on any stack (`ToolCall.arguments`), so a call fails here exactly
    as in the agent episode."""
    try:
        result = answer_tool_call(episode, call, position, timeout=timeout)
    except EndpointError:
        return (ENDPOINT_ERROR,)
    except TimeoutError:
        return (DEADLINE,)
    except (OSError, RuntimeError, ValueError):
        return (INFRASTRUCTURE,)
    except Exception:
        raise Ungradable("tool_replay_error") from None
    if executions is not None and executions.take():
        # The execution raised, or its unknown result's append failed.
        failed = _append_failures(ledger, stop, "tool_result", None, False)
        return tuple(dict.fromkeys((INFRASTRUCTURE, *failed)))
    deep = _nesting(result) > SHALLOW_NESTING
    return _encode_failures(ledger, stop, "tool_result", result, deep)


def _nesting(value: Any) -> int:
    """Container depth of a value, or the bracket depth of JSON text outside its
    string literals, as deep as a parser of that text recurses."""
    if isinstance(value, str):
        return json_nesting(value)
    deepest, stack = 0, [(value, 0)]
    while stack:
        item, depth = stack.pop()
        deepest = max(deepest, depth)
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return deepest


def _response_bytes() -> int:
    """Bound on the payload the loop journals for a counted response: the
    response itself, or its marker above `RESPONSE_BYTES` (which a lowered
    limit may undercut)."""
    return max(agent_episode.RESPONSE_BYTES, RESPONSE_MARKER_BYTES)


def _omitted_response(
    events: list[dict[str, Any]],
    stop: int,
    index: int,
    marker: dict[str, Any],
    ledger: _Ledger,
    used: int,
    prior: int,
) -> _Stop:
    """An oversized, quota-omitted or emergency response ends the loop.

    Every marker in a response's place is the runner's: an "oversized" one
    within the limit, a quota marker the quota did not require, or an
    emergency one whose record could not cross the journal's ceiling is
    corrupted evidence; an emergency one only ends an error path. No body
    re-encodes larger than `REENCODED_RESPONSE_BYTES`. A counted response is
    journaled within `_response_bytes()`, a late record within
    `LATE_RECORD_OVERHEAD` more, and only a rejected-response record larger,
    whichever marker replaces it; an emergency marker replaces only the
    latter two. Its request measured a response for a counted one, the
    deadline for a late one and an endpoint error for a rejected one, which
    every rejected result names; a relabel may replace any of these results.
    """
    _ends(index, stop)
    end = events[stop]["payload"]
    reason, size = marker["omitted"], marker["bytes"]
    if (
        size > REENCODED_RESPONSE_BYTES
        or (reason == "case_journal_byte_budget" and not ledger.crosses(marker))
        or (reason == "response_byte_budget" and size <= agent_episode.RESPONSE_BYTES)
        or (
            reason == "segment_byte_budget"
            and not ledger.overflows(index - 1, "model_response", size)
        )
    ):
        raise Ungradable("malformed_omission")
    if reason == "segment_byte_budget":
        ledger.emergency = index - 1
    outcomes: tuple[tuple[str, str], ...]
    measured: tuple[str, ...]
    if (reason == "case_journal_byte_budget" and size > _response_bytes()) or (
        reason == "segment_byte_budget"
        and (end.get("status"), end.get("reason")) in ERROR_RESPONSE_ENDS
    ):
        if size > _response_bytes() + LATE_RECORD_OVERHEAD:
            # Only a rejected-response record, which counts no tokens.
            rejected = _rejected(prior)
            return _Stop(None, used, outcomes=rejected, measured=("endpoint_error",))
        outcomes = (*_rejected(prior), DEADLINE)
        measured = ("endpoint_error", "deadline")
    elif reason == "response_byte_budget":
        outcomes = (("output_limit", "response_byte_budget"), ACCOUNTING, DEADLINE)
        measured = ("response", "deadline")
    elif reason == "case_journal_byte_budget":
        outcomes = (DEFAULT_OUTCOME, ACCOUNTING, DEADLINE, *_rejected(prior))
        measured = ("response", "deadline", "endpoint_error")
    else:
        raise Ungradable("malformed_omission")
    # A rejected body counts no tokens; a late one only with the deadline,
    # whose result a relabel may replace.
    ended = (end.get("status"), end.get("reason"))
    if ended in REJECTED_ENDS or (
        reason == "segment_byte_budget"
        and end.get("status") != "timeout"
        and ended not in RELABELS
    ):
        return _Stop(None, used, outcomes=outcomes, measured=("endpoint_error",))
    return _Stop(None, used, "response", outcomes, measured)


def _call_payload(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments_raw": call.arguments_raw}


def _check_submission(
    episode: Episode,
    case_id: str,
    branch_id: str,
    recorded: dict[str, Any],
    snapshot: bytes | None,
) -> dict[str, Any]:
    """The submission must be the rebuilt one; returns it as the runner held it."""
    metadata, data = episode.submission()
    expected: dict[str, Any] = {}
    if data is not None:
        try:
            name = snapshot_name(case_id, branch_id)
        except JournalError:
            raise Ungradable("snapshot_contract") from None
        expected.update(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), snapshot=name
        )
    expected.update(metadata)
    if not _same(expected, recorded):
        raise Ungradable("submission_mismatch")
    if data is None:
        if snapshot is not None:
            raise Ungradable("unexpected_snapshot")
        return expected
    if snapshot is None:
        raise Ungradable("snapshot_missing")
    if hashlib.sha256(snapshot).hexdigest() != recorded.get("sha256") or len(
        snapshot
    ) != recorded.get("bytes"):
        raise Ungradable("snapshot_digest_mismatch")
    if snapshot != data:
        raise Ungradable("snapshot_mismatch")
    return expected


def _check_security(security: dict[str, int] | None, end: dict[str, Any]) -> None:
    """Replayed counters must equal the journaled `case_end` security record."""
    if "security" not in end or not _same(security, end["security"]):
        raise Ungradable("security_mismatch")


def _check_result(
    events: list[dict[str, Any]], expected: dict[str, Any], budget: tuple[int, int]
) -> None:
    """The journaled `case_end` must be the expected result record.

    Its elapsed time is not proven, only required to be a float duration within
    the journal's clock: the runner starts it before `case_start` and stops it
    after the last other event and before `case_end`, on the same clock. It
    must reach the deadline its result names: the agent budget for the agent
    deadline, which the loop and the model process record only once it passed,
    and the wall budget for the cleanup deadline, which `finalize` applies
    after it. A relabel may replace any result, a graded one included: a
    failed removal of the case directory, and the cleanup deadline, which must
    once the last other event lies past the wall budget, as `finalize` reads
    the clock after it. `budget` is the recorded (wall, verifier reserve) pair.
    """
    end = events[-1]["payload"]
    elapsed: Any = end.get("elapsed_seconds")
    span = events[-2]["elapsed_seconds"] - events[0]["elapsed_seconds"]
    if not _duration(elapsed) or not (
        span - CLOCK_TOLERANCE <= elapsed <= events[-1]["elapsed_seconds"]
    ):
        raise Ungradable("result_mismatch")
    wall, reserve = budget
    cleanup = ("timeout", "cleanup_deadline")
    if span > wall + CLOCK_TOLERANCE and not _same(
        cleanup, (end.get("status"), end.get("reason"))
    ):
        raise Ungradable("result_mismatch")
    deadlines = {"agent_deadline": wall - reserve, "cleanup_deadline": wall}
    if end.get("status") == "timeout" and isinstance(end.get("reason"), str):
        deadline = deadlines.get(end["reason"])
        if deadline is not None and elapsed < deadline - CLOCK_TOLERANCE:
            raise Ungradable("result_mismatch")
    for relabel in RELABELS:
        if _same(relabel, (end.get("status"), end.get("reason"))):
            expected.update(status=relabel[0], reason=relabel[1], strict_success=0)
    expected["elapsed_seconds"] = elapsed
    # Measured times are not reproducible; they must be consistent with the
    # result, the journal's clock and the journaled usage instead.
    expected.update(requests=end.get("requests"), timing=end.get("timing"))
    if not _same(expected, end):
        raise Ungradable("result_mismatch")
    # `_check_provenance` settled the mode before the loop replayed.
    streaming = events[0]["payload"]["streaming"]
    try:
        cached = _cached(events)
        validate_measurements(end, streaming=streaming, cached=cached)
        check_journal(events, end, streaming=streaming, cached=cached)
    except (ValueError, TypeError, KeyError):
        raise Ungradable("measurement_mismatch") from None


def _expected_result(
    events: list[dict[str, Any]],
    case_id: str,
    branch_id: str,
    episode: Episode,
    outcome: tuple[Any, Any, Any],
    verification: list[int],
) -> dict[str, Any]:
    """The `case_end` record `result_record` writes for this outcome."""
    state = EpisodeState()
    state.status, state.reason, state.used = outcome
    return result_record(
        state,
        run_id=events[0]["run_id"],
        case_id=case_id,
        branch_id=branch_id,
        contract=CASES[case_id],
        episode=episode,
        started=0.0,
        evidence_sequences=verification,
        schema_version=CONTRACTS[
            events[0]["payload"]["run_contract_version"]
        ].result_schema,
    )


def _check_ungraded_result(
    end: dict[str, Any], stop: _Stop, exhausted: bool, deadline: float
) -> None:
    """A segment without a graded submission records where its loop stopped.

    Its status and reason must be one the runner records at the replayed stop,
    after `finalize` (an exhausted quota relabels any result that is not an
    error, the segment budget wins, and a relabel may replace any result
    recorded there). A relabel needs a result to replace, so none applies
    where the loop cannot stop. The cleanup deadline needs the wall budget
    (see `_check_result`); a failed removal of the case directory needs what
    the result it replaced needs: the agent `deadline` for that result, and a
    timeout for a request measured at the deadline, which no other result
    ends. Its tokens must be the replayed ones, or at least those where a
    counted response may be missing.
    """
    status, reason, tokens = (
        end.get("status"),
        end.get("reason"),
        end.get("output_tokens"),
    )
    elapsed: Any = end.get("elapsed_seconds")
    due = _duration(elapsed) and elapsed >= deadline - CLOCK_TOLERANCE
    requests = end.get("requests")
    late = (
        isinstance(requests, list)
        and bool(requests)
        and isinstance(requests[-1], dict)
        and requests[-1].get("outcome") == "deadline"
    )
    finals = []
    if stop.outcomes:
        finals.append(CLEANUP_DEADLINE)
        if any(due if each == DEADLINE else not late for each in stop.outcomes):
            finals.append(REMOVAL_FAILED)
    for outcome in stop.outcomes:
        if late and outcome[0] != "timeout":
            continue
        if (
            exhausted
            and outcome != SEGMENT
            and outcome[0] not in QUOTA_PRIORITY_STATUSES
            and outcome[1] not in QUOTA_PRIORITY_REASONS
        ):
            outcome = ("output_limit", "case_journal_byte_budget")
        finals.append(outcome)
    if (
        not isinstance(status, str)
        or not isinstance(reason, str)
        or (status, reason) not in finals
        or type(tokens) is not int
        or tokens < stop.used
        or (tokens != stop.used and (not stop.hidden or (status, reason) == ACCOUNTING))
    ):
        raise Ungradable("result_mismatch")


def _check_agent_end(
    events: list[dict[str, Any]],
    expected: dict[str, Any],
    wall: int,
) -> dict[str, Any]:
    """The journaled `agent_end` must be the expected agent record: the result
    record without its classification, its measurements consistent with the
    journal (see `_check_result`), and within the wall budget, past which
    `finalize` would have relabeled it. Returns it."""
    agent = events[-1]["payload"]
    elapsed: Any = agent.get("elapsed_seconds")
    span = events[-2]["elapsed_seconds"] - events[0]["elapsed_seconds"]
    if (
        not _duration(elapsed)
        or not span - CLOCK_TOLERANCE <= elapsed <= events[-1]["elapsed_seconds"]
        or span > wall + CLOCK_TOLERANCE
    ):
        raise Ungradable("agent_end_mismatch")
    expected.update(
        elapsed_seconds=elapsed,
        requests=agent.get("requests"),
        timing=agent.get("timing"),
    )
    if not _same(expected, agent):
        raise Ungradable("agent_end_mismatch")
    streaming = events[0]["payload"]["streaming"]
    status, reason = agent_episode.DEFERRED
    try:
        cached = _cached(events)
        validate_measurements(agent, streaming=streaming, cached=cached)
        check_journal(
            events,
            {**agent, "status": status, "reason": reason},
            streaming=streaming,
            cached=cached,
        )
    except (ValueError, TypeError, KeyError):
        raise Ungradable("measurement_mismatch") from None
    return agent


def _verify(
    episode: Episode,
    events: list[dict[str, Any]],
    case_id: str,
    branch_id: str,
    agent: dict[str, Any],
    reserve: int,
) -> Verified:
    """Verify a code episode's submission once, within the verifier reserve,
    and derive its result record from the agent phase's record."""
    started = time.monotonic()
    verification = failure = None
    try:
        checks = episode.verify(timeout=reserve)
        verification = verification_payload(episode, checks)
        status, reason = graded_status(episode, checks)
    except TimeoutError as error:
        status, reason, failure = (*VERIFICATION_DEADLINE, type(error).__name__)
    except (OSError, RuntimeError, ValueError) as error:
        status, reason, failure = (*INFRASTRUCTURE, type(error).__name__)
    seconds = time.monotonic() - started
    result = _expected_result(
        events,
        case_id,
        branch_id,
        episode,
        (status, reason, agent["output_tokens"]),
        agent["evidence_event_sequences"],
    )
    result.update(
        elapsed_seconds=agent["elapsed_seconds"],
        requests=agent["requests"],
        timing=agent["timing"],
    )
    return Verified(verification, result, seconds, failure)


def replay(
    case_id: str,
    branch_id: str,
    events: list[dict[str, Any]],
    snapshot: bytes | None = None,
    *,
    verify_timeout: float,
    executor: Executor | None = None,
) -> Grade | NotGraded | Verified:
    """Regrade one complete journaled episode segment (`case_start` to `case_end`).

    `snapshot` is the stored submission file for cases that write one.
    `verify_timeout` bounds verifier and tool calls that accept a timeout. A
    code case needs `executor`, which verifies its submission within the
    recorded verifier reserve (`Verified`).
    """
    if (
        type(verify_timeout) not in {int, float}
        or not math.isfinite(verify_timeout)
        or verify_timeout <= 0
    ):
        raise ValueError("Positive finite verifier timeout required")
    if snapshot is not None and type(snapshot) is not bytes:
        raise ValueError("Snapshot bytes required")
    if not isinstance(case_id, str) or case_id not in CATALOG or case_id not in CASES:
        raise Ungradable("unknown_case")
    spec = CATALOG[case_id]
    if not isinstance(branch_id, str) or branch_id not in spec.branches:
        raise Ungradable("unknown_branch")
    code = spec.sandbox
    if code and executor is None:
        raise Ungradable("sandbox_case")
    submission_index = _check_structure(case_id, branch_id, events, code=code)
    cap, turns, per_call = _check_contract(case_id, events[0]["payload"], code=code)
    start = events[0]["payload"]
    budget = (start["wall_seconds"], start["verifier_reserve_seconds"])
    _check_quota(events)
    ledger = _Ledger(events)
    submitted = submission_index is not None
    end = events[-1]["payload"]
    start = 1
    if (
        code
        and len(events) > 2
        and events[1]["kind"] == "tool_result"
        and events[1]["trust"] == "trusted_runner"
    ):
        _check_preflight(events[1]["payload"])
        ledger.admit(1, events[1]["payload"])
        start = 2
    executions = _Executions(executor) if executor is not None and code else None
    with tempfile.TemporaryDirectory(prefix="werkbank-replay-") as temporary:
        episode = spec.build(EpisodeContext(branch_id, Path(temporary), executions))
        deadline = budget[0] - budget[1]
        _check_deadlines(events, episode.final_answer, deadline, budget[0])
        replayed = _replay_loop(
            episode,
            events,
            len(events) - 1 if submission_index is None else submission_index,
            submitted=submitted,
            ledger=ledger,
            cap=cap,
            turns=turns,
            per_call=per_call,
            timeout=verify_timeout,
            deadline=deadline,
            executions=executions,
            start=start,
        )
        if submission_index is None:
            ledger.check()
            if snapshot is not None:
                raise Ungradable("unexpected_snapshot")
            security = security_record(episode, CASES[case_id])
            _check_security(security, end)
            _check_ungraded_result(end, replayed, ledger.exhausted, deadline)
            outcome = (end["status"], end["reason"], end["output_tokens"])
            _check_result(
                events,
                _expected_result(events, case_id, branch_id, episode, outcome, []),
                budget,
            )
            # Which records a marker can replace, its reason and the response
            # limit say; report validation knows the reason, not the limit.
            measured = replayed.measured
            if measured and end["requests"][-1]["outcome"] not in measured:
                raise Ungradable("measurement_mismatch")
            return NotGraded(security=security)
        final, used = replayed.reply, replayed.used
        if final is None:
            raise Ungradable("event_order_violation")
        if episode.final_answer:
            episode.submit_final(final.content)
        submission = _check_submission(
            episode,
            case_id,
            branch_id,
            events[submission_index]["payload"],
            snapshot,
        )
        ledger.wrote(submission_index, submission)
        ledger.check()
        if executions is not None:
            executions.recording = False
            evidence = [events[submission_index]["sequence"]]
            deferred = _expected_result(
                events,
                case_id,
                branch_id,
                episode,
                (*agent_episode.DEFERRED, used),
                evidence,
            )
            if events[-1]["kind"] == "case_end":
                # Only a relabel ends the agent phase after a submission.
                if not any(
                    _same(relabel, (end.get("status"), end.get("reason")))
                    for relabel in RELABELS
                ):
                    raise Ungradable("result_mismatch")
                security = security_record(episode, CASES[case_id])
                _check_security(security, end)
                _check_result(events, deferred, budget)
                return NotGraded(security=security, reason="relabeled")
            agent = _check_agent_end(
                events, agent_episode.agent_record(deferred), budget[0]
            )
            return _verify(episode, events, case_id, branch_id, agent, budget[1])
        try:
            checks = episode.verify(timeout=verify_timeout)
            verification = verification_payload(episode, checks)
            status, reason = graded_status(episode, checks)
            security = security_record(episode, CASES[case_id])
        except Exception:
            raise Ungradable("verifier_error") from None
    recorded = events[submission_index + 1]
    if not _same(verification, recorded["payload"]):
        raise Ungradable("verification_mismatch")
    _check_security(security, end)
    expected = _expected_result(
        events,
        case_id,
        branch_id,
        episode,
        (status, reason, used),
        [recorded["sequence"]],
    )
    _check_result(events, expected, budget)
    return Grade(verification, status, reason)
