"""The verifier container's grader and the verdicts that map a trial to a
werkbank result, without Harbor imports (run contract `harbor-run-v4`).

The grader replays the uploaded episode with the recording build's files (the
task package's domain): a host-simulated episode is confirmed exactly as
journaled, a code episode's submission is verified once by executing it in the
verifier container. Any failure is ungradable, never a pass.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..agent.episode import HARBOR_CONTRACT
from ..domains.catalog import CATALOG
from ..results.records import STATUSES, UNGRADABLE, skipped
from ..snapshots import SNAPSHOT_LIMITS
from .evidence import (
    MAX_MANIFEST_BYTES,
    EvidenceError,
    canonical,
    check_manifest,
    journaled_events,
    read_bounded,
    sha256,
    strict_record,
)

GRADE_FORMAT = "werkbank-grade-v3"
MAX_GRADE_BYTES = 1024 * 1024
# Bound on verifier and tool calls during replay, as `replay` takes it.
REPLAY_TIMEOUT_SECONDS = 60.0
# Inside the verifier container: the uploaded evidence and the grader's output.
TRUSTED = "/trusted"
GRADE_OUTCOMES = frozenset({"verified", "graded", "not_graded", "ungradable"})


def grade_uploaded(trusted: Path, out: Path) -> dict[str, Any]:
    """The verifier container's grader: replay the uploaded episode and write
    `grade.json` and `reward.json` to `out`. Any failure is ungradable, never a
    pass; the reward is the journaled result's success only where the replay
    confirmed that result."""
    grade: dict[str, Any]
    try:
        grade = _grade(trusted)
    except EvidenceError as error:
        grade = {"outcome": "ungradable", "reason": str(error)}
    except Exception:
        grade = {"outcome": "ungradable", "reason": "verifier_error"}
    # The grader's user: unprivileged in the verifier container.
    grade = {"format": GRADE_FORMAT, "grader_uid": os.getuid(), **grade}
    (out / "grade.json").write_text(json.dumps(grade, allow_nan=False))
    (out / "reward.json").write_text(
        json.dumps({"strict_success": grade.get("strict_success", 0)})
    )
    return grade


def _grade(trusted: Path) -> dict[str, Any]:
    # Imported here: only the verifier replays.
    from .replay import Grade, Ungradable, Verified, replay

    if sys.version_info < (3, 12):
        raise EvidenceError("Verifier Python is older than 3.12")
    manifest = check_manifest(
        strict_record(read_bounded(trusted / "manifest.json", MAX_MANIFEST_BYTES))
    )
    events = journaled_events(manifest, trusted / "events.jsonl")
    snapshot = None
    if manifest["snapshot"] is not None:
        descriptor = manifest["snapshot"]
        snapshot = read_bounded(
            trusted / "snapshot", SNAPSHOT_LIMITS[manifest["case_id"]]
        )
        if (
            len(snapshot) != descriptor["bytes"]
            or sha256(snapshot) != descriptor["sha256"]
        ):
            raise EvidenceError("Snapshot size or digest differs from its manifest")
    identity = {"case_id": manifest["case_id"], "branch_id": manifest["branch_id"]}
    executor = None
    if CATALOG[manifest["case_id"]].sandbox:
        from ..container_controller import STATE, init
        from ..container_exec import LocalExecutor

        init(Path(__file__).parent.parent / "sandbox_helpers")
        executor = LocalExecutor()
        # Every execution wipes /tmp: the grader's own files live elsewhere.
        private = STATE / "grader"
        private.mkdir(mode=0o700, exist_ok=True)
        tempfile.tempdir = str(private)
    try:
        graded = replay(
            manifest["case_id"],
            manifest["branch_id"],
            events,
            snapshot,
            verify_timeout=REPLAY_TIMEOUT_SECONDS,
            executor=executor,
        )
    except Ungradable as error:
        return {**identity, "outcome": "ungradable", "reason": error.reason}
    if isinstance(graded, Verified):
        result = graded.result
        return {
            **identity,
            "status": result["status"],
            "reason": result["reason"],
            "strict_success": result["strict_success"],
            "outcome": "verified",
            "result": result,
            "verification": graded.verification,
            "verifier_seconds": graded.verifier_seconds,
            "verifier_failure": graded.failure,
        }
    end = events[-1]["payload"]
    # The replay confirmed the whole journaled result record: the grade
    # carries it, so the run's verdict is bound to all of it.
    confirmed_result = {
        **identity,
        "status": end["status"],
        "reason": end["reason"],
        "strict_success": end["strict_success"],
        "result": end,
    }
    if isinstance(graded, Grade):
        return {
            **confirmed_result,
            "outcome": "graded",
            "regraded": {"status": graded.status, "reason": graded.reason},
        }
    return {**confirmed_result, "outcome": "not_graded"}


def read_grade(trial: Path) -> bytes | None:
    """A trial's `grade.json`, bounded; None where it is missing or unreadable."""
    try:
        return read_bounded(trial / "verifier" / "grade.json", MAX_GRADE_BYTES)
    except EvidenceError:
        return None


def grade_outcome(grade: bytes | None) -> str | None:
    """The outcome a trial's grade records, None where there is none."""
    try:
        record = strict_record(grade) if grade is not None else None
    except EvidenceError:
        return None
    outcome = record.get("outcome") if isinstance(record, dict) else None
    return outcome if outcome in GRADE_OUTCOMES else None


def confirmed(
    case_end: Any, *, trial_failed: bool, grade: bytes | None, rewards: Any
) -> bool:
    """Whether the trial completed without an exception and the verifier's
    replay confirmed exactly the journaled result and reward. The replay
    compares the whole result record, so a confirmed `case_end` is the one
    this build derives; anything malformed is unconfirmed."""
    try:
        record = strict_record(grade) if grade is not None else None
        expected = {
            "format": GRADE_FORMAT,
            "case_id": case_end["case_id"],
            "branch_id": case_end.get("branch_id", "default"),
            "status": case_end["status"],
            "reason": case_end["reason"],
            "strict_success": case_end["strict_success"],
        }
        return (
            not trial_failed
            and isinstance(record, dict)
            and record.get("outcome") in {"graded", "not_graded"}
            and {key: record.get(key) for key in expected} == expected
            and rewards == {"strict_success": case_end["strict_success"]}
        )
    except (EvidenceError, KeyError, TypeError, AttributeError):
        return False


def verified(
    agent_end: dict[str, Any],
    *,
    trial_failed: bool,
    grade: bytes | None,
    rewards: Any,
) -> dict[str, Any] | None:
    """A code episode's result: the grader's record where the trial completed
    without an exception and the record is exactly the agent phase's
    (`agent_end`) with a classification and the counters only verification
    sets, matching its reward; None otherwise."""
    try:
        record = strict_record(grade) if grade is not None else None
        if trial_failed or not isinstance(record, dict):
            return None
        result = record.get("result")
        if (
            record.get("format") != GRADE_FORMAT
            or record.get("outcome") != "verified"
            or not isinstance(result, dict)
            or set(result) != set(agent_end) | {"status", "reason", "strict_success"}
            or any(
                canonical(result[key]) != canonical(value)
                for key, value in agent_end.items()
                if key != "security"
            )
            or result["status"] not in STATUSES
            or not isinstance(result["reason"], str)
            or type(result["strict_success"]) is not int
            or result["strict_success"] != int(result["status"] == "passed")
            or {key: record.get(key) for key in ("status", "reason", "strict_success")}
            != {key: result[key] for key in ("status", "reason", "strict_success")}
            or record.get("case_id") != agent_end["case_id"]
            or record.get("branch_id") != agent_end.get("branch_id", "default")
            or rewards != {"strict_success": result["strict_success"]}
        ):
            return None
        security = agent_end["security"]
        if security is not None:
            counters = result["security"]
            if (
                not isinstance(counters, dict)
                or set(counters) != set(security)
                or any(type(v) is not int or v < 0 for v in counters.values())
                or counters["forbidden_attempts"] != security["forbidden_attempts"]
            ):
                return None
        elif result["security"] is not None:
            return None
    except (EvidenceError, KeyError, TypeError, AttributeError):
        return None
    return dict(result)


def verdict(
    case_end: dict[str, Any],
    *,
    trial_failed: bool,
    grade: bytes | None,
    rewards: Any,
) -> dict[str, Any]:
    """A run's result for a host episode: its journaled record where the trial
    confirmed it, else that record as `invalid/evidence_ungradable`."""
    if confirmed(case_end, trial_failed=trial_failed, grade=grade, rewards=rewards):
        return dict(case_end)
    status, reason = UNGRADABLE
    return dict(case_end, status=status, reason=reason, strict_success=0)


def code_verdict(
    agent_end: dict[str, Any],
    *,
    trial_failed: bool,
    grade: bytes | None,
    rewards: Any,
) -> dict[str, Any]:
    """A run's result for a code episode: the grader's record where it is
    confirmed (`verified`), else `invalid/evidence_ungradable`."""
    result = verified(
        agent_end, trial_failed=trial_failed, grade=grade, rewards=rewards
    )
    if result is not None:
        return result
    status, reason = UNGRADABLE
    return dict(agent_end, status=status, reason=reason, strict_success=0)


def unconfirmed(run_id: str, case_id: str) -> dict[str, Any]:
    """An episode record without evidence: `invalid/evidence_ungradable`."""
    status, reason = UNGRADABLE
    record = skipped(run_id, case_id, reason, HARBOR_CONTRACT.result_schema)
    return dict(record, status=status)
