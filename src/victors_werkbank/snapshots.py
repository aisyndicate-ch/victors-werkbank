"""Bounded submission-file persistence and integrity, never execution or regrading."""

import hashlib
import os
from pathlib import Path
from typing import Any

from .journal import ByteBudget, JournalError

# L03 keeps the last answer of each asked service: two paths of at most 8
# fixed-form source IDs each and a region of at most 16 characters (6-byte
# escapes in canonical ASCII JSON), about 3 KB in the worst case
# (`tests/test_snapshots.py`). C04 escapes three and C07 two
# 64 KiB source files in an ASCII JSON bundle, C01 its one 16 KiB module and
# C10 its three; single-file S04 stays 64 KiB.
SNAPSHOT_LIMITS = {
    "L03": 64 * 1024,
    "C04": 2 * 1024 * 1024,
    "C07": 1024 * 1024,
    "S04": 64 * 1024,
    "C08": 1024 * 1024,
    "C01": 128 * 1024,
    "C09": 1024 * 1024,
    "C10": 512 * 1024,
}


class SnapshotLimitError(JournalError):
    """Supported snapshot byte allowance exceeded before writing."""


class SnapshotWriteError(OSError):
    """Durability is uncertain; leave the run interrupted, never publish it."""


def snapshot_name(case_id: str, branch_id: str) -> str:
    if case_id not in SNAPSHOT_LIMITS:
        raise JournalError("Case has no supported snapshot contract")
    if case_id == "S04":
        if branch_id not in {"attack", "benign"}:
            raise JournalError("Invalid snapshot branch")
        return f"S04-{branch_id}-submission.txt"
    if branch_id != "default":
        raise JournalError("Invalid snapshot branch")
    return f"{case_id}-submission.{'json' if case_id == 'L03' else 'txt'}"


def validate_descriptor(value: Any, case_id: str, branch_id: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "bytes", "sha256"}
        or value["path"] != snapshot_name(case_id, branch_id)
        or type(value["bytes"]) is not int
        or not 0 <= value["bytes"] <= SNAPSHOT_LIMITS[case_id]
        or not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["sha256"])
    ):
        raise JournalError("Invalid snapshot descriptor or supported byte limit")


def write_snapshot(
    directory: Path,
    case_id: str,
    branch_id: str,
    data: bytes,
    budget: ByteBudget | None,
) -> dict[str, Any]:
    name = snapshot_name(case_id, branch_id)
    if len(data) > SNAPSHOT_LIMITS[case_id]:
        raise SnapshotLimitError("Snapshot exceeds supported byte limit")
    # Charge before creating a file; failed writes never refund uncertain bytes.
    if budget is not None:
        try:
            budget.charge(len(data))
        except JournalError as error:
            raise SnapshotLimitError(str(error)) from error
    try:
        fd = os.open(
            directory / name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("Snapshot write made no progress")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        directory_fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise SnapshotWriteError("Snapshot persistence failed") from error
    return {
        "path": name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
