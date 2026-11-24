"""Evidence of one episode, admitted without Harbor: its manifest, journal and
snapshot under `episodes/<name>/` of a run (run contract `harbor-run-v6`).

Every file is a bounded regular file reached without following a link, with
the exact size and digest its manifest names; anything else is ungradable,
never a pass. The same rules admit the evidence on the host before upload and
in the verifier container after it.
"""

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from ..agent.transport import strict_json
from ..domains.catalog import CATALOG
from ..journal import JournalError, read_events
from ..registry import CASES, lifecycle
from ..snapshots import SNAPSHOT_LIMITS, validate_descriptor

MANIFEST_FORMAT = "werkbank-manifest-v3"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024
JOURNAL = "events.jsonl"
MANIFEST = "manifest.json"
MANIFEST_KEYS = frozenset(
    {
        "format",
        "run_id",
        "position",
        "case_id",
        "branch_id",
        "journal",
        "bytes",
        "sha256",
        "events",
        "start_sequence",
        "snapshot",
    }
)


class EvidenceError(ValueError):
    """Required evidence is missing, linked, oversized or corrupt: ungradable."""


def read_bounded(path: Path, maximum: int) -> bytes:
    """The bytes of the regular file at `path`, reached without following a
    link, refused above `maximum` bytes.

    A read buffer is as large as the read it is asked for, so the file is read
    for its size plus one byte, not for its bound: a small file under a large
    bound costs only its own bytes. Only where that extra byte arrives, so the
    file grew after its size was taken, is it read again from its start, once
    the first bytes are released, for one byte past its bound: no read and no
    held buffer is larger than the bound plus one byte, and a file past its
    bound is refused."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise EvidenceError("Required evidence is missing or linked") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise EvidenceError("Required evidence is not a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(info.st_size + 1)
            if info.st_size < len(data) <= maximum:
                # Read again rather than join a follow-on read: a join holds
                # both parts and their result at once, up to twice the bound.
                del data
                stream.seek(0)
                data = stream.read(maximum + 1)
    finally:
        os.close(fd)
    if len(data) > maximum:
        raise EvidenceError("Evidence grew past its bound during read")
    return data


def strict_record(data: bytes) -> Any:
    try:
        return strict_json(data.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise EvidenceError("Evidence is not strict JSON") from None


def read_record(path: Path) -> Any:
    """A run record read strictly and bounded."""
    return strict_record(read_bounded(path, MAX_RECORD_BYTES))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def is_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(record: Any) -> bytes:
    return (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()


def write_exclusive(path: Path, data: bytes) -> None:
    """Create `path` exclusively, durable before it is visible as complete."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("Write made no progress")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def tree_sha256(root: Path) -> str:
    """Digest of every directory's and regular file's relative path and every
    file's bytes, each length-prefixed so no two trees share it; a link or any
    other file type is refused."""
    digest = hashlib.sha256()

    def field(data: bytes) -> None:
        digest.update(len(data).to_bytes(8, "big") + data)

    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        name = str(path.relative_to(root)).encode()
        if stat.S_ISDIR(mode):
            digest.update(b"d")
            field(name)
        elif stat.S_ISREG(mode):
            digest.update(b"f")
            field(name)
            field(path.read_bytes())
        else:
            raise EvidenceError("Task package holds a non-regular file")
    return digest.hexdigest()


def check_manifest(manifest: Any) -> dict[str, Any]:
    """Raise EvidenceError unless `manifest` is an episode manifest; returns it."""
    try:
        if (
            not isinstance(manifest, dict)
            or set(manifest) != MANIFEST_KEYS
            or manifest["format"] != MANIFEST_FORMAT
            or not isinstance(manifest["run_id"], str)
            or type(manifest["position"]) is not int
            or manifest["position"] < 0
            or manifest["case_id"] not in CASES
            or manifest["case_id"] not in CATALOG
            or manifest["branch_id"] not in CASES[manifest["case_id"]].branches
            or manifest["journal"] != JOURNAL
            or type(manifest["bytes"]) is not int
            or not 0
            < manifest["bytes"]
            <= lifecycle(manifest["case_id"], manifest["branch_id"]).journal_limit
            or not is_digest(manifest["sha256"])
            or type(manifest["events"]) is not int
            or manifest["events"] < 2
            or type(manifest["start_sequence"]) is not int
            or manifest["start_sequence"] < 0
        ):
            raise EvidenceError("Invalid episode manifest")
        # A snapshot case without a submission has none; the replay requires
        # one exactly where the journal holds the submission.
        snapshot = manifest["snapshot"]
        if snapshot is not None:
            validate_descriptor(snapshot, manifest["case_id"], manifest["branch_id"])
    except (JournalError, TypeError):
        raise EvidenceError("Invalid episode manifest") from None
    return manifest


def episode_manifest(
    run_id: str,
    position: int,
    case_id: str,
    branch_id: str,
    journal: Any,
    start_sequence: int,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """The manifest of an episode whose journal `journal` is closed."""
    return check_manifest(
        {
            "format": MANIFEST_FORMAT,
            "run_id": run_id,
            "position": position,
            "case_id": case_id,
            "branch_id": branch_id,
            "journal": JOURNAL,
            "bytes": journal.bytes_written,
            "sha256": journal.digest.hexdigest(),
            "events": len(journal.events),
            "start_sequence": start_sequence,
            "snapshot": dict(snapshots[0]) if snapshots else None,
        }
    )


Admitted = tuple[dict[str, Any], bytes, bytes, bytes | None]


def read_evidence(directory: Path) -> Admitted:
    """Admit one episode's evidence directory: its manifest, journal and
    snapshot. Returns the manifest, its bytes, the journal and the snapshot
    (None where the case writes none)."""
    if directory.is_symlink() or not directory.is_dir():
        raise EvidenceError("Missing or linked evidence directory")
    data = read_bounded(directory / MANIFEST, MAX_MANIFEST_BYTES)
    manifest = check_manifest(strict_record(data))
    limit = lifecycle(manifest["case_id"], manifest["branch_id"]).journal_limit
    journal = read_bounded(directory / manifest["journal"], limit)
    if len(journal) != manifest["bytes"] or sha256(journal) != manifest["sha256"]:
        raise EvidenceError("Journal size or digest differs from its manifest")
    snapshot = None
    descriptor = manifest["snapshot"]
    if descriptor is not None:
        snapshot = read_bounded(
            directory / descriptor["path"], SNAPSHOT_LIMITS[manifest["case_id"]]
        )
        if (
            len(snapshot) != descriptor["bytes"]
            or sha256(snapshot) != descriptor["sha256"]
        ):
            raise EvidenceError("Snapshot size or digest differs from its manifest")
    return manifest, data, journal, snapshot


def journaled_events(manifest: dict[str, Any], journal: Path) -> list[dict[str, Any]]:
    """The episode's events, read strictly from an admitted journal file: one
    `case_start` through one `case_end` of the manifest's case and run, or
    through `agent_end` for a code case."""
    try:
        events = read_events(
            journal,
            start_sequence=manifest["start_sequence"],
            expected_bytes=manifest["bytes"],
            expected_sha256=manifest["sha256"],
        )
    except JournalError:
        raise EvidenceError("Journal is not a complete episode") from None
    kinds = [e["kind"] for e in events]
    code = CATALOG[manifest["case_id"]].sandbox
    ends = ("case_end", "agent_end") if code else ("case_end",)
    if (
        len(events) != manifest["events"]
        or kinds[0] != "case_start"
        or kinds[-1] not in ends
        or kinds.count("case_start") != 1
        or sum(kinds.count(end) for end in ends) != 1
        or any(e["case_id"] != manifest["case_id"] for e in events)
        or any(e["run_id"] != manifest["run_id"] for e in events)
        or events[0]["payload"].get("branch_id") != manifest["branch_id"]
    ):
        raise EvidenceError("Journal is not the manifest's episode")
    return events
