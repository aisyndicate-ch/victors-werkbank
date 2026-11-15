"""The submission archive: the envelope, the deterministic tar archive with
gzip compression, and the bounded reader every reader of a submission uses.

The tar archive (POSIX pax format, written without extended headers) holds
`submission.json` first and then every other member in sorted path order,
regular files only, each with mode 0644, owner and group 0, empty owner and
group names and modification time 0. The gzip layer has a fixed header
(modification time 0, no file name), so the same members give the same bytes
with one zlib. The reader counts while it reads, refuses at the first
exceeded limit and returns nothing before the whole archive has passed.
"""

import io
import re
import struct
import tarfile
import zlib
from typing import Any

from victors_werkbank import __version__
from victors_werkbank.agent.episode import harness_digest
from victors_werkbank.grading.evidence import encode, sha256

from .texts import CONSENT_FORMAT, QUESTIONS, QUESTIONS_FORMAT, SUBMISSION_FORMAT

MAX_BODY_BYTES = 67_108_864
MAX_MEMBERS = 10_000
MAX_UNPACKED_BYTES = 2_147_483_648

ENVELOPE = "submission.json"
# Every other member lies under one of these directories.
TOP = ("evaluation", "result", "raw")
# Member paths: segments of letters, digits, `.`, `_` and `-`, never `.` or
# `..`, at most 100 bytes in all, so no member needs an extended header.
MAX_NAME_BYTES = 100
_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")
# Magic, deflate, no flags, modification time 0, maximum compression, no
# operating system named: the same header on every platform.
GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
_TRAILER = 8
_BLOCK = 512
_RECORD = 20 * _BLOCK


class ArchiveError(ValueError):
    """An archive that breaks the submission archive rules or limits."""


def plain_path(name: str) -> bool:
    """A relative path of plain segments, none of them `.` or `..`."""
    segments = name.split("/")
    return all(
        _SEGMENT.fullmatch(part) is not None and part not in {".", ".."}
        for part in segments
    )


def safe_name(name: str) -> bool:
    """A plain relative path within the member name limit."""
    return len(name.encode("utf-8")) <= MAX_NAME_BYTES and plain_path(name)


def _member_name(name: str) -> bool:
    top, _, rest = name.partition("/")
    return top in TOP and bool(rest) and safe_name(name)


def envelope(
    *,
    installation_id: str,
    evaluation: dict[str, Any],
    evaluation_sha256: str,
    case_set: dict[str, Any],
    answers: dict[str, str | None],
    raw_data: bool,
    members: list[tuple[str, bytes]],
) -> bytes:
    """`submission.json` for the other `members`, its fields in their fixed
    order: two-space indented ASCII JSON with a final newline."""
    record = {
        "format": SUBMISSION_FORMAT,
        "installation_id": installation_id,
        "werkbank": {"version": __version__, "harness_sha256": harness_digest()},
        "evaluation": {"format": evaluation["format"], "sha256": evaluation_sha256},
        "case_set": {"name": case_set["name"], "case_ids": list(case_set["case_ids"])},
        "runs": [
            {key: item[key] for key in ("run_id", "plan_sha256", "run_sha256")}
            for item in evaluation["inputs"]
        ],
        "answers": {
            "format": QUESTIONS_FORMAT,
            **{field: answers.get(field) for field, _, _ in QUESTIONS},
        },
        "consent": {"text": CONSENT_FORMAT, "raw_data": raw_data},
        "files": [
            {"path": path, "bytes": len(data), "sha256": sha256(data)}
            for path, data in sorted(members)
        ],
    }
    return encode(record)


def _check_members(members: list[tuple[str, bytes]]) -> None:
    if not members or members[0][0] != ENVELOPE:
        raise ArchiveError("The archive does not start with submission.json")
    if len(members) > MAX_MEMBERS:
        raise ArchiveError("The archive holds more than 10,000 members")
    names = [name for name, _ in members[1:]]
    if not all(_member_name(name) for name in names):
        raise ArchiveError("An archive member has an unsafe name")
    if any(first >= second for first, second in zip(names, names[1:])):
        raise ArchiveError("Archive members are not in sorted, distinct order")
    if sum(len(data) for _, data in members) > MAX_UNPACKED_BYTES:
        raise ArchiveError("The archive exceeds 2 GiB unpacked")


def _info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.type = tarfile.REGTYPE
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def tar_bytes(members: list[tuple[str, bytes]]) -> bytes:
    """The uncompressed tar archive of `members` (`submission.json` first,
    the rest sorted); ArchiveError beyond the member, name or size limits."""
    _check_members(members)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in members:
            tar.addfile(_info(name, len(data)), io.BytesIO(data))
    return buffer.getvalue()


def compress(tar: bytes) -> bytes:
    """gzip with the fixed header: one deflate stream, its CRC-32 and size."""
    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    body = GZIP_HEADER + deflate.compress(tar) + deflate.flush()
    return body + struct.pack("<II", zlib.crc32(tar), len(tar) & 0xFFFFFFFF)


def build(members: list[tuple[str, bytes]]) -> bytes:
    return compress(tar_bytes(members))


def _decompress(body: bytes, limit: int) -> bytes:
    """The tar bytes of a gzip body with exactly the fixed header, one member
    and nothing after it; refused once they would exceed `limit` bytes, which
    is also the most that is ever unpacked."""
    if len(body) < len(GZIP_HEADER) + _TRAILER or not body.startswith(GZIP_HEADER):
        raise ArchiveError("The body is not gzip with the fixed header")
    inflate = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        tar = inflate.decompress(memoryview(body)[len(GZIP_HEADER) :], limit + 1)
    except zlib.error:
        raise ArchiveError("The gzip stream is corrupt") from None
    if len(tar) > limit:
        raise ArchiveError("The archive exceeds its unpacked limit")
    if not inflate.eof:
        raise ArchiveError("The gzip stream ends early")
    trailer = inflate.unused_data
    if len(trailer) != _TRAILER:
        raise ArchiveError("The gzip stream has no exact trailer")
    crc, size = struct.unpack("<II", trailer)
    if crc != zlib.crc32(tar) or size != len(tar) & 0xFFFFFFFF:
        raise ArchiveError("The gzip trailer does not match its content")
    return tar


def read(
    body: bytes,
    *,
    max_body: int = MAX_BODY_BYTES,
    max_members: int = MAX_MEMBERS,
    max_unpacked: int = MAX_UNPACKED_BYTES,
) -> list[tuple[str, bytes]]:
    """Every member of a submission body as (path, bytes), in archive order;
    ArchiveError at the first broken rule or exceeded limit, and nothing
    returned before the whole archive has passed."""
    if len(body) > max_body:
        raise ArchiveError("The body exceeds 64 MiB")
    # Each member costs at most two blocks beyond its bytes, and the end of
    # the archive at most one record.
    overhead = (max_members * 2 + 2) * _BLOCK + _RECORD
    tar = _decompress(body, max_unpacked + overhead)
    members: list[tuple[str, bytes]] = []
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as archive:
            while (info := archive.next()) is not None:
                if len(members) >= max_members:
                    raise ArchiveError("The archive holds more than 10,000 members")
                total += info.size
                if total > max_unpacked:
                    raise ArchiveError("The archive exceeds 2 GiB unpacked")
                if (
                    info.type != tarfile.REGTYPE
                    or info.pax_headers
                    or (info.mode, info.uid, info.gid, info.mtime) != (0o644, 0, 0, 0)
                    or (info.uname, info.gname) != ("", "")
                ):
                    raise ArchiveError("An archive member is not a plain file")
                name = info.name
                if members:
                    if not _member_name(name):
                        raise ArchiveError("An archive member has an unsafe name")
                    if len(members) > 1 and members[-1][0] >= name:
                        raise ArchiveError(
                            "Archive members are not in sorted, distinct order"
                        )
                elif name != ENVELOPE:
                    raise ArchiveError(
                        "The archive does not start with submission.json"
                    )
                stream = archive.extractfile(info)
                if stream is None:
                    raise ArchiveError("An archive member is not a plain file")
                data = stream.read()
                if len(data) != info.size:
                    raise ArchiveError("An archive member ends early")
                members.append((name, data))
    except tarfile.TarError:
        raise ArchiveError("The archive is not a valid tar archive") from None
    if not members:
        raise ArchiveError("The archive does not start with submission.json")
    # Anything the checks above leave open (padding, header fields, data after
    # the end) differs from the archive this client writes for these members.
    if tar_bytes(members) != tar:
        raise ArchiveError("The archive is not in its written form")
    return members
