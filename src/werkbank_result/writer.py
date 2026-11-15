"""Writing and checking a result directory: `index.html`, the page, and
`result.json` (format `werkbank-result-v1`), which binds the page to the
evaluation it shows, names the renderer and records the passes that were
repeated, which the page discloses. A trial's `result.json` also holds
`"trial": true`: its page says it is a trial and no benchmark and shows no
score, and `werkbank submit` refuses it. A trial is an evaluation of one pass
in one pool. A single pass of a benchmark mode, which `werkbank start` runs on
request, holds `"single_pass": true` instead: its page says so, keeps its
numbers, and `werkbank submit` refuses it too. Any other result has neither
field, so its form is unchanged.

The evaluation is checked first, and only an evaluation this build wrote is
rendered, so the page's numbers come from the same versioned code as the
evaluation's. The result directory is new, never inside the evaluation
directory (which admits only its own files) and never containing it; every
file is rendered before the directory exists and `result.json` is written
last. The same evaluation and repeats always give the same bytes.

A repeat is a fixed-form record, never a path or free text: the pass and its
seed, and why its first attempt failed, a technical result the run records
(`invalid` with a reason of the outcome table) or a fixed kind. On the
command line it is `PASS:SEED:KIND[:STATUS/REASON]` (`repeat_argument`), so
a series finished by hand discloses its repeats as `werkbank start` does.
"""

import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from victors_werkbank.grading.evidence import (
    EvidenceError,
    encode,
    read_bounded,
    sha256,
    strict_record,
    write_exclusive,
)
from victors_werkbank.results.evaluation_check import check_evaluation, this_build
from victors_werkbank.results.outcomes import TERMINAL
from victors_werkbank.results.run import RecordError

from .page import render

RESULT_FORMAT = "werkbank-result-v1"
PAGE = "index.html"
RESULT = "result.json"
MAX_RESULT_BYTES = 16 * 1024 * 1024
# A repeat record's fields, and why a pass's first attempt failed: a
# technical result its run recorded, an interruption (no run record written)
# or no valid run record at all.
REPEATS = "repeats"
TRIAL = "trial"
SINGLE_PASS = "single_pass"
REPEAT_FIELDS = ("pass", "seed", "kind", "status", "reason")
REPEAT_KINDS = ("technical_result", "interrupted", "no_run_record")
MAX_PASSES = 3
# A repeat on the command line: pass and seed as decimal numbers, the kind
# and, for a technical result only, its status and reason.
REPEAT_FORM = "PASS:SEED:KIND[:STATUS/REASON]"
_REPEAT_ARGUMENT = re.compile(
    r"([0-9]{1,10}):([0-9]{1,10}):([a-z_]{1,64})(?::([a-z_]{1,64})/([a-z_]{1,64}))?"
)


class ResultError(ValueError):
    """A refused evaluation, output or result directory; the text names no
    path and no file's contents."""


def _bytecode(path: Path, root: Path) -> bool:
    return "__pycache__" in path.relative_to(root).parts or path.suffix in {
        ".pyc",
        ".pyo",
    }


def renderer_digest() -> str:
    """The digest of this package's files (bytecode caches never), so a
    result names the renderer that wrote its page."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(
        p for p in root.rglob("*") if p.is_file() and not _bytecode(p, root)
    ):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _checked(evaluation: Path) -> dict[str, Any]:
    """The record of a valid evaluation this build wrote."""
    try:
        record, _ = check_evaluation(evaluation)
    except RecordError as error:
        raise ResultError(f"Invalid evaluation: {error}") from None
    if not this_build(record):
        raise ResultError(
            "The evaluation was written by another build; only that build"
            " renders its result page"
        )
    return record


def checked_repeats(repeats: object, record: dict[str, Any]) -> list[dict[str, Any]]:
    """The repeat records, each with exactly `REPEAT_FIELDS` in that order,
    sorted by pass; ResultError unless they belong to the evaluation's one
    pool: a pass from 1 to its passes (at most `MAX_PASSES`), each once, a
    seed of the pool, and a known kind, with an `invalid` status and a reason
    of the outcome table exactly for a technical result."""
    if not isinstance(repeats, (list, tuple)):
        raise ResultError("The repeated passes are not a list")
    if not repeats:
        return []
    pools = record["pools"]
    if len(pools) != 1:
        raise ResultError("Repeated passes belong to an evaluation of one pool")
    (pool,) = pools
    found = []
    for item in repeats:
        if not isinstance(item, dict) or set(item) != set(REPEAT_FIELDS):
            raise ResultError("A repeated pass has other fields than the format's")
        number, seed, kind = item["pass"], item["seed"], item["kind"]
        status, reason = item["status"], item["reason"]
        if (
            type(number) is not int
            or not 1 <= number <= min(MAX_PASSES, pool["repetitions"])
            or type(seed) is not int
            or seed not in pool["seeds"]
        ):
            raise ResultError("A repeated pass is not a pass of the pool")
        if kind == "technical_result":
            known = status == "invalid" and (status, reason) in TERMINAL
        else:
            known = kind in REPEAT_KINDS and status is None and reason is None
        if not known:
            raise ResultError("A repeated pass names an unknown cause")
        found.append({field: item[field] for field in REPEAT_FIELDS})
    if len({item["pass"] for item in found}) != len(found):
        raise ResultError("A pass is repeated twice")
    return sorted(found, key=lambda item: item["pass"])


def checked_trial(trial: object, record: dict[str, Any]) -> bool:
    """Whether the page marks a trial; ResultError unless `trial` is a
    boolean and a trial is an evaluation of one pool with one pass."""
    if type(trial) is not bool:
        raise ResultError("The trial mark is not true or false")
    pools = record["pools"]
    if trial and (len(pools) != 1 or pools[0]["repetitions"] != 1):
        raise ResultError("A trial is an evaluation of one pass in one pool")
    return trial


def checked_single(single_pass: object, trial: bool, record: dict[str, Any]) -> bool:
    """Whether the page marks a single pass; ResultError unless
    `single_pass` is a boolean, a single pass is an evaluation of one pool
    with one pass and the page is not marked as a trial as well."""
    if type(single_pass) is not bool:
        raise ResultError("The single-pass mark is not true or false")
    pools = record["pools"]
    if single_pass and (len(pools) != 1 or pools[0]["repetitions"] != 1):
        raise ResultError("A single pass is an evaluation of one pass in one pool")
    if single_pass and trial:
        raise ResultError("A page is a trial or a single pass, not both")
    return single_pass


def repeat_argument(text: str) -> dict[str, Any]:
    """The repeat record `text` names in the form `REPEAT_FORM`; ResultError,
    naming no part of the text, where it has another form. Whether it belongs
    to the evaluation is `checked_repeats`' question."""
    match = _REPEAT_ARGUMENT.fullmatch(text)
    if match is None:
        raise ResultError(f"A repeated pass is not of the form {REPEAT_FORM}")
    number, seed, kind, status, reason = match.groups()
    return {
        "pass": int(number),
        "seed": int(seed),
        "kind": kind,
        "status": status,
        "reason": reason,
    }


def repeat_text(repeat: dict[str, Any]) -> str:
    """A repeat record in the form `repeat_argument` reads."""
    text = f"{repeat['pass']}:{repeat['seed']}:{repeat['kind']}"
    if repeat["status"] is not None:
        text += f":{repeat['status']}/{repeat['reason']}"
    return text


def _rendered(
    record: dict[str, Any],
    repeats: list[dict[str, Any]],
    trial: bool = False,
    single_pass: bool = False,
) -> dict[str, bytes]:
    """Both files of a result directory, `result.json` last."""
    try:
        page = render(record, repeats, trial=trial, single_pass=single_pass).encode(
            "utf-8"
        )
    except RecordError as error:
        raise ResultError(f"Invalid evaluation: {error}") from None
    result: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "evaluation": {
            "format": record["format"],
            # This build wrote the evaluation, so its written form is exactly
            # the checked file's bytes.
            "sha256": sha256(encode(record)),
        },
        "renderer_sha256": renderer_digest(),
        REPEATS: repeats,
        "files": {PAGE: sha256(page)},
    }
    if trial:
        result[TRIAL] = True
    if single_pass:
        result[SINGLE_PASS] = True
    return {PAGE: page, RESULT: encode(result)}


def write(
    evaluation: Path,
    output: Path,
    *,
    repeats: Sequence[dict[str, Any]] = (),
    trial: bool = False,
    single_pass: bool = False,
) -> dict[str, Any]:
    """Check `evaluation` and write its result page into the new directory
    `output`, disclosing the `repeats` (fixed-form records, `checked_repeats`)
    and marking a `trial` (`checked_trial`) or a `single_pass`
    (`checked_single`); returns the result record."""
    record = _checked(evaluation)
    found = checked_repeats(repeats, record)
    marked = checked_trial(trial, record)
    single = checked_single(single_pass, marked, record)
    if output.exists() or output.is_symlink():
        raise ResultError("Output must be a new directory; nothing is overwritten")
    try:
        target, source = output.resolve(), evaluation.resolve()
    except (OSError, RuntimeError):
        raise ResultError("Evaluation or output path cannot be resolved") from None
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ResultError("Output and evaluation must not contain each other")
    files = _rendered(record, found, marked, single)
    output.mkdir(mode=0o700, parents=True)
    for name, data in files.items():
        write_exclusive(output / name, data)
    result: dict[str, Any] = json.loads(files[RESULT])
    return result


def _read(path: Path) -> bytes:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ResultError("The result directory holds a link or device")
        return read_bounded(path, MAX_RESULT_BYTES)
    except (EvidenceError, OSError):
        raise ResultError("A result file is missing or linked") from None


def check(evaluation: Path, result: Path) -> dict[str, Any]:
    """Whether `result` is exactly the result directory this package renders
    for `evaluation`, the repeats and the trial or single-pass mark its
    `result.json` records
    (checked as `write` checks them): its two files and nothing else, byte
    for byte.
    Returns the result record; ResultError otherwise."""
    record = _checked(evaluation)
    if result.is_symlink() or not result.is_dir():
        raise ResultError("A result is a directory")
    if sorted(os.listdir(result)) != sorted([PAGE, RESULT]):
        raise ResultError("The result directory differs from its file list")
    found = {name: _read(result / name) for name in (PAGE, RESULT)}
    try:
        recorded = strict_record(found[RESULT])
    except EvidenceError:
        raise ResultError("A result file differs from this build's rendering") from None
    if not isinstance(recorded, dict) or REPEATS not in recorded:
        raise ResultError("A result file differs from this build's rendering")
    trial = checked_trial(recorded.get(TRIAL, False), record)
    expected = _rendered(
        record,
        checked_repeats(recorded[REPEATS], record),
        trial,
        checked_single(recorded.get(SINGLE_PASS, False), trial, record),
    )
    for name, data in expected.items():
        if found[name] != data:
            raise ResultError("A result file differs from this build's rendering")
    checked: dict[str, Any] = json.loads(expected[RESULT])
    return checked
