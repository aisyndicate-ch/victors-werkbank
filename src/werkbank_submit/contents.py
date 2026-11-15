"""What goes into a submission, and the checks before it is built: the
evaluation as this build validates it, one case list across its pools, the
result page beside it and, with the evaluated runs, their exports.

Every member's bytes are read once, bounded and without following a link, and
the digests the evaluation records are checked again on exactly those bytes,
so a file that changed after its check is never sent under its old digest.
"""

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from victors_werkbank.grading.evidence import (
    EvidenceError,
    canonical,
    encode,
    read_bounded,
    sha256,
    strict_record,
)
from victors_werkbank.registry import case_set_name
from victors_werkbank.results.evaluation import EVALUATION
from victors_werkbank.results.evaluation_check import (
    MAX_EVALUATION_BYTES,
    MAX_FILE_BYTES,
    check_evaluation,
    this_build,
)
from victors_werkbank.results.export import ARCHIVE, INDEX, export
from victors_werkbank.results.run import PLAN, RUN, RecordError
from werkbank_result import ResultError, check
from werkbank_result.writer import PAGE, RESULT, SINGLE_PASS, TRIAL
from werkbank_start.flow import TRIAL_CASES

from .archive import MAX_MEMBERS, MAX_UNPACKED_BYTES, safe_name

OTHER_BUILD = (
    "This evaluation was not written by this build of Werkbank: write a new"
    " evaluation with this build (werkbank compare) and submit that one."
)
NOT_COMPLETE = (
    "The evaluation could not be checked completely against its runs; give"
    " exactly the runs it was written from (--source), in any order."
)
SOURCE_UNKNOWN = (
    "A --source is not one of the runs this evaluation was written from; give"
    " exactly those runs (--source), in any order."
)
SOURCE_TWICE = (
    "A run is given twice with --source; give each run the evaluation was"
    " written from once."
)
SOURCE_MISSING = (
    "A run this evaluation was written from is missing from --source; give"
    " each of its runs once."
)
DIFFERENT_CASES = (
    "The pools of this evaluation were evaluated on different cases; a"
    " submission covers one case list (write the evaluation with compare --set)."
)
TRIAL_REFUSED = (
    "This is a trial of werkbank start, not a benchmark: a trial is never"
    " submitted. Start Quick, Normal or Full for results to send."
)
SINGLE_PASS_REFUSED = (
    "This is a single pass of werkbank start: results to send hold three"
    " passes. Start Quick, Normal or Full with three passes for results to send."
)
CHANGED = "A file changed while it was read; run the submission again."
PAGE_PLACE = (
    "The result page must be a directory of its own beside the evaluation"
    " directory, neither inside it nor around it."
)
STALE_PAGE = (
    "The result page is not the one this build writes for this evaluation;"
    " write it again with werkbank result"
)
PAGE_FILES = (
    "The result page directory may hold only regular files with short plain"
    " names (letters, digits, '.', '_' and '-'), no links, and at least one"
    " file."
)
RAW_CHANGED = (
    "A run's export does not match the digests the evaluation records for it;"
    " run the submission again with the unchanged runs."
)


class Refusal(Exception):
    """A check before the confirmation failed; the message says what to fix."""


class RawTooLarge(Exception):
    """The raw data alone would exceed the archive limits."""


@dataclass(frozen=True)
class Evaluation:
    record: dict[str, Any]
    sha256: str
    case_set: dict[str, Any]
    members: list[tuple[str, bytes]]


def _bounded(path: Path, maximum: int) -> bytes:
    try:
        return read_bounded(path, maximum)
    except EvidenceError:
        raise Refusal(CHANGED) from None


def evaluation(root: Path, sources: Sequence[Path]) -> Evaluation:
    """The evaluation at `root` as this build validates it, with `sources`
    when given (then completely); its members under `evaluation/`."""
    try:
        record, complete = check_evaluation(root, list(sources) or None)
    except RecordError as error:
        raise Refusal(f"The evaluation is not valid: {error}") from None
    except (ValueError, OSError) as error:
        raise Refusal(f"The evaluation is not valid: {type(error).__name__}") from None
    if not this_build(record):
        raise Refusal(OTHER_BUILD)
    if sources and not complete:
        raise Refusal(NOT_COMPLETE)
    lists = [pool["case_ids"] for pool in record["pools"]]
    if any(cases != lists[0] for cases in lists):
        raise Refusal(DIFFERENT_CASES)
    named = record["parameters"]["case_set"]
    if named is None:
        case_set = {"name": case_set_name(lists[0]), "case_ids": list(lists[0])}
    elif set(named["case_ids"]) != set(lists[0]):
        raise Refusal(DIFFERENT_CASES)
    else:
        case_set = {"name": named["name"], "case_ids": list(named["case_ids"])}
    # An evaluation of exactly the trial's cases is a trial, whichever page
    # was written for it.
    if set(lists[0]) == set(TRIAL_CASES):
        raise Refusal(TRIAL_REFUSED)
    data = _bounded(root / EVALUATION, MAX_EVALUATION_BYTES)
    try:
        same = canonical(strict_record(data)) == canonical(record)
    except EvidenceError:
        same = False
    if not same:
        raise Refusal(CHANGED)
    members = [(f"evaluation/{EVALUATION}", data)]
    for name, digest in record["files"].items():
        if not safe_name(f"evaluation/{name}"):
            raise Refusal(CHANGED)
        content = _bounded(root / name, MAX_FILE_BYTES)
        if sha256(content) != digest:
            raise Refusal(CHANGED)
        members.append((f"evaluation/{name}", content))
    return Evaluation(record, sha256(data), case_set, sorted(members))


def beside(page: Path, evaluation_root: Path) -> None:
    """Refusal unless `page` is a directory (no link) that neither lies in
    the evaluation directory nor holds it."""
    if page.is_symlink() or not page.is_dir():
        raise Refusal(PAGE_PLACE)
    try:
        real, other = page.resolve(), evaluation_root.resolve()
    except (OSError, RuntimeError):
        raise Refusal(PAGE_PLACE) from None
    if real.is_relative_to(other) or other.is_relative_to(real):
        raise Refusal(PAGE_PLACE)


def result_page(
    page: Path, evaluation_root: Path, evaluation_sha256: str
) -> list[tuple[str, bytes]]:
    """The result page directory's files as members under `result/`, once
    the page is exactly what this build renders for the evaluation whose
    `evaluation.json` has `evaluation_sha256`, checked again on the bytes
    read here."""
    beside(page, evaluation_root)
    try:
        expected = check(evaluation_root, page)
    except (ResultError, OSError) as error:
        detail = error if isinstance(error, ResultError) else type(error).__name__
        raise Refusal(f"{STALE_PAGE} ({detail}).") from None
    if expected.get(TRIAL):
        raise Refusal(TRIAL_REFUSED)
    if expected.get(SINGLE_PASS):
        raise Refusal(SINGLE_PASS_REFUSED)
    members: list[tuple[str, bytes]] = []

    def refuse(_: OSError) -> None:
        raise Refusal(PAGE_FILES)

    for directory, names, files in os.walk(page, onerror=refuse):
        for name in [*names, *files]:
            path = Path(directory) / name
            relative = path.relative_to(page).as_posix()
            info = path.lstat()
            if not safe_name(f"result/{relative}"):
                raise Refusal(PAGE_FILES)
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or len(members) >= MAX_MEMBERS:
                raise Refusal(PAGE_FILES)
            members.append((f"result/{relative}", _bounded(path, MAX_FILE_BYTES)))
    found = dict(members)
    if (
        set(found) != {f"result/{PAGE}", f"result/{RESULT}"}
        or found[f"result/{RESULT}"] != encode(expected)
        or sha256(found[f"result/{PAGE}"]) != expected["files"][PAGE]
        or expected["evaluation"]["sha256"] != evaluation_sha256
    ):
        raise Refusal(f"{STALE_PAGE}.")
    return sorted(members)


def _run_id(source: Path) -> str | None:
    """The run ID the plan of the run at `source` records, None where it has
    no readable one."""
    try:
        plan = strict_record(read_bounded(source / PLAN, MAX_FILE_BYTES))
    except EvidenceError:
        return None
    run_id = plan.get("run_id") if isinstance(plan, dict) else None
    return run_id if isinstance(run_id, str) else None


def matched(
    record: dict[str, Any], sources: Sequence[Path]
) -> list[tuple[Path, dict[str, Any]]]:
    """Each of the evaluation's inputs, in its order, with the source whose
    plan records its run ID, whatever order the sources are given in;
    Refusal for a source of no input, a run given twice or a missing one."""
    inputs = {item["run_id"] for item in record["inputs"]}
    found: dict[str, Path] = {}
    for source in sources:
        run_id = _run_id(source)
        if run_id is None or run_id not in inputs:
            raise Refusal(SOURCE_UNKNOWN)
        if run_id in found:
            raise Refusal(SOURCE_TWICE)
        found[run_id] = source
    if set(found) != inputs:
        raise Refusal(SOURCE_MISSING)
    return [(found[item["run_id"]], item) for item in record["inputs"]]


def raw_data(
    record: dict[str, Any], sources: Sequence[Path], scratch: Path
) -> list[tuple[str, bytes]]:
    """Each evaluated run's export, written under `scratch`, as the members
    `raw/<run_id>/export.json` and `raw/<run_id>/run.tar`, after matching
    each source to its input by run ID (`matched`) and checking that it
    exports exactly the run the evaluation records. RawTooLarge once the
    exports exceed the unpacked archive limit."""
    members: list[tuple[str, bytes]] = []
    total = 0
    for source, item in matched(record, sources):
        run_id = item["run_id"]
        output = scratch / run_id
        try:
            export(source, output)
        except (RecordError, ValueError, OSError):
            raise Refusal(RAW_CHANGED) from None
        try:
            total += (output / INDEX).stat().st_size + (output / ARCHIVE).stat().st_size
        except OSError:
            raise Refusal(RAW_CHANGED) from None
        if total > MAX_UNPACKED_BYTES:
            raise RawTooLarge()
        index = _bounded(output / INDEX, MAX_FILE_BYTES)
        tar = _bounded(output / ARCHIVE, MAX_UNPACKED_BYTES)
        try:
            parsed = strict_record(index)
            files = {entry["path"]: entry["sha256"] for entry in parsed["files"]}
            matches = (
                parsed["source_run_id"] == run_id
                and files.get(PLAN) == item["plan_sha256"]
                and files.get(RUN) == item["run_sha256"]
                and parsed["archive_sha256"] == sha256(tar)
            )
        except (EvidenceError, KeyError, TypeError):
            matches = False
        if not matches:
            raise Refusal(RAW_CHANGED)
        members += [(f"raw/{run_id}/{INDEX}", index), (f"raw/{run_id}/{ARCHIVE}", tar)]
    return sorted(members)
