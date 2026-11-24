"""The steps of a submission, in order: the questions, the checks, the
summary, the consent text and the prompts, with raw data the leaderboard's
consent text, its question and the display name; then, only after an
explicit `y` at a terminal, one send or one written file.

Exit codes: 0 a confirmed receipt or a written file; 2 refused before the
confirmation, something needs fixing; 3 nothing confirmed (declined, end of
input, interrupted, or no confirmed receipt from the receiver).
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from victors_werkbank import __version__
from victors_werkbank.agent.episode import harness_digest
from victors_werkbank.grading.evidence import (
    EvidenceError,
    read_bounded,
    sha256,
    write_exclusive,
)
from victors_werkbank.registry import case_sets, scored_cases, scored_sets
from victors_werkbank.results.comparison import SOURCE_TEXTS
from victors_werkbank.results.evaluation import entry_labels
from victors_werkbank.results.report import case_set_text

from . import archive, contents, identity, leaderboard, privacy, receiver, send, texts

CONFIRMED = 0
REFUSED = 2
NOT_CONFIRMED = 3

NOT_A_TERMINAL = (
    "werkbank submit asks for your confirmation at a terminal; run it in an"
    " interactive terminal, without redirected input or output. Nothing was sent."
)
FILE_EXISTS = "The --write file must be new, in an existing directory."
OTHER_FILE = (
    "{path} already exists with other content; move it away or give --write FILE."
)
ALREADY_WRITTEN = "Already written: {path} holds exactly this archive."
NAMED_TOO_LARGE = (
    "The archive with raw data and the display name breaks the archive limits."
)
SHOWN_FINDINGS = 20

# The archive with raw data published under a display name (None for
# "anonymous"): (its body, or None beyond the archive limits, and the privacy
# check's findings in its envelope, which hold the name).
Published = Callable[[str | None], tuple[bytes | None, list[privacy.Finding]]]


class Console:
    """Where a submission's dialog happens: prompts and answers on `input`
    and `output`, the real standard streams unless given. Only a place where
    a person answers confirms: here, both streams must be a terminal."""

    def __init__(self, input: TextIO | None = None, output: TextIO | None = None):
        self.input = sys.stdin if input is None else input
        self.output = sys.stdout if output is None else output

    def is_terminal(self) -> bool:
        return self.input.isatty() and self.output.isatty()

    def ask(self, prompt: str) -> str:
        """One answer line without its line end; EOFError at the end of
        input."""
        self.output.write(prompt)
        self.output.flush()
        line = self.input.readline()
        if not line:
            raise EOFError
        return line[:-1] if line.endswith("\n") else line

    def say(self, text: str) -> None:
        self.output.write(text + "\n")
        self.output.flush()

    def choose(self, question: str, labels: Sequence[str]) -> int | None:
        """The index of the label picked by number, None for a skipped
        question (Enter); EOFError at the end of input."""
        self.say(question)
        for number, label in enumerate(labels, 1):
            self.say(f"  {number}) {label}")
        numbers = {str(number): number - 1 for number in range(1, len(labels) + 1)}
        while True:
            reply = self.ask(texts.CHOICE.format(count=len(labels))).strip()
            if reply == "":
                return None
            if reply in numbers:
                return numbers[reply]
            self.say(texts.CHOICE_AGAIN.format(count=len(labels)))

    def confirm(self, prompt: str) -> bool:
        """Only an explicit `y` confirms; EOFError at the end of input."""
        return self.ask(prompt) == "y"


def default_file(evaluation: Path, body: bytes) -> Path:
    """Where an archive goes without a receiver and without --write: beside
    the evaluation directory, named by the archive's digest."""
    return evaluation.resolve().parent / f"submission-{sha256(body)[:12]}.tar.gz"


def _shown(text: str) -> str:
    """`text` for a terminal: characters that are not printable escaped."""
    return "".join(
        char if char.isprintable() else char.encode("unicode_escape").decode("ascii")
        for char in text
    )


def _findings(found: list[privacy.Finding]) -> list[str]:
    lines = [f"  {finding.text()}" for finding in found[:SHOWN_FINDINGS]]
    if len(found) > SHOWN_FINDINGS:
        lines.append(f"  and {len(found) - SHOWN_FINDINGS} more")
    return lines


def _questions(console: Console) -> dict[str, str | None]:
    answers: dict[str, str | None] = {}
    for field, question, choices in texts.QUESTIONS:
        index = console.choose(question, [label for _, label in choices])
        answers[field] = None if index is None else choices[index][0]
    return answers


def leaderboard_set() -> str | None:
    """The case set the leaderboard takes: the scored set that holds every
    scored case, the Full set; None where no scored set does."""
    every = sorted(scored_cases())
    sets = case_sets()
    for name in scored_sets():
        if sorted(sets[name]) == every:
            return name
    return None


def ranked(case_set: dict[str, Any]) -> bool:
    """Whether results on `case_set` can enter the leaderboard: only those
    on the Full set (`leaderboard_set`), never another scored set, a
    module's, a trial's or an own selection."""
    name = case_set["name"]
    return name is not None and name == leaderboard_set()


def _display_name(console: Console, published: Published) -> tuple[bytes, str | None]:
    """The body published under the display name the tester types, asked
    again until the name has the accepted form and the privacy check finds
    nothing in it; Enter gives "anonymous". EOFError at the end of input,
    contents.Refusal when the name makes the archive break its limits."""
    while True:
        try:
            name = leaderboard.display_name(console.ask(texts.DISPLAY_NAME))
        except leaderboard.NameRefused:
            console.say(texts.NAME_AGAIN)
            continue
        body, found = published(name)
        if found:
            kinds = sorted({finding.kind for finding in found})
            console.say(texts.NAME_PRIVATE.format(kind=", ".join(kinds)))
            continue
        if body is None:
            raise contents.Refusal(NAMED_TOO_LARGE)
        return body, name


def _pool_lines(label: str, pool: dict[str, Any]) -> list[str]:
    """One entry of the evaluation in the summary, under its name: a pool,
    or runs with a given instance value, which are no pool."""
    serving = [
        f"{name} {_shown(str(value))}"
        for name, value in pool["serving"].items()
        if value is not None
    ]
    configuration = [
        f"{name} {_shown(json.dumps(value, sort_keys=True))}"
        for name, value in pool["configuration"].items()
    ]
    hardware = pool["hardware"]
    return [
        label,
        f"  Served model: {_shown(pool['model_id'])}",
        f"  Serving settings: {', '.join(serving) or 'none recorded'}",
        f"  Run configuration: {', '.join(configuration)}",
        "  Hardware: none recorded"
        if hardware is None
        else f"  Hardware: {_shown(hardware['text'])}"
        f" (from the {SOURCE_TEXTS[hardware['source']]})",
        f"  Container image: {pool['image']}",
    ]


def _summary(
    *,
    address: str | None,
    target: str,
    contact: str | None,
    found: contents.Evaluation,
    always: list[tuple[str, bytes]],
    raw: list[tuple[str, bytes]],
    without: bytes,
    with_raw: bytes | None,
    dropped: str | None,
    installation_id: str,
    stored: bool,
) -> list[str]:
    record = found.record
    pools = record["pools"]
    case_set = found.case_set
    lines = [
        "",
        "Summary of the submission",
        f"Receiver: {_shown(address)}" if address else f"Receiver: {texts.NO_RECEIVER}",
        *([f"File: {target}"] if target else []),
        f"Deletion contact: {_shown(contact)}"
        if contact
        else f"Deletion contact: {texts.NO_CONTACT}",
    ]
    for label, pool in zip(entry_labels(record), pools, strict=True):
        lines += _pool_lines(label, pool)
    count = len(case_set["case_ids"])
    # As the result page names it; a named set with its size.
    shown = case_set_text(case_set["name"], case_set["case_ids"])
    if case_set["name"] is not None:
        shown += f", {count} case{'s' if count != 1 else ''}"
    lines += [
        f"Case set: {shown}",
        "Runs: " + ", ".join(item["run_id"] for item in record["inputs"]),
        f"Werkbank: version {__version__}, build {harness_digest()[:12]}",
        "Files always sent:",
        *(f"  {name}  {len(data)} bytes" for name, data in always),
    ]
    if raw:
        lines += [
            "Raw data files:",
            *(f"  {name}  {len(data)} bytes" for name, data in raw),
        ]
    lines.append(f"Compressed size without raw data: {len(without)} bytes")
    if with_raw is not None:
        lines.append(f"Compressed size with raw data: {len(with_raw)} bytes")
    else:
        lines.append(f"Raw data cannot be included: {dropped}")
    lines.append(
        f"Installation ID: {installation_id}"
        + (
            " (stored on this computer)"
            if stored
            else " (new; stored on this computer once you confirm)"
        )
    )
    return lines


def submit(
    evaluation: Path,
    result: Path,
    sources: Sequence[Path] = (),
    *,
    endpoint: str | None = None,
    write: Path | None = None,
    address: str | None = receiver.ADDRESS,
    contact: str | None = receiver.DELETION_CONTACT,
    console: Console | None = None,
    timeout: float = send.TIMEOUT,
    values: Sequence[tuple[str, str]] | None = None,
) -> int:
    """Submit the evaluation at `evaluation` with its result page `result`
    and, for raw data, its runs `sources`; returns the exit code.

    The dialog runs on `console` (the real standard streams by default) and
    only where it is a terminal. `endpoint` is the endpoint the runs used,
    when the caller knows it: the privacy check looks for it, nothing stores
    it. With `write`, or without an `address`, the archive is written to a
    new local file (`write`, or by default beside the evaluation directory)
    instead of being sent. `values` replaces this computer's local values in
    the privacy check."""
    console = Console() if console is None else console
    if not console.is_terminal():
        print(NOT_A_TERMINAL, file=sys.stderr)
        return REFUSED
    try:
        return _submit(
            evaluation,
            result,
            list(sources),
            endpoint=endpoint,
            write=write,
            address=address,
            contact=contact,
            console=console,
            timeout=timeout,
            values=values,
        )
    except KeyboardInterrupt:
        console.say(f"\n{texts.NOTHING_SENT}")
        return NOT_CONFIRMED


def _existing(path: Path) -> bytes | None:
    """The bytes of an archive file already at `path`, None where there is
    none; EvidenceError for a link or anything but a bounded regular file."""
    if not os.path.lexists(path):
        return None
    return read_bounded(path, archive.MAX_BODY_BYTES)


def _submit(
    evaluation: Path,
    result: Path,
    sources: list[Path],
    *,
    endpoint: str | None,
    write: Path | None,
    address: str | None,
    contact: str | None,
    console: Console,
    timeout: float,
    values: Sequence[tuple[str, str]] | None,
) -> int:
    say = console.say
    try:
        if address is not None:
            receiver.check_address(address)
        if write is not None and (os.path.lexists(write) or not write.parent.is_dir()):
            raise contents.Refusal(FILE_EXISTS)
        if evaluation.is_symlink() or not evaluation.is_dir():
            raise contents.Refusal("The evaluation must be a directory.")
        if any(source.is_symlink() or not source.is_dir() for source in sources):
            raise contents.Refusal("Each --source must be a run directory.")
        contents.beside(result, evaluation)
        id_path = identity.config_path()
        stored = identity.load(id_path)
    except (
        contents.Refusal,
        receiver.AddressError,
        identity.InstallationError,
        RuntimeError,
    ) as error:
        say(str(error))
        say(texts.NOTHING_SENT)
        return REFUSED
    to_file = write is not None or address is None
    if write is not None:
        target = str(write)
    elif address is None:
        target = str(default_file(evaluation, b"").parent / texts.DEFAULT_FILE)
    else:
        target = ""
    installation_id = stored or identity.new_id()
    # What can never be submitted is refused before anything is asked: an
    # evaluation that is not valid or not this build's, a trial, a single pass
    # or a result page of another evaluation.
    try:
        checked = contents.evaluation(evaluation, sources)
        contents.result_page(result, evaluation, checked.sha256)
    except contents.Refusal as error:
        say(str(error))
        say(texts.NOTHING_SENT)
        return REFUSED
    except OSError:
        say(contents.CHANGED)
        say(texts.NOTHING_SENT)
        return REFUSED
    try:
        answers = _questions(console)
    except EOFError:
        say(f"\n{texts.NOTHING_SENT}")
        return NOT_CONFIRMED
    try:
        with tempfile.TemporaryDirectory(prefix="werkbank-submit-") as scratch:
            prepared = _prepare(
                evaluation,
                result,
                sources,
                Path(scratch),
                installation_id=installation_id,
                answers=answers,
                endpoint=endpoint,
                values=values,
            )
    except contents.Refusal as error:
        say(str(error))
        say(texts.NOTHING_SENT)
        return REFUSED
    except OSError:
        say(contents.CHANGED)
        say(texts.NOTHING_SENT)
        return REFUSED
    found, always, raw, without, with_raw, dropped, published = prepared
    for line in _summary(
        address=address,
        target=target,
        contact=contact,
        found=found,
        always=always,
        raw=raw if with_raw is not None else [],
        without=without,
        with_raw=with_raw,
        dropped=dropped,
        installation_id=installation_id,
        stored=stored is not None,
    ):
        say(line)
    say("")
    say(texts.CONSENT_TEXT)
    nothing = texts.NOTHING_WRITTEN if to_file else texts.NOTHING_SENT
    try:
        include = with_raw is not None and console.confirm(texts.INCLUDE_RAW)
        body = with_raw if include and with_raw is not None else without
        if include and not ranked(found.case_set):
            say(texts.NOT_RANKED)
        elif include:
            say("")
            say(texts.LEADERBOARD_CONSENT_TEXT)
            if console.confirm(texts.PUBLISH):
                body, name = _display_name(console, published)
                say("Leaderboard: publish yes")
                say(f"Display name: {texts.ANONYMOUS if name is None else name}")
            else:
                say("Leaderboard: publish no")
        else:
            say(texts.NO_RAW_NOT_PUBLISHED)
        say(f"Archive: {len(body)} bytes, SHA-256 {sha256(body)}")
        path = write if write is not None else default_file(evaluation, body)
        if to_file:
            say(f"File: {path}")
        if write is None and address is None:
            try:
                there = _existing(path)
            except EvidenceError:
                there = b""
            if there == body:
                say(ALREADY_WRITTEN.format(path=path))
                return CONFIRMED
            if there is not None:
                say(OTHER_FILE.format(path=path))
                say(texts.NOTHING_WRITTEN)
                return REFUSED
        if not console.confirm(texts.WRITE_FILE if to_file else texts.SEND_NOW):
            say(nothing)
            return NOT_CONFIRMED
    except EOFError:
        say(f"\n{nothing}")
        return NOT_CONFIRMED
    except contents.Refusal as error:
        say(str(error))
        say(nothing)
        return REFUSED
    if stored is None:
        try:
            identity.store(id_path, installation_id)
        except OSError:
            say(f"The installation ID could not be stored as the new file {id_path}.")
            say(nothing)
            return NOT_CONFIRMED
    if to_file:
        try:
            write_exclusive(path, body)
        except OSError:
            say("The file could not be written as a new file.")
            say(texts.NOTHING_WRITTEN)
            return NOT_CONFIRMED
        say(f"Written: {path} ({len(body)} bytes).")
        return CONFIRMED
    assert address is not None
    try:
        outcome = send.send(
            body,
            installation_id=installation_id,
            address=address,
            raw_data=include,
            timeout=timeout,
        )
    except KeyboardInterrupt:
        say(texts.NO_RECEIPT)
        return NOT_CONFIRMED
    say(outcome.message)
    return CONFIRMED if outcome.confirmed else NOT_CONFIRMED


def _prepare(
    evaluation: Path,
    result: Path,
    sources: list[Path],
    scratch: Path,
    *,
    installation_id: str,
    answers: dict[str, str | None],
    endpoint: str | None,
    values: Sequence[tuple[str, str]] | None,
) -> tuple[
    contents.Evaluation,
    list[tuple[str, bytes]],
    list[tuple[str, bytes]],
    bytes,
    bytes | None,
    str | None,
    Published,
]:
    """The checks of a submission and both archives: (the evaluation, the
    always-sent members, the raw members, the body without raw data, the
    body with it or None, why raw data cannot be included, and how the body
    with raw data is built for publication under a display name)."""
    found = contents.evaluation(evaluation, sources)
    page = contents.result_page(result, evaluation, found.sha256)
    always = sorted(found.members + page)
    raw: list[tuple[str, bytes]] = []
    dropped: str | None = None
    if not sources:
        dropped = "the evaluated runs were not given (--source)"
    else:
        try:
            raw = contents.raw_data(found.record, sources, scratch)
        except contents.RawTooLarge:
            dropped = "the raw data would exceed the archive limits"

    def members(
        raw_members: list[tuple[str, bytes]],
        publish: bool = False,
        display_name: str | None = None,
    ) -> list[tuple[str, bytes]]:
        listed = sorted(always + raw_members)
        head = archive.envelope(
            installation_id=installation_id,
            evaluation=found.record,
            evaluation_sha256=found.sha256,
            case_set=found.case_set,
            answers=answers,
            raw_data=bool(raw_members),
            members=listed,
            offered=bool(raw_members) and ranked(found.case_set),
            publish=publish,
            display_name=display_name,
        )
        return [(archive.ENVELOPE, head), *listed]

    plain = members([])
    try:
        tar = archive.tar_bytes(plain)
    except archive.ArchiveError as error:
        raise contents.Refusal(f"The submission breaks the archive limits: {error}.")
    local = privacy.usable(
        privacy.local_values([evaluation, result, *sources], endpoint)
        if values is None
        else values,
        privacy.vocabulary(),
    )
    always_found, _ = privacy.check(plain, local)
    if always_found:
        raise contents.Refusal(
            "\n".join(
                [
                    "The privacy check found something identifying in what is"
                    " always sent; fix its source and write the evaluation"
                    " and result page again:",
                    *_findings(always_found),
                ]
            )
        )
    full = members(raw) if raw else None
    raw_found: list[privacy.Finding] = []
    if full is not None:
        # The always-sent members are checked above; with raw data only the
        # envelope differs, and a finding there drops the raw data too.
        envelope_found, raw_found = privacy.check([full[0], *raw], local)
        raw_found = envelope_found + raw_found
    without = archive.compress(tar)
    if len(without) > archive.MAX_BODY_BYTES:
        raise contents.Refusal("The submission is larger than 64 MiB.")
    with_raw: bytes | None = None
    if full is not None:
        if raw_found:
            dropped = "\n".join(
                [
                    "the privacy check found something identifying in it:",
                    *_findings(raw_found),
                ]
            )
        else:
            try:
                with_raw = archive.build(full)
            except archive.ArchiveError:
                dropped = "the raw data would exceed the archive limits"
            else:
                if len(with_raw) > archive.MAX_BODY_BYTES:
                    with_raw = None
                    dropped = "the archive with raw data would be larger than 64 MiB"

    def published(
        display_name: str | None,
    ) -> tuple[bytes | None, list[privacy.Finding]]:
        # The raw members passed the check above; only the envelope, which
        # now holds the name, is checked again.
        named = members(raw, True, display_name)
        envelope_found, _ = privacy.check(named[:1], local)
        if envelope_found:
            return None, envelope_found
        try:
            body = archive.build(named)
        except archive.ArchiveError:
            return None, []
        return (body if len(body) <= archive.MAX_BODY_BYTES else None), []

    return found, always, raw, without, with_raw, dropped, published
