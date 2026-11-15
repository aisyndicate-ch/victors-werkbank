"""The viewer's pages: the list of directories, a run, an evaluation and its
documents and tables.

Everything shown comes through the package's own reading: a run this build
validates is rendered from its validated records exactly as `werkbank report`
renders it; an evaluation is checked as `werkbank validate` checks it. The
report a run of another build wrote is shown as written and labelled
unchecked. Journals, reasoning, raw responses, snapshots and Harbor's
`trials/` are never read for display. Every value is escaped."""

import csv
import hashlib
import html
import io
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from victors_werkbank.agent.episode import harness_digest
from victors_werkbank.grading.evidence import EvidenceError, read_bounded, strict_record
from victors_werkbank.results.evaluation_check import (
    MAX_FILE_BYTES,
    check_evaluation,
    this_build,
)
from victors_werkbank.results.report import render
from victors_werkbank.results.run import (
    MAX_PLAN_BYTES,
    PLAN,
    PLAN_FORMAT,
    RUN,
    RecordError,
    RunView,
    read_run,
)
from werkbank_result.writer import MAX_RESULT_BYTES, PAGE, ResultError, check

from . import catalog
from .markdown import to_html

REPORT = "report.md"
MAX_REPORT_BYTES = 4 * 1024 * 1024
DOCUMENTS = {"model-card": "model-card.md", "dataset-card": "dataset/README.md"}
TABLES = "dataset/data/"

CHANGED = (
    '<p class="note">This file no longer has the digest its evaluation record'
    " binds; nothing of it is shown.</p>"
)
RESULT_DIRECTORY = "result"
RESULT_DARK_RULE = "@media (prefers-color-scheme: dark)"
COMPLETE = "complete"
INTERRUPTED = "interrupted"
OTHER_BUILD = "another build"
INVALID = "invalid"


READ_ONLY_HEADER = '<a class="brand" href="/">werkbank</a> · read-only view'
CONTROL_HEADER = (
    '<a class="brand" href="/">werkbank</a>'
    '<nav><a href="/series">Series</a><a href="/">Runs and evaluations</a></nav>'
    '<span class="where">local interface · 127.0.0.1 only</span>'
)


def page(
    title: str, body: str, *, control: bool = False, refresh: int | None = None
) -> str:
    """A page of the interface; with `refresh`, one that stays current: the
    interface's script updates it in place every that many seconds, and
    without scripts it loads itself again instead."""
    again = (
        ""
        if refresh is None
        else f'<noscript><meta http-equiv="refresh" content="{refresh}"></noscript>'
        '<script src="/static/live.js" defer></script>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta name="color-scheme" content="dark">{again}'
        f"<title>{html.escape(title)} · werkbank</title>"
        '<link rel="stylesheet" href="/static/style.css"></head><body>'
        f"<header>{CONTROL_HEADER if control else READ_ONLY_HEADER}</header>"
        f"<main>{body}</main></body></html>\n"
    )


def dark(document: str) -> str:
    """A result page with its dark scheme applied whatever the browser
    prefers, as the interface is dark throughout. The page follows the
    browser's preference in two media rules; here they always apply. The
    file itself is never changed."""
    return (
        document.replace(RESULT_DARK_RULE, "@media all")
        .replace("color-scheme: light dark", "color-scheme: dark")
        .replace('content="light dark"', 'content="dark"')
    )


def checked_result(evaluation: Path, result: Path) -> str | None:
    """The page of the result directory `result`, dark, where it is exactly
    this build's rendering for `evaluation`; None otherwise."""
    try:
        check(evaluation, result)
        data = read_bounded(result / PAGE, MAX_RESULT_BYTES)
    except (ResultError, RecordError, EvidenceError, OSError, ValueError):
        return None
    return dark(data.decode("utf-8", errors="replace"))


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def badge(status: str) -> str:
    slug = status.replace(" ", "-")
    return f'<span class="badge {slug}">{html.escape(status)}</span>'


def _text(value: Any) -> str:
    return html.escape(str(value))


def _short(value: Any) -> str:
    return value[:12] if isinstance(value, str) else "unknown"


def _cases(plan: dict[str, Any]) -> str:
    """A plan's case list as written, or empty when it is not a list of text."""
    cases = plan.get("case_ids")
    if isinstance(cases, list) and all(isinstance(case, str) for case in cases):
        return " ".join(cases)
    return ""


def _plan(path: Path) -> dict[str, Any] | None:
    """The plan record as written, unchecked; None if it is no plan."""
    try:
        plan = strict_record(read_bounded(path / PLAN, MAX_PLAN_BYTES))
    except EvidenceError:
        return None
    return (
        plan if isinstance(plan, dict) and plan.get("format") == PLAN_FORMAT else None
    )


def _signature(root: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    """Every entry under `root` that validation may read (Harbor's `trials/`
    aside), with its kind, time, size and inode: any write changes it."""
    found = []
    for directory, names, files in os.walk(root, followlinks=False):
        if Path(directory) == root and "trials" in names:
            names.remove("trials")
        for name in [*names, *files]:
            path = Path(directory) / name
            try:
                info = path.lstat()
            except OSError:
                continue
            found.append(
                (
                    path.relative_to(root).as_posix(),
                    info.st_mode,
                    info.st_mtime_ns,
                    info.st_size,
                    info.st_ino,
                )
            )
    return tuple(sorted(found))


@dataclass
class RunState:
    status: str
    detail: str
    plan: dict[str, Any] | None
    view: RunView | None


@dataclass
class EvaluationState:
    status: str
    detail: str
    record: dict[str, Any] | None


class Pages:
    """The pages over a fixed set of roots; validation results are kept until
    the files they read change."""

    def __init__(self, roots: list[Path], *, control: bool = False) -> None:
        self.roots = roots
        self.control = control
        self.digest = harness_digest()
        self.entries: dict[str, catalog.Entry] = {}
        self._cache: dict[Path, tuple[Any, Any]] = {}
        self._lock = threading.Lock()

    def page(self, title: str, body: str) -> str:
        return page(title, body, control=self.control)

    def entry(self, key: str, kind: str) -> catalog.Entry | None:
        if key not in self.entries:
            self.entries = catalog.scan(self.roots)
        found = self.entries.get(key)
        return found if found is not None and found.kind == kind else None

    def _cached(self, path: Path, signature: Any, compute: Any) -> Any:
        with self._lock:
            hit = self._cache.get(path)
        if hit is not None and hit[0] == signature:
            return hit[1]
        value = compute()
        with self._lock:
            self._cache[path] = (signature, value)
        return value

    def run_state(self, path: Path) -> RunState:
        return self._cached(path, _signature(path), lambda: self._run_state(path))

    def _run_state(self, path: Path) -> RunState:
        plan = _plan(path)
        if plan is None:
            return RunState(INVALID, "No plan record of this run contract", None, None)
        if plan.get("harness_sha256") != self.digest:
            return RunState(
                OTHER_BUILD,
                f"Written by build {_short(plan.get('harness_sha256'))}; this build"
                f" is {self.digest[:12]} and validates only its own runs",
                plan,
                None,
            )
        try:
            view = read_run(path, complete=False)
        except RecordError as error:
            return RunState(INVALID, str(error), plan, None)
        if view.run is None:
            done = len(view.episodes)
            return RunState(
                INTERRUPTED,
                f"No run.json: {done} of {len(view.plan['episodes'])} episodes"
                " recorded (interrupted, or still running)",
                plan,
                view,
            )
        return RunState(COMPLETE, "Validated by this build", plan, view)

    def evaluation_state(self, path: Path) -> EvaluationState:
        return self._cached(
            path, _signature(path), lambda: self._evaluation_state(path)
        )

    def _evaluation_state(self, path: Path) -> EvaluationState:
        try:
            record, _ = check_evaluation(path)
        except RecordError as error:
            return EvaluationState(INVALID, str(error), None)
        written = record["evaluator"]["harness_sha256"]
        detail = "Checked by this build: shape, consistency and every file's digest" + (
            "; this build wrote it, so its renderings were compared too"
            if this_build(record)
            else f"; written by build {written[:12]}"
        )
        return EvaluationState(COMPLETE, detail, record)

    def index(self) -> str:
        self.entries = catalog.scan(self.roots)
        rows = []
        for entry in self.entries.values():
            link = f'<a href="/{entry.kind}/{entry.key}">{_text(entry.label)}</a>'
            if entry.kind == catalog.RUN:
                plan = _plan(entry.path)
                finished = (entry.path / RUN).exists()
                if plan is None:
                    cells = ["no plan record", "", "", ""]
                else:
                    cells = [
                        _text(plan.get("model_id", "")),
                        _text(_cases(plan)),
                        "this"
                        if plan.get("harness_sha256") == self.digest
                        else "other",
                        "run.json" if finished else "no run.json",
                    ]
            else:
                cells = ["", "", "", ""]
            rows.append(
                f"<tr><td>{link}</td><td>{entry.kind}</td>"
                + "".join(f"<td>{cell}</td>" for cell in cells)
                + "</tr>"
            )
        roots = ", ".join(f"<code>{_text(root.name)}</code>" for root in self.roots)
        table = (
            '<div class="table"><table><thead><tr><th>Directory</th><th>Kind</th>'
            "<th>Model</th><th>Cases</th><th>Build</th><th>State</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            if rows
            else "<p>No run or evaluation directories found.</p>"
        )
        return self.page(
            "Directories",
            f"<h1>Runs and evaluations</h1><p>Under {roots}. Model, cases and"
            " state come from each run's files as written; a run's own page"
            " validates it.</p>" + table,
        )

    def run(self, entry: catalog.Entry) -> str:
        state = self.run_state(entry.path)
        head = (
            f"<h1>{_text(entry.label)}</h1><p>{badge(state.status)}"
            f" {_text(state.detail)}</p>"
        )
        if state.status == COMPLETE and state.view is not None:
            body = (
                '<p class="note">Rendered from the records this build validated,'
                " as <code>werkbank report</code> renders them.</p>"
                + to_html(render(state.view)[0])
            )
        elif state.status == INTERRUPTED and state.view is not None:
            body = self._episodes(state.view)
        elif state.status == OTHER_BUILD and state.plan is not None:
            body = self._stored_report(entry.path, state.plan)
        else:
            body = ""
        return self.page(entry.label, head + body)

    def _episodes(self, view: RunView) -> str:
        rows = "".join(
            f"<tr><td>{_text(e.row['name'])}</td>"
            f"<td>{_text(e.record['result'].get('status'))}</td>"
            f"<td>{_text(e.record['result'].get('reason'))}</td></tr>"
            for e in view.episodes
        )
        return (
            "<h2>Recorded episodes</h2>"
            '<div class="table"><table><thead><tr><th>Episode</th><th>Status</th>'
            f"<th>Reason</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )

    def _stored_report(self, path: Path, plan: dict[str, Any]) -> str:
        try:
            data = read_bounded(path / REPORT, MAX_REPORT_BYTES)
        except EvidenceError:
            data = None
        if data is not None:
            return (
                '<p class="note">The report the run\'s build wrote. This build did'
                " not validate the run, so nothing below is checked.</p>"
                + to_html(data.decode("utf-8", errors="replace"))
            )
        items = [
            ("Run", plan.get("run_id")),
            ("Model", plan.get("model_id")),
            ("Cases", _cases(plan)),
            ("Harness", plan.get("harness_sha256")),
        ]
        listed = "".join(
            f"<li>{name}: <code>{_text(value)}</code></li>" for name, value in items
        )
        return (
            "<p>No report; <code>werkbank report</code> of the run's build renders"
            " one. From plan.json, unchecked:</p>"
            f"<ul>{listed}</ul>"
        )

    def evaluation(self, entry: catalog.Entry) -> str:
        state = self.evaluation_state(entry.path)
        head = (
            f"<h1>{_text(entry.label)}</h1><p>{badge(state.status)}"
            f" {_text(state.detail)}</p>"
        )
        if state.record is None:
            return self.page(entry.label, head)
        links = [
            f'<a href="/evaluation/{entry.key}/{name}">{name.replace("-", " ")}</a>'
            for name, file in DOCUMENTS.items()
            if file in state.record["files"]
        ] + [
            f'<a href="/evaluation/{entry.key}/table/{stem}">{_text(stem)}</a>'
            for stem in self._tables(state.record)
        ]
        if _regular(entry.path.parent / RESULT_DIRECTORY / PAGE):
            links.insert(0, f'<a href="/evaluation/{entry.key}/result">result page</a>')
        body = self._markdown(entry.path, state.record, "evaluation.md")
        return self.page(
            entry.label,
            head + f'<nav class="files">{" · ".join(links)}</nav>' + body,
        )

    def result(self, entry: catalog.Entry) -> str | None:
        """The result page of the series this evaluation belongs to, in the
        interface's dark scheme: only while the `result` directory beside
        the evaluation is exactly what this build renders for it."""
        return checked_result(entry.path, entry.path.parent / RESULT_DIRECTORY)

    def _tables(self, record: dict[str, Any]) -> list[str]:
        return [
            name[len(TABLES) : -len(".csv")]
            for name in record["files"]
            if name.startswith(TABLES) and name.endswith(".csv")
        ]

    def _read(self, path: Path, record: dict[str, Any], name: str) -> str | None:
        """A file of a checked evaluation, only while it still has the digest
        its record binds; None otherwise."""
        try:
            data = read_bounded(path / name, MAX_FILE_BYTES)
        except EvidenceError:
            return None
        if hashlib.sha256(data).hexdigest() != record["files"].get(name):
            return None
        return data.decode("utf-8", errors="replace")

    def _markdown(self, path: Path, record: dict[str, Any], name: str) -> str:
        text = self._read(path, record, name)
        return CHANGED if text is None else to_html(text)

    def document(self, entry: catalog.Entry, name: str) -> str | None:
        state = self.evaluation_state(entry.path)
        file = DOCUMENTS.get(name)
        if state.record is None or file is None or file not in state.record["files"]:
            return None
        title = f"{entry.label}: {name.replace('-', ' ')}"
        back = f'<p><a href="/evaluation/{entry.key}">back to the evaluation</a></p>'
        return self.page(title, back + self._markdown(entry.path, state.record, file))

    def table(self, entry: catalog.Entry, stem: str) -> str | None:
        state = self.evaluation_state(entry.path)
        if state.record is None or stem not in self._tables(state.record):
            return None
        text = self._read(entry.path, state.record, f"{TABLES}{stem}.csv")
        back = f'<p><a href="/evaluation/{entry.key}">back to the evaluation</a></p>'
        if text is None:
            return self.page(f"{entry.label}: {stem}", back + CHANGED)
        # A cell may be as long as the file the record admits (a pool's run
        # list grows with its runs); the module's default limit is 128 KiB.
        csv.field_size_limit(MAX_FILE_BYTES)
        rows = list(csv.reader(io.StringIO(text, newline="")))
        header, body = (rows[0], rows[1:]) if rows else ([], [])
        head = "".join(f"<th>{_text(cell)}</th>" for cell in header)
        cells = "".join(
            "<tr>" + "".join(f"<td>{_text(cell)}</td>" for cell in row) + "</tr>"
            for row in body
        )
        return self.page(
            f"{entry.label}: {stem}",
            back
            + f"<h1>{_text(stem)}.csv</h1>"
            + f'<div class="table"><table><thead><tr>{head}</tr></thead>'
            + f"<tbody>{cells}</tbody></table></div>",
        )
