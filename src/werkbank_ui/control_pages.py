"""The pages of the control side: the form that begins a series and the page
that follows it. Every value is escaped; nothing here starts or reads
anything, the pages only show a session's snapshot."""

import html
import re
from typing import Any

from werkbank_start import discovery
from werkbank_start.console import duration
from werkbank_start.flow import TRIAL_TEXT

from . import control
from .pages import page

REFRESH_SECONDS = 5
# The newest lines of a session the page shows.
SHOWN_LINES = 400
YES_NO = re.compile(r"\s*\[[yY]/[nN]\]\s*$")
LEVELS = ("off", "low", "medium", "high", "xhigh", "max")

# The form's fields: name, the `werkbank start` flag it stands for, label, hint.
FIELDS: tuple[tuple[str, str, str, str], ...] = (
    (
        "endpoint",
        "--endpoint",
        "Server address",
        "The /v1 base URL of your server, for example http://127.0.0.1:8000/v1."
        " Empty: Werkbank looks on this computer.",
    ),
    (
        "model",
        "--model",
        "Model",
        "The served model name. Empty: chosen from the server's list.",
    ),
    (
        "hardware_label",
        "--hardware-label",
        "Hardware label",
        "The serving hardware in your words, 1 to 64 characters. Optional.",
    ),
    (
        "max_tokens_per_call",
        "--max-tokens-per-call",
        "Per-response cap",
        "1 to 65,536 tokens. Empty: 65,536 where the server's context covers"
        " the mode with it, else 32,768.",
    ),
    ("temperature", "--temperature", "Temperature", "0 to 2. Empty: server default."),
    ("top_p", "--top-p", "Top-p", "Above 0, at most 1. Empty: server default."),
    ("top_k", "--top-k", "Top-k", "1 to 1,000,000. Empty: server default."),
)
SETTINGS_FROM = 2

MARKS = {control.ERROR: "! ", control.ANSWER: "> ", control.RUN: "  ", control.SAID: ""}


def _duration(seconds: float) -> str:
    """A time span in words; under a minute in seconds."""
    if seconds < 60:
        return f"{int(seconds)} s"
    return duration(seconds)


def _text(value: Any) -> str:
    return html.escape(str(value))


def _hidden(token: str) -> str:
    return f'<input type="hidden" name="form" value="{_text(token)}">'


def _default(defaults: Any, name: str) -> str:
    value = getattr(defaults, name, None) if defaults is not None else None
    return "" if value is None or value is False else str(value)


def new_series(token: str, defaults: Any, problem: str | None = None) -> str:
    """The form that begins a series, with the command line's values as its
    defaults and, after a refused form, why."""
    ports = ", ".join(str(port) for port in discovery.USUAL_PORTS)
    rows = []
    for index, (name, _, label, hint) in enumerate(FIELDS):
        if index == SETTINGS_FROM:
            rows.append(
                "<details><summary>Settings (all optional; the server's defaults"
                " apply where a field is empty)</summary>"
            )
        rows.append(
            f'<div class="field"><label for="{name}">{label}</label>'
            f'<input type="text" id="{name}" name="{name}" autocomplete="off"'
            f' spellcheck="false" value="{_text(_default(defaults, name))}">'
            f'<p class="hint">{_text(hint)}</p></div>'
        )
    chosen = _default(defaults, "thinking")
    levels = "".join(
        f'<option value="{level}"{" selected" if level == chosen else ""}>'
        f"{level or 'not sent'}</option>"
        for level in ("", *LEVELS)
    )
    stream = " checked" if getattr(defaults, "stream", False) else ""
    rows.append(
        '<div class="field"><label for="thinking">Thinking</label>'
        f'<select id="thinking" name="thinking">{levels}</select>'
        '<p class="hint">A level the served model\'s recipe supports; sent as'
        " template variables only when set.</p></div>"
        '<div class="field"><label><input type="checkbox" name="stream"'
        f'{stream}> Stream responses</label><p class="hint">Streamed and'
        " non-streamed runs form separate pools.</p></div></details>"
    )
    refused = (
        f'<p class="problem" role="alert">Not started: {_text(problem)}</p>'
        if problem
        else ""
    )
    return page(
        "New series",
        "<h1>New series</h1>"
        '<p class="lead">The guided start, as <code>werkbank start</code> runs'
        " it: readiness of Docker and the image, the server, the model, a"
        " server check, the mode, the passes, the evaluation and the result"
        " page.</p>"
        f"{refused}"
        '<form method="post" action="/series/start" class="panel">'
        f"{_hidden(token)}{''.join(rows)}"
        '<p><button type="submit" class="primary">Begin</button></p></form>'
        '<h2>What happens, and when</h2><ul class="plain">'
        "<li>Without a server address, Werkbank sends one read-only model list"
        f" request to each usual port on 127.0.0.1 ({ports}) and to no other"
        " host.</li>"
        "<li>A request to the model goes out only after your choice, or with"
        " exactly one server and one model found.</li>"
        "<li>You choose the mode after the server check: Trial ("
        f"{_text(TRIAL_TEXT[0].lower() + TRIAL_TEXT[1:])}), Quick, Normal or"
        " Full, each shown with its scope and context need. For Quick, Normal"
        " and Full you then choose three passes or one; a single pass is never"
        " submitted.</li>"
        "<li>A running pass can be stopped, not paused: a stopped pass stays as"
        " evidence and is run again as a whole.</li>"
        "<li>The series runs in the process that serves this page. Closing the"
        " browser changes nothing; ending that process stops the running"
        " pass.</li></ul>",
        control=True,
    )


def _steps(snapshot: control.Snapshot) -> str:
    names = [name for name, _ in control.STEPS]
    at = names.index(snapshot.step) if snapshot.step in names else -1
    items = []
    for index, (name, title) in enumerate(control.STEPS):
        if name == "submission" and snapshot.unsubmitted:
            continue
        if index < at or (not snapshot.running and index == at and snapshot.code == 0):
            items.append(f'<li class="done">{title} <span>done</span></li>')
        elif index == at:
            word = "now" if snapshot.running else "ended here"
            items.append(
                f'<li class="current" aria-current="step">{title}'
                f" <span>{word}</span></li>"
            )
        else:
            items.append(f"<li>{title}</li>")
    return f'<ol class="steps">{"".join(items)}</ol>'


def _question(question: control.Question, token: str) -> str:
    context = (
        '<pre class="context">'
        + "\n".join(_text(line) for line in question.context)
        + "</pre>"
        if question.context
        else ""
    )
    head = (
        '<section class="question" id="question"><h2>Your answer is needed</h2>'
        f'{context}<form method="post" action="/series/answer">{_hidden(token)}'
        f'<input type="hidden" name="question" value="{_text(question.id)}">'
    )
    if question.kind == control.CHOOSE:
        choices = []
        for index, option in enumerate(question.options):
            first, *rest = option.split("\n")
            detail = "".join(
                f'<span class="detail">{_text(line.strip())}</span>' for line in rest
            )
            choices.append(
                f'<label class="choice"><input type="radio" name="value"'
                f' value="{index}" required><span><strong>{_text(first)}</strong>'
                f"{detail}</span></label>"
            )
        if question.skippable:
            choices.append(
                '<label class="choice"><input type="radio" name="value" value=""'
                " checked><span><strong>Skip this question</strong></span></label>"
            )
        body = (
            f"<fieldset><legend>{_text(question.prompt)}</legend>"
            f"{''.join(choices)}</fieldset>"
            '<p><button type="submit" class="primary">Continue</button></p>'
        )
    elif question.kind == control.CONFIRM:
        yes = ' class="primary"' if question.default else ""
        no = "" if question.default else ' class="primary"'
        body = (
            f'<p class="prompt">{_text(YES_NO.sub("", question.prompt))}</p>'
            f'<p><button type="submit" name="value" value="yes"{yes}>Yes</button> '
            f'<button type="submit" name="value" value="no"{no}>No</button></p>'
        )
    else:
        body = (
            f'<p class="prompt"><label for="value">{_text(question.prompt)}</label></p>'
            '<p><input type="text" id="value" name="value" autocomplete="off"'
            ' spellcheck="false"> <button type="submit" class="primary">Continue'
            "</button></p>"
        )
    return f"{head}{body}</form></section>"


def _passes(snapshot: control.Snapshot) -> str:
    parts = []
    for index, item in enumerate(snapshot.passes):
        last = index == len(snapshot.passes) - 1
        done = item.finished()
        took = (
            f"running for {_duration(snapshot_seconds(snapshot, item))}"
            if item.seconds is None
            else f"took {_duration(item.seconds)}"
        )
        rows = "".join(
            f'<tr><td>{_text(name)}</td><td class="status {_slug(state)}">'
            f"{_text(state)}</td><td>{_text(reason)}</td></tr>"
            for name, (state, reason) in item.episodes.items()
        )
        table = (
            '<div class="table"><table><thead><tr><th>Episode</th><th>Status</th>'
            f"<th>Reason</th></tr></thead><tbody>{rows}</tbody></table></div>"
            if rows
            else '<p class="note">No episode has started yet.</p>'
        )
        parts.append(
            f"<details{' open' if last else ''}><summary>Pass with seed {item.seed}:"
            f" {done} of {item.planned} episodes finished, {took}"
            f" <code>{_text(item.folder.name)}</code></summary>{table}</details>"
        )
    if not parts:
        return ""
    return (
        "<h2>Passes</h2>"
        '<p class="note">As each run reports them while it works; the run\'s'
        " records, shown under Runs and evaluations, are what counts.</p>"
        + "".join(parts)
    )


def snapshot_seconds(snapshot: control.Snapshot, item: control.Pass) -> float:
    """How long a pass that has not ended has run: from its start to the
    snapshot, both on the session's clock."""
    return max(0.0, snapshot.now - item.started)


def _slug(state: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in state)[:32]


def _log(snapshot: control.Snapshot) -> str:
    lines = snapshot.lines[-SHOWN_LINES:]
    hidden = snapshot.dropped + len(snapshot.lines) - len(lines)
    text = "\n".join(
        f'<span class="{kind}">{MARKS.get(kind, "")}{_text(line)}</span>'
        for kind, line in lines
    )
    earlier = (
        f'<p class="note">{hidden} earlier lines are not shown.</p>' if hidden else ""
    )
    return (
        '<h2 id="log">Log</h2><p class="note">What the start and its runs said,'
        " newest last. Lines of a run are indented, your answers begin with"
        f' &gt;, problems with !.</p>{earlier}<pre class="log">{text}</pre>'
    )


def series(snapshot: control.Snapshot, token: str) -> str:
    """The page of the session: where the start stands, the pending question,
    the passes, the result page and the log."""
    name = snapshot.folder.name if snapshot.folder is not None else "not started yet"
    took = _duration(snapshot.seconds)
    if snapshot.running:
        state = "stopping" if snapshot.stopping else "running"
        if snapshot.question is not None:
            state = "waiting for you"
        status = f'<span class="badge {_slug(state)}">{state}</span> for {took}'
    else:
        word = {0: "complete", 1: "stopped", 3: "evaluated"}.get(
            snapshot.code or 0, "failed"
        )
        outcome = control.OUTCOMES.get(snapshot.code or 0, control.OUTCOMES[2])
        status = (
            f'<span class="badge {word}">{word}</span> after {took}. {_text(outcome)}'
        )
    head = (
        f"<h1>Series <code>{_text(name)}</code></h1>"
        f'<p class="state">{status}</p>{_steps(snapshot)}'
    )
    asked = "" if snapshot.question is None else _question(snapshot.question, token)
    result = ""
    if snapshot.result is not None:
        result = (
            '<h2>Result page</h2><p class="note">The page written to'
            f" <code>{_text(snapshot.result.name)}/index.html</code> in the series"
            ' folder, shown here in dark. <a href="/series/result">Open it'
            " alone</a>.</p>"
            '<iframe class="result" src="/series/result" title="Result page">'
            "</iframe>"
        )
    parts: list[str] = []
    if snapshot.running:
        what = (
            "The running pass is interrupted as with Ctrl-C and ends by its own"
            " rules, which can take a minute while its containers stop; its"
            " folder stays as evidence."
            if snapshot.child_running
            else "The start ends at this step; nothing more is started."
        )
        if snapshot.stopping:
            parts.append(f'<p class="note">Stop requested. {what}</p>')
        else:
            parts.append(
                '<form method="post" action="/series/stop" class="stop">'
                f'{_hidden(token)}<button type="submit" class="danger">Stop</button>'
                f' <span class="note">{what} There is no pause and no resume.'
                "</span></form>"
            )
    else:
        parts.append(
            '<form method="post" action="/series/close">'
            f'{_hidden(token)}<button type="submit">Close this series</button>'
            ' <span class="note">Its folder stays; the page then offers a new'
            " series.</span></form>"
        )
    refresh = (
        REFRESH_SECONDS if snapshot.running and snapshot.question is None else None
    )
    # The parts the interface's script updates in place on a running series,
    # always all of them and in this order, so the page keeps its form while
    # it stays current; the question stands outside them, as a page with one
    # is loaded whole.
    live = "" if refresh is None else f' data-live="{refresh}"'
    body = (
        f'<div id="series"{live}>'
        f'<div data-part="head">{head}</div>{asked}'
        f'<div data-part="result">{result}</div>'
        f'<div data-part="passes">{_passes(snapshot)}</div>'
        f'<div data-part="controls">{"".join(parts)}</div>'
        f'<div data-part="log">{_log(snapshot)}</div></div>'
    )
    return page(f"Series {name}", body, control=True, refresh=refresh)
