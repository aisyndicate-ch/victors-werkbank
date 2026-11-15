"""The control side of the interface: one guided start at a time.

A session runs the steps of `werkbank start` (`werkbank_start.flow`) on a
worker thread, with the browser in place of the terminal: what the start
says becomes the session's lines, what it asks becomes one pending question
that a form answers. Every pass stays a `werkbank run` child process; a stop
reaches it as the interrupt Ctrl-C sends, so the run ends by its own rules.
The offer to submit runs the submit flow of `werkbank_submit` the same way,
and only that flow contacts a receiver. Nothing here is stored: the endpoint
address, the lines and the answers live in this process only. An API key is
never entered here: it comes from `WERKBANK_API_KEY` in the environment of
the process that started the interface, and the start decides which server
gets it.
"""

import os
import secrets
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from werkbank_start import child
from werkbank_start.console import Console, Stopped
from werkbank_start.flow import Boundaries, Options, start
from werkbank_submit import flow as submit_flow

# The newest lines a session keeps; older ones are dropped and counted.
MAX_LINES = 4000
MAX_LINE_CHARS = 2000
MAX_ANSWER_CHARS = 2048
# The lines said since the last question that a question shows as its context.
MAX_CONTEXT_LINES = 400

SAID = "said"
ERROR = "error"
RUN = "run"
ANSWER = "answer"

CHOOSE = "choose"
CONFIRM = "confirm"
TEXT = "text"

# The steps of a start in order, as `werkbank_start.flow` names them, with
# their titles. A start that asks nothing passes some within a moment.
STEPS: tuple[tuple[str, str], ...] = (
    ("readiness", "Readiness"),
    ("server", "Server"),
    ("model", "Model"),
    ("check", "Server check"),
    ("mode", "Mode"),
    ("pass", "Passes"),
    ("evaluation", "Evaluation"),
    ("result", "Result"),
    ("submission", "Submission"),
)
# The steps before the series folder exists: a stop there ends the start at
# once. From the first pass on, a stop goes to the running pass instead.
BEFORE_SERIES = frozenset({"readiness", "server", "model", "check", "mode"})

EPISODE_STARTED = "running"
PROGRESS = "progress: "
# The verifier's verdict on an episode. A code episode ends without a result
# of its own, as its verifier grades it, so this is the only line that says
# how it ended; for a host episode it repeats the result.
VERDICT = "harbor: "
STARTED = " started"

# The exit codes of a start, in words.
OUTCOMES = {
    0: "Complete: every pass was measured and the result page is written.",
    1: "Stopped by you.",
    2: "Refused or failed; the lines below name the reason.",
    3: "Evaluated, but a pass left cases unmeasured; the result page shows bounds.",
}


class Busy(Exception):
    """A session is still running; only one runs at a time."""


class Stale(Exception):
    """The answer is not for the pending question, or not one of its choices."""


@dataclass(frozen=True)
class Question:
    """One pending question: what is asked, how it is answered and the lines
    said since the question before it, which a confirmation refers to."""

    id: str
    kind: str
    prompt: str
    options: tuple[str, ...] = ()
    default: bool | None = None
    skippable: bool = False
    context: tuple[str, ...] = ()


@dataclass
class Pass:
    """A pass as its `werkbank run` reports it on stderr: display only, the
    run's own records stay authoritative."""

    seed: int
    folder: Path
    planned: int
    started: float
    seconds: float | None = None
    # Episode name -> (state, reason), in the order the run started them.
    episodes: dict[str, tuple[str, str]] = field(default_factory=dict)

    def finished(self) -> int:
        return sum(1 for state, _ in self.episodes.values() if state != EPISODE_STARTED)


@dataclass(frozen=True)
class Snapshot:
    """A session at one moment, for a page."""

    step: str | None
    lines: tuple[tuple[str, str], ...]
    dropped: int
    question: Question | None
    passes: tuple[Pass, ...]
    running: bool
    child_running: bool
    stopping: bool
    code: int | None
    folder: Path | None
    evaluation: Path | None
    result: Path | None
    # A trial or a single pass: the series has no submission step.
    unsubmitted: bool
    seconds: float
    # The session clock's reading at the snapshot.
    now: float


def _printable(text: str) -> str:
    """`text` as one bounded line without control characters."""
    line = "".join(
        ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii")
        for ch in text[:MAX_LINE_CHARS]
    )
    return line + ("…" if len(text) > MAX_LINE_CHARS else "")


def _block(text: str) -> str:
    """`text` with each of its lines made printable."""
    return "\n".join(_printable(part) for part in text.split("\n"))


class Session:
    """One guided start and what it has said, asked and run so far."""

    def __init__(
        self,
        options: Options,
        env: Boundaries,
        *,
        command: Sequence[str] = child.COMMAND,
        submit: dict[str, Any] | None = None,
    ) -> None:
        self._options = options
        self._env = env
        self._command = tuple(command)
        self._submit = dict(submit or {})
        self._changed = threading.Condition()
        self._lines: list[tuple[str, str]] = []
        self._dropped = 0
        self._context: list[str] = []
        self._question: Question | None = None
        self._reply: Any = None
        self._replied = False
        self._step: str | None = None
        self._passes: list[Pass] = []
        self._process: subprocess.Popen[str] | None = None
        self._in_pass = False
        self._stopping = False
        self._code: int | None = None
        self._folder: Path | None = None
        self._evaluation: Path | None = None
        self._result: Path | None = None
        self._unsubmitted = False
        self._began = env.clock()
        self._ended: float | None = None
        self._thread = threading.Thread(
            target=self._work, name="werkbank-ui-session", daemon=True
        )

    def begin(self) -> None:
        self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Whether the start has ended within `timeout`."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # The worker thread.

    def _work(self) -> None:
        env = self._env
        if env.runner is child.run:
            env.runner = self._run
        else:
            given = env.runner
            env.runner = lambda argv, **how: self._guarded(given, argv, **how)
        env.submitter = self._send
        try:
            code = start(self._options, env, BrowserConsole(None, None, None, self))
        except Exception as error:  # A fault here must not leave a page waiting.
            self.say(f"Internal error ({type(error).__name__}).", ERROR)
            code = 2
        with self._changed:
            self._code = code
            self._ended = env.clock()
            self._question = None
            self._changed.notify_all()

    def _guarded(
        self, runner: Callable[..., child.Child], argv: list[str], **how: Any
    ) -> child.Child:
        """A given runner, not started once a stop is requested; `how` is
        the environment the start gives a pass, where it gives one."""
        with self._changed:
            if self._stopping:
                self._stopping = False
                return child.Child(1, True)
            self._in_pass = True
        try:
            return runner(argv, **how)
        finally:
            self._pass_ended()

    def _run(
        self, argv: Sequence[str], *, env: Mapping[str, str] | None = None
    ) -> child.Child:
        """`werkbank argv` as a child process in a process group of its own.
        Its stderr (progress and messages) becomes lines of the session; a
        requested stop interrupts the group once, as Ctrl-C at a terminal
        interrupts a run and what it started. It inherits this process's
        environment unless the start gives `env`, the environment of a
        series whose server gets no API key."""
        with self._changed:
            if self._stopping:
                self._stopping = False
                return child.Child(1, True)
            process = subprocess.Popen(
                [*self._command, *argv],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                process_group=0,
                env=None if env is None else dict(env),
            )
            self._process = process
            self._in_pass = True
            self._changed.notify_all()
        assert process.stderr is not None
        with process.stderr as stream:
            for line in stream:
                self._child_line(line.rstrip("\r\n"))
        code = process.wait()
        with self._changed:
            self._process = None
            interrupted = self._stopping
            self._stopping = False
            self._changed.notify_all()
        self._pass_ended()
        return child.Child(code, interrupted)

    def _pass_ended(self) -> None:
        with self._changed:
            self._in_pass = False
            if self._passes and self._passes[-1].seconds is None:
                current = self._passes[-1]
                current.seconds = self._env.clock() - current.started

    def _child_line(self, line: str) -> None:
        self.say(line, RUN)
        verdict = line.startswith(VERDICT)
        if not (verdict or line.startswith(PROGRESS)):
            return
        rest = line[len(VERDICT if verdict else PROGRESS) :]
        with self._changed:
            if not self._passes:
                return
            episodes = self._passes[-1].episodes
            if not verdict and rest.endswith(STARTED):
                episodes[_printable(rest[: -len(STARTED)])] = (EPISODE_STARTED, "")
                return
            name, _, outcome = rest.partition(" ")
            status, _, reason = outcome.partition(" ")
            # A verdict counts only for an episode the run reported as started.
            if verdict and _printable(name) not in episodes:
                return
            episodes[_printable(name)] = (
                _printable(status),
                _printable(reason.strip("()")),
            )

    def _send(
        self, evaluation: Path, result: Path, sources: list[Path], **kwargs: Any
    ) -> int:
        """The submit flow of `werkbank_submit`, with its dialog in the browser."""
        return submit_flow.submit(
            evaluation,
            result,
            sources,
            endpoint=kwargs.get("endpoint"),
            console=BrowserSubmission(self),
            **self._submit,
        )

    # What the consoles call.

    def say(self, text: str, kind: str = SAID) -> None:
        with self._changed:
            for part in text.split("\n"):
                line = _printable(part)
                self._lines.append((kind, line))
                if kind in {SAID, ERROR}:
                    self._context.append(line)
            excess = len(self._lines) - MAX_LINES
            if excess > 0:
                del self._lines[:excess]
                self._dropped += excess
            del self._context[:-MAX_CONTEXT_LINES]
            self._changed.notify_all()

    def step(self, name: str, facts: dict[str, Any]) -> None:
        with self._changed:
            if self._stopping and name in BEFORE_SERIES:
                self._stopping = False
                raise Stopped
            self._step = name
            if name == "series":
                self._folder = facts["folder"]
                self._unsubmitted = not facts["mode"].submitted
                self._step = "pass"
            elif name == "pass":
                self._passes.append(
                    Pass(
                        facts["seed"],
                        facts["folder"],
                        facts["episodes"],
                        self._env.clock(),
                    )
                )
            elif name == "result":
                self._evaluation = facts["evaluation"]
                self._result = facts["result"]
            self._changed.notify_all()

    def ask(
        self,
        kind: str,
        prompt: str,
        *,
        options: Sequence[str] = (),
        default: bool | None = None,
        skippable: bool = False,
        keep: bool = False,
    ) -> Any:
        """Block until the pending question is answered; `Stopped` when the
        start is stopped instead. The question shows the lines said since
        the last one as its context; with `keep` they stay for the next
        question too, so a later confirmation still stands under them."""
        with self._changed:
            if self._stopping:
                self._stopping = False
                raise Stopped
            question = Question(
                secrets.token_urlsafe(16),
                kind,
                _printable(prompt),
                tuple(_block(option) for option in options),
                default,
                skippable,
                tuple(self._context),
            )
            if not keep:
                self._context.clear()
            self._question = question
            self._replied = False
            self._changed.notify_all()
            while not self._replied and not self._stopping:
                self._changed.wait()
            self._question = None
            if not self._replied:
                self._stopping = False
                self._lines.append((ANSWER, f"{question.prompt} (stopped)"))
                self._changed.notify_all()
                raise Stopped
            reply = self._reply
            self._reply = None
            self._replied = False
            if kind == CHOOSE:
                shown = (
                    "skipped"
                    if reply is None
                    else question.options[reply].split("\n")[0]
                )
            elif kind == CONFIRM:
                shown = "yes" if reply else "no"
            else:
                shown = _printable(reply)
            self._lines.append((ANSWER, f"{question.prompt} {shown}"))
            self._changed.notify_all()
            return reply

    # What the server calls.

    def answer(self, question_id: str, value: str) -> None:
        """Answer the pending question with a form's `value`; `Stale` when it
        is not the pending one or `value` is none of its answers."""
        with self._changed:
            question = self._question
            if (
                question is None
                or self._replied
                or not secrets.compare_digest(question.id, question_id)
            ):
                raise Stale
            reply: Any
            if question.kind == CHOOSE:
                if value == "" and question.skippable:
                    reply = None
                elif (
                    value.isascii()
                    and value.isdigit()
                    and int(value) < len(question.options)
                ):
                    reply = int(value)
                else:
                    raise Stale
            elif question.kind == CONFIRM:
                if value not in {"yes", "no"}:
                    raise Stale
                reply = value == "yes"
            else:
                if len(value) > MAX_ANSWER_CHARS or not value.isprintable():
                    raise Stale
                reply = value
            self._reply = reply
            self._replied = True
            self._changed.notify_all()

    def stop(self) -> None:
        """Stop the start: a running pass is interrupted once and ends by its
        own rules; a pending question ends the start; anything else stops at
        the next step."""
        with self._changed:
            if self._code is not None or self._stopping:
                return
            self._stopping = True
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except OSError:
                    pass
            self._changed.notify_all()

    def running(self) -> bool:
        with self._changed:
            return self._code is None

    def snapshot(self) -> Snapshot:
        with self._changed:
            ended = self._env.clock() if self._ended is None else self._ended
            return Snapshot(
                step=self._step,
                lines=tuple(self._lines),
                dropped=self._dropped,
                question=self._question if not self._replied else None,
                passes=tuple(
                    Pass(
                        p.seed,
                        p.folder,
                        p.planned,
                        p.started,
                        p.seconds,
                        dict(p.episodes),
                    )
                    for p in self._passes
                ),
                running=self._code is None,
                child_running=self._in_pass,
                stopping=self._stopping,
                code=self._code,
                folder=self._folder,
                evaluation=self._evaluation,
                result=self._result,
                unsubmitted=self._unsubmitted,
                seconds=ended - self._began,
                now=ended,
            )


class BrowserConsole(Console):
    """The start's console in the browser: a person answers its questions
    there, so it confirms as a terminal does."""

    def __init__(self, stdin: Any, stdout: Any, stderr: Any, session: Session) -> None:
        super().__init__(stdin, stdout, stderr)
        self.session = session

    def say(self, text: str = "") -> None:
        self.session.say(text)

    def error(self, text: str) -> None:
        self.session.say(text, ERROR)

    def terminal(self) -> bool:
        return True

    def step(self, name: str, **facts: object) -> None:
        self.session.step(name, facts)

    def submission(self) -> Any:
        return BrowserSubmission(self.session)

    def answer(self, prompt: str) -> str:
        reply: str = self.session.ask(TEXT, prompt)
        return reply

    def confirm(self, prompt: str, *, default: bool) -> bool:
        reply: bool = self.session.ask(CONFIRM, prompt, default=default)
        return reply

    def choose(self, title: str, options: list[str]) -> int:
        reply: int = self.session.ask(CHOOSE, title, options=options)
        return reply


class BrowserSubmission(submit_flow.Console):
    """The submit flow's console in the browser. A stop at one of its
    questions is the end of input there: nothing is sent."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def is_terminal(self) -> bool:
        return True

    def say(self, text: str) -> None:
        self.session.say(text)

    def _ask(self, kind: str, prompt: str, **how: Any) -> Any:
        # Every question of a submission keeps what was said before it: the
        # final confirmation stands under the whole summary and consent text.
        try:
            return self.session.ask(kind, prompt, keep=True, **how)
        except Stopped:
            raise EOFError from None

    def ask(self, prompt: str) -> str:
        reply: str = self._ask(TEXT, prompt)
        return reply

    def choose(self, question: str, labels: Sequence[str]) -> int | None:
        reply: int | None = self._ask(CHOOSE, question, options=labels, skippable=True)
        return reply

    def confirm(self, prompt: str) -> bool:
        reply: bool = self._ask(CONFIRM, prompt, default=False)
        return reply


class Control:
    """The sessions of one interface: at most one at a time, new series under
    `runs`. `boundaries` gives a start's boundaries (tests replace them),
    `command` the `werkbank` command of a pass and `submit` extra arguments
    of the submit flow."""

    def __init__(
        self,
        runs: Path,
        defaults: Any = None,
        *,
        boundaries: Callable[[], Boundaries] | None = None,
        command: Sequence[str] = child.COMMAND,
        submit: dict[str, Any] | None = None,
    ) -> None:
        self.runs = runs
        self.defaults = defaults
        self.form_token = secrets.token_urlsafe(32)
        self._boundaries = boundaries
        self._command = command
        self._submit = submit
        self._lock = threading.Lock()
        self.session: Session | None = None

    def begin(self, options: Options) -> Session:
        with self._lock:
            if self.session is not None and self.session.running():
                raise Busy
            if self._boundaries is not None:
                env = self._boundaries()
            else:
                env = Boundaries(None, None, None, runs=self.runs)
            self.session = Session(
                options, env, command=self._command, submit=self._submit
            )
            self.session.begin()
            return self.session

    def close(self) -> None:
        """Forget an ended session; a running one stays."""
        with self._lock:
            if self.session is not None and not self.session.running():
                self.session = None

    def shutdown(self, patience: float = 600.0) -> None:
        """Stop a running session and wait for its pass to end by its own
        rules, so the interface never leaves a run behind unstopped."""
        session = self.session
        if session is None or not session.running():
            return
        deadline = time.monotonic() + patience
        while not session.wait(0.5) and time.monotonic() < deadline:
            # A question or a new step may follow the stopped pass.
            session.stop()
