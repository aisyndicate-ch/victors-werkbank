"""One `werkbank` command as a child process, exactly as the command line
runs it, so a pass keeps the CLI's admission, exit codes, Ctrl-C handling and
evidence rules."""

import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import FrameType

COMMAND = (sys.executable, "-m", "victors_werkbank")


@dataclass(frozen=True)
class Child:
    """A finished child: its exit code, and whether Ctrl-C reached it."""

    code: int
    interrupted: bool


def run(
    argv: Sequence[str],
    *,
    command: Sequence[str] = COMMAND,
    env: Mapping[str, str] | None = None,
) -> Child:
    """Run `werkbank argv` and wait for it. It reads no input, its stdout (a
    JSON summary start does not use) is discarded and its stderr (progress,
    the per-case table and its messages) reaches the tester's terminal.

    The child inherits this process's environment, and with it an API key in
    `WERKBANK_API_KEY`, which `werkbank run` reads itself; `env` replaces
    that environment for a series whose server gets no key.

    Ctrl-C reaches the child, which ends its run by its own rules; this
    process only notes it, with a handler of its own and never by ignoring
    the signal, which the child would inherit and could then not be
    interrupted. OSError where the child cannot start."""
    interrupted = False

    def note(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True

    previous = signal.signal(signal.SIGINT, note)
    try:
        process = subprocess.Popen(
            [*command, *argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            env=None if env is None else dict(env),
        )
        code = process.wait()
    finally:
        signal.signal(signal.SIGINT, previous)
    return Child(code, interrupted)
