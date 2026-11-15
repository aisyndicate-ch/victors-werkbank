"""The tester's console: prompts and answers on injectable streams.

Prompts and messages go to stdout, refusals and failures to stderr. End of
input or Ctrl-C at a prompt stops the start (`Stopped`); nothing is ever read
from stdin without a prompt, and with `--yes` nothing is read at all.
"""

from dataclasses import dataclass
from typing import Any, TextIO

# Wrong answers asked again at one prompt before it counts as a stop.
MAX_AGAIN = 3


class Stopped(Exception):
    """The tester stopped the start at a prompt: a declined prompt, end of
    input or Ctrl-C."""


def duration(seconds: float) -> str:
    """A duration in hours and minutes, as a tester reads it."""
    minutes = max(1, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def shown(text: str) -> str:
    """`text` for a terminal: characters that are not printable escaped, so
    a server-reported name never moves the cursor or rings the bell."""
    return "".join(
        char if char.isprintable() else char.encode("unicode_escape").decode("ascii")
        for char in text
    )


@dataclass
class Console:
    stdin: TextIO
    stdout: TextIO
    stderr: TextIO

    def say(self, text: str = "") -> None:
        print(text, file=self.stdout, flush=True)

    def error(self, text: str) -> None:
        print(text, file=self.stderr, flush=True)

    def terminal(self) -> bool:
        """A person answers here: both streams are a terminal. Only there
        does a `y` confirm a submission."""
        return self.stdin.isatty() and self.stdout.isatty()

    def step(self, name: str, **facts: object) -> None:
        """The start reaches step `name`, with what is known there. A terminal
        shows nothing for it; the browser interface follows the start by it."""

    def submission(self) -> Any:
        """The console the submit flow talks on: the same streams."""
        from werkbank_submit import flow as submit_flow

        return submit_flow.Console(self.stdin, self.stdout)

    def ask(self, prompt: str) -> str:
        """One line read after `prompt`, without its line end; EOFError at the
        end of input, as `input` raises it."""
        print(prompt, end="", file=self.stdout, flush=True)
        line = self.stdin.readline()
        if not line:
            raise EOFError
        return line.rstrip("\r\n")

    def answer(self, prompt: str) -> str:
        """`ask`, with end of input and Ctrl-C as a stop."""
        try:
            return self.ask(prompt)
        except (EOFError, KeyboardInterrupt):
            self.say()
            raise Stopped from None

    def confirm(self, prompt: str, *, default: bool) -> bool:
        """y or n; Enter takes `default`."""
        for _ in range(MAX_AGAIN + 1):
            reply = self.answer(prompt).strip().lower()
            if reply == "":
                return default
            if reply in {"y", "yes"}:
                return True
            if reply in {"n", "no"}:
                return False
            self.say("Answer y or n.")
        raise Stopped

    def choose(self, title: str, options: list[str]) -> int:
        """The index of the option the tester picks by number; no default."""
        self.say(title)
        for number, option in enumerate(options, 1):
            self.say(f"  {number}) {option}")
        numbers = {str(number): number - 1 for number in range(1, len(options) + 1)}
        for _ in range(MAX_AGAIN + 1):
            reply = self.answer(f"Choice (1-{len(options)}): ").strip()
            if reply in numbers:
                return numbers[reply]
            self.say(f"Enter a number from 1 to {len(options)}.")
        raise Stopped
