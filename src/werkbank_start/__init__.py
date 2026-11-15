"""`werkbank start`: the guided path for a tester, from a server on this
computer to a result page: three passes and an offer to submit, or a trial
of one pass that is no benchmark and is never submitted.

A separate package beside `victors_werkbank`, so it lies outside the harness
digest: its texts, the usual ports and the prompts change without changing
the Werkbank version. It uses the
standard library and `victors_werkbank`, `werkbank_result` and
`werkbank_submit` in process, and runs each pass as a `werkbank run` child
process, so a pass keeps the command line's rules.
"""

from .cli import main

__all__ = ["main"]
