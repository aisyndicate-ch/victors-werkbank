"""`werkbank submit`: send an evaluation's results once to the Werkbank
maintainer, after the tester has seen exactly what is sent and confirmed it.

A separate package beside `victors_werkbank`, so it lies outside the harness
digest: the receiver address, the deletion contact, the texts and fixes to the
client never change the Werkbank version that results are compared by. It uses
the standard library and the validation of `victors_werkbank` in process, and
contacts only the receiver address fixed here, with one request, after the
tester's explicit confirmation at a terminal (`docs/SUBMISSION.md`).
"""

from .flow import submit

__all__ = ["submit"]
