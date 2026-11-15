"""The result page of an evaluation (`werkbank result`): one self-contained,
static HTML page per evaluation, the tester's summary of what `werkbank
compare` measured.

A separate package beside `victors_werkbank`, so it lies outside the harness
digest: a change to the page's layout or wording never changes the Werkbank
version. Every number it shows is computed by the package's versioned code
(`victors_werkbank.results.outcomes`) or read from the checked evaluation
record; the page only displays them. `result.json` names the renderer by a
digest of this package's files. Standard library only, no JavaScript.
"""

from .page import render
from .writer import ResultError, check, write

__all__ = ["ResultError", "check", "render", "write"]
