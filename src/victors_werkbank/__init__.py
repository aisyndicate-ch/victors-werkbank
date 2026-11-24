"""Local benchmark infrastructure; no model calls at import time."""

from importlib.resources import files


def __getattr__(name: str) -> str:
    """`__version__`, read from the package's `VERSION` file when first asked
    for: the version number is data outside the harness digest, so a release
    that changes nothing a run does keeps the build. The verifier's copy of
    the harness has no such file and never asks."""
    if name == "__version__":
        return files(__name__).joinpath("VERSION").read_text(encoding="ascii").strip()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
