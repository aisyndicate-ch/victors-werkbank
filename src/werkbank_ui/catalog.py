"""The run and evaluation directories under the viewer's roots.

A run directory holds `plan.json`, an evaluation directory `evaluation.json`;
the scan stops at either and never descends into them. Links are never
followed, hidden directories are skipped, and the depth is bounded. An entry is
addressed by an opaque key, never by a path from a request."""

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

MAX_DEPTH = 8
RUN = "run"
EVALUATION = "evaluation"
MARKERS = (("plan.json", RUN), ("evaluation.json", EVALUATION))


@dataclass(frozen=True)
class Entry:
    key: str
    kind: str
    path: Path
    label: str


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _kind(path: Path) -> str | None:
    for name, kind in MARKERS:
        if _regular(path / name):
            return kind
    return None


def scan(roots: list[Path]) -> dict[str, Entry]:
    """Every run and evaluation directory under `roots`, by key, in path order."""
    found: dict[str, Entry] = {}
    for index, root in enumerate(roots):
        pending = [(root, 0)]
        while pending:
            directory, depth = pending.pop()
            kind = _kind(directory)
            if kind is not None:
                relative = directory.relative_to(root).as_posix()
                label = root.name if relative == "." else f"{root.name}/{relative}"
                key = hashlib.sha256(f"{index}\0{relative}".encode()).hexdigest()[:16]
                found[key] = Entry(key, kind, directory, label)
                continue
            if depth == MAX_DEPTH:
                continue
            try:
                with os.scandir(directory) as children:
                    names = sorted(
                        child.name
                        for child in children
                        if child.is_dir(follow_symlinks=False)
                        and not child.name.startswith(".")
                    )
            except OSError:
                continue
            pending.extend((directory / name, depth + 1) for name in reversed(names))
    return dict(sorted(found.items(), key=lambda item: item[1].label))
