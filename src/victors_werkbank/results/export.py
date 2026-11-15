"""Portable export of a validated run: the inputs a regrade needs, nothing else.

`run.tar` holds `plan.json`, `run.json`, the task packages and every episode's
evidence and record, in sorted order with fixed metadata (mode 0644 or 0755,
owner 0, time 0), so the same run always exports the same bytes. Harbor's
trial files (which hold absolute host paths) and the readable report are not
exported. `export.json` lists every file's size and digest and the archive's.
Extract the archive into a new directory and `werkbank regrade` it there; an
imported archive is not otherwise admitted (no hostile-archive claim).
"""

import io
import tarfile
from pathlib import Path
from typing import Any

from ..grading.evidence import encode, sha256, write_exclusive
from .run import EPISODES_DIR, PLAN, RUN, RunView, episode_files, read_run

EXPORT_FORMAT = "werkbank-export-v3"
ARCHIVE = "run.tar"
INDEX = "export.json"


def _files(view: RunView) -> list[Path]:
    """Exactly the regrade inputs: the plan, the run record, the task packages
    (bound by their digests) and each episode's admitted files; nothing else
    under the run is ever archived."""
    root = view.root
    paths = [root / PLAN, root / RUN]
    for row in view.plan["episodes"]:
        paths.extend(p for p in (root / row["task"]).rglob("*") if p.is_file())
    for episode in view.episodes:
        directory = root / EPISODES_DIR / episode.row["name"]
        paths.extend(
            directory / name
            for name in episode_files(root, episode.record, episode.row)
        )
    return sorted(paths, key=lambda p: str(p.relative_to(root)))


class ExportError(ValueError):
    """The export refuses its arguments or the run's files; the text names no
    path."""


def export(source: Path, output: Path) -> dict[str, Any]:
    """Export the validated run `source` into the new directory `output`;
    returns the index record."""
    view = read_run(source)
    if output.exists() or output.is_symlink():
        raise ExportError(
            "Output must be a new directory; originals are never overwritten"
        )
    try:
        real_source, real_output = source.resolve(), output.resolve()
    except (OSError, RuntimeError):
        raise ExportError("Source or output path cannot be resolved") from None
    # Writing inside the run would change what was just validated.
    if real_output.is_relative_to(real_source) or real_source.is_relative_to(
        real_output
    ):
        raise ExportError("Output and source must not contain each other")
    files = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        directories: set[str] = set()
        for path in _files(view):
            if path.is_symlink():
                raise ExportError("A run file is a link")
            name = str(path.relative_to(source))
            data = path.read_bytes()
            parents = Path(name).parents
            for parent in reversed(list(parents)[:-1]):
                if str(parent) not in directories:
                    directories.add(str(parent))
                    info = tarfile.TarInfo(str(parent))
                    info.type, info.mode = tarfile.DIRTYPE, 0o755
                    archive.addfile(info)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if path.stat().st_mode & 0o100 else 0o644
            archive.addfile(info, io.BytesIO(data))
            files.append({"path": name, "bytes": len(data), "sha256": sha256(data)})
    data = buffer.getvalue()
    record = {
        "format": EXPORT_FORMAT,
        "source_run_id": view.plan["run_id"],
        "run_contract_version": view.plan["run_contract_version"],
        "archive": ARCHIVE,
        "archive_bytes": len(data),
        "archive_sha256": sha256(data),
        "files": files,
    }
    output.mkdir(mode=0o700, parents=True)
    write_exclusive(output / ARCHIVE, data)
    write_exclusive(output / INDEX, encode(record))
    return record
