"""The installation ID: 16 random bytes in hexadecimal that tell one
installation's submissions apart without an account.

It is created in memory for the summary and stored only once the tester
confirms sending or writing the archive, in
`$XDG_CONFIG_HOME/werkbank/installation-id` (`~/.config/...` when the variable
is unset or not absolute), directory mode 0700, file mode 0600, one line. An
unreadable or malformed file stops the submission and is never replaced; an
existing file is never overwritten. The ID never enters a run, plan,
evaluation or export.
"""

import os
import re
import secrets
import stat
from pathlib import Path
from typing import Mapping

from victors_werkbank.grading.evidence import write_exclusive

DIRECTORY = "werkbank"
FILE = "installation-id"
_LINE = re.compile(rb"[0-9a-f]{32}\n")
_MAX_BYTES = 33


class InstallationError(Exception):
    """The stored installation ID cannot be used; the file stays as it is."""


def config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Where the installation ID is stored."""
    environ = os.environ if environ is None else environ
    base = environ.get("XDG_CONFIG_HOME", "")
    root = Path(base) if base and os.path.isabs(base) else Path.home() / ".config"
    return root / DIRECTORY / FILE


def _unusable(path: Path) -> InstallationError:
    return InstallationError(
        f"The installation ID file {path} is unreadable or malformed; it was not"
        " replaced. Delete it to start a new installation ID."
    )


def load(path: Path) -> str | None:
    """The stored ID, or None when no file exists. InstallationError for a
    link, a non-regular, unreadable or malformed file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        if os.path.lexists(path):
            raise _unusable(path) from None
        return None
    except OSError:
        raise _unusable(path) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
            raise _unusable(path)
        data = os.read(fd, _MAX_BYTES + 1)
    except OSError:
        raise _unusable(path) from None
    finally:
        os.close(fd)
    if _LINE.fullmatch(data) is None:
        raise _unusable(path)
    return data[:32].decode("ascii")


def new_id() -> str:
    return secrets.token_hex(16)


def store(path: Path, value: str) -> None:
    """Store `value` as a new file; never replaces one (FileExistsError)."""
    if _LINE.fullmatch(value.encode("ascii") + b"\n") is None:
        raise ValueError("An installation ID is 32 lowercase hexadecimal digits")
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(path.parent, 0o700)
    except FileExistsError:
        pass
    else:
        os.chmod(path.parent, 0o700)
    write_exclusive(path, value.encode("ascii") + b"\n")
