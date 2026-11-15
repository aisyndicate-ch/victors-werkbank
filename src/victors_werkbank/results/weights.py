"""Weight profiles for rescoring an evaluation (`werkbank compare`).

A weight profile replaces only the category weights of the overall points;
category values, bounds and their bootstrap draws never depend on it, and no
run changes. The standard profile is the design profile's `weights_percent`,
read from `profiles/core-v0.3.json` and never copied. Reviewed profiles are
package files under `weight_profiles/`: they are evaluation-only data outside
the harness digest, so a profile alone never splits pools. Own profiles
(`--weight-profile FILE`) are unreviewed and are recorded by ID and digest,
never by path.
"""

import os
import re
import stat
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..agent.transport import strict_json
from ..core_profile import design_profile, design_profile_sha256
from ..grading.evidence import sha256
from ..serving import MAX_TEXT, is_local_path

WEIGHT_FORMAT = "werkbank-weight-profile-v2"
STANDARD = "standard"
STANDARD_TITLE = "Standard weights of the design profile"
DIRECTORY = "weight_profiles"
MAX_BYTES = 16 * 1024
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
REVIEW_KEYS = {"pull_request", "maintainer_decision"}
DECISION_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
INVALID = "Invalid weight profile"


class WeightError(ValueError):
    """A weight profile file that this build does not admit."""


def _is_text(value: object) -> bool:
    # Printable ASCII, so no Unicode database version decides a title.
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_TEXT
        and all(" " <= ch <= "~" for ch in value)
    )


def categories() -> list[str]:
    """The design profile's categories in their order."""
    return list(design_profile()["weights_percent"])


def check_weights(value: object) -> dict[str, int]:
    """Exactly the design profile's categories, each an integer of at least
    one, summing to 100; returned in the design profile's order, so the key
    order of a file never matters."""
    names = categories()
    if (
        not isinstance(value, dict)
        or set(value) != set(names)
        or not all(type(value[k]) is int and 1 <= value[k] <= 100 for k in names)
        or sum(value.values()) != 100
    ):
        raise WeightError(INVALID)
    return {name: value[name] for name in names}


def _check_review(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != REVIEW_KEYS
        or type(value["pull_request"]) is not int
        or value["pull_request"] < 1
        or not isinstance(value["maintainer_decision"], str)
        or DECISION_DATE.fullmatch(value["maintainer_decision"]) is None
    ):
        raise WeightError(INVALID)


def parse(data: bytes, *, reviewed: bool) -> dict[str, Any]:
    """A weight profile entry of the evaluation record from a file's bytes.
    A reviewed (packaged) profile carries its review; an own one must not."""
    try:
        value = strict_json(data.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise WeightError(INVALID) from None
    keys = {"format", "id", "title", "weights_percent"}
    if reviewed:
        keys.add("review")
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["format"] != WEIGHT_FORMAT
        or not isinstance(value["id"], str)
        or IDENTIFIER.fullmatch(value["id"]) is None
        or value["id"] == STANDARD
        or not _is_text(value["title"])
        or is_local_path(value["title"])
    ):
        raise WeightError(INVALID)
    if reviewed:
        _check_review(value["review"])
    return {
        "id": value["id"],
        "title": value["title"],
        "reviewed": reviewed,
        "sha256": sha256(data),
        "weights_percent": check_weights(value["weights_percent"]),
    }


def standard() -> dict[str, Any]:
    return {
        "id": STANDARD,
        "title": STANDARD_TITLE,
        "reviewed": True,
        "sha256": design_profile_sha256(),
        "weights_percent": check_weights(design_profile()["weights_percent"]),
    }


def packaged() -> list[dict[str, Any]]:
    """The reviewed profiles shipped with this build, by ID; none when the
    package has no `weight_profiles/` directory."""
    root = files("victors_werkbank").joinpath(DIRECTORY)
    if not root.is_dir():
        return []
    found = [
        parse(entry.read_bytes(), reviewed=True)
        for entry in root.iterdir()
        if entry.name.endswith(".json")
    ]
    return sorted(found, key=lambda item: item["id"])


def read_own(path: Path) -> bytes:
    """A bounded regular file, never followed through a link; errors never
    name the path or the contents."""
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise WeightError(INVALID) from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WeightError(INVALID)
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_BYTES:
            chunk = os.read(descriptor, MAX_BYTES + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    except OSError:
        raise WeightError(INVALID) from None
    finally:
        os.close(descriptor)
    if total > MAX_BYTES:
        raise WeightError(INVALID)
    return b"".join(chunks)


def profiles(own: list[bytes] | None = None) -> list[dict[str, Any]]:
    """Every profile an evaluation applies: the standard one, the reviewed
    ones by ID, then the own ones by ID. A file given twice or an ID used
    twice is refused, so the argument order never changes the evaluation."""
    reviewed = packaged()
    mine = [parse(data, reviewed=False) for data in own or []]
    if len({item["sha256"] for item in mine}) != len(mine):
        raise WeightError("A weight profile is given more than once")
    ids = [STANDARD, *(item["id"] for item in reviewed + mine)]
    if len(set(ids)) != len(ids):
        raise WeightError("Weight profile IDs must be unique")
    return [standard(), *reviewed, *sorted(mine, key=lambda item: item["id"])]


def check_entries(value: object) -> list[dict[str, Any]]:
    """The recorded profiles of an evaluation, as `profiles` writes them."""
    if not isinstance(value, list) or not value:
        raise WeightError(INVALID)
    seen = set()
    for index, item in enumerate(value):
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "title", "reviewed", "sha256", "weights_percent"}
            or not isinstance(item["id"], str)
            or item["id"] in seen
            or type(item["reviewed"]) is not bool
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            or not _is_text(item["title"])
            or list(item["weights_percent"]) != categories()
        ):
            raise WeightError(INVALID)
        check_weights(item["weights_percent"])
        if (index == 0) != (item["id"] == STANDARD):
            raise WeightError(INVALID)
        if item["id"] != STANDARD and IDENTIFIER.fullmatch(item["id"]) is None:
            raise WeightError(INVALID)
        seen.add(item["id"])
    return value
