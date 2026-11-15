"""Execution-environment provenance (`werkbank-environment-v2`): the host's
Python and platform, and the Python the code cases' container preflight
reports; unknown parts are null with a reason."""

import platform
from typing import Any, Callable

ENVIRONMENT_FORMAT = "werkbank-environment-v2"
MAX_TEXT = 256
INVALID = "Invalid environment provenance"


def _is_text(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_TEXT and value.isprintable()


def _clean(value: str) -> str:
    return "".join(c if c.isprintable() else "?" for c in value)[:MAX_TEXT]


def _absent(reason: str) -> dict[str, Any]:
    return {"version": None, "reason": reason}


def _sandbox_python(version: str | None) -> dict[str, Any]:
    if not version:
        return _absent("not probed")
    if not _is_text(version):
        return _absent("invalid sandbox python version")
    return {"version": version, "reason": None}


def collect_environment(sandbox_python: str | None = None) -> dict[str, Any]:
    """Record versions only; no paths, user names, host names or variables."""
    return {
        "format": ENVIRONMENT_FORMAT,
        "runner": {
            "python_version": _clean(platform.python_version()),
            "implementation": _clean(platform.python_implementation()),
        },
        "platform": {
            "system": _clean(platform.system()),
            "release": _clean(platform.release()),
            "machine": _clean(platform.machine()),
        },
        "sandbox_python": _sandbox_python(sandbox_python),
    }


def with_sandbox_python(
    environment: dict[str, Any], preflights: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run environment plus the sandbox Python reported by journaled preflights."""
    versions = {
        preflight.get("python")
        for preflight in preflights
        if isinstance(preflight, dict)
    }
    version = versions.pop() if len(versions) == 1 else None
    if preflights and (len(versions) > 0 or not isinstance(version, str)):
        section = _absent("differing or invalid preflights")
    elif preflights:
        section = _sandbox_python(version)
    else:
        section = _absent("not probed")
    return {**environment, "sandbox_python": section}


def _section(value: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(INVALID)
    return value


def _check(condition: bool) -> None:
    if not condition:
        raise ValueError(INVALID)


def _check_reason(present: bool, reason: object) -> None:
    _check(reason is None if present else _is_text(reason))


def _check_version(section: object, valid: Callable[[object], bool]) -> None:
    fields = _section(section, {"version", "reason"})
    present = fields["version"] is not None
    _check(not present or valid(fields["version"]))
    _check_reason(present, fields["reason"])


def validate_environment(value: object) -> None:
    """Raise ValueError unless value has exactly the collect_environment shape."""
    fields = _section(
        value,
        {"format", "runner", "platform", "sandbox_python"},
    )
    _check(fields["format"] == ENVIRONMENT_FORMAT)
    for key, names in (
        ("runner", {"python_version", "implementation"}),
        ("platform", {"system", "release", "machine"}),
    ):
        section = _section(fields[key], names)
        _check(
            all(
                isinstance(item, str) and len(item) <= MAX_TEXT and item.isprintable()
                for item in section.values()
            )
        )
    _check_version(fields["sandbox_python"], _is_text)


# Fixed reasons `collect_environment` writes for each absent component.
COLLECTED_REASONS = {
    "sandbox_python": frozenset({"not probed", "invalid sandbox python version"}),
}


def validate_collected(value: Any) -> None:
    """Raise ValueError unless `collect_environment` can write value, with its
    fixed reasons."""
    validate_environment(value)
    for key, reasons in COLLECTED_REASONS.items():
        reason = value[key]["reason"]
        _check(reason is None or reason in reasons)
