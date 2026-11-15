"""The `--thinking LEVEL` shorthand: a level resolved, before admission, into
the chat template variables `--chat-template-kwargs` would send.

The recipe table `thinking_recipes.json` names, per model family, the served
model names it applies to (case-insensitive patterns) and the variables of
each level its chat template supports, read from the family's template. It
changes the requests a run sends, so it is part of the harness digest. Only
the resolved variables are sent and recorded, exactly as if they had been
given with `--chat-template-kwargs`, so the setup identity never depends on
the shorthand. A served model name no recipe matches, or a level its recipe
lacks, is refused; nothing is guessed and nothing falls back.
"""

import fnmatch
import json
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

from .agent.measurements import validate_template_kwargs

RECIPES = "thinking_recipes.json"
FORMAT = "werkbank-thinking-recipes-v1"
# The level names `--thinking` takes; a family's recipe supports a subset.
LEVELS = ("off", "low", "medium", "high", "xhigh", "max")
_FAMILY_KEYS = {"family", "models", "source", "levels"}


class RecipeError(ValueError):
    """`--thinking` has no recipe for the served model or level; the text is
    fixed and names the recipes, never the model name."""


@dataclass(frozen=True)
class Family:
    """One model family: its name, the served model names it matches and the
    template variables of each level it supports, in `LEVELS` order."""

    name: str
    models: tuple[str, ...]
    levels: dict[str, dict[str, Any]]

    def matches(self, model: str) -> bool:
        return any(
            fnmatch.fnmatchcase(model.lower(), pattern) for pattern in self.models
        )


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate key")
    return dict(pairs)


def _family(value: object) -> Family:
    if not isinstance(value, dict) or set(value) != _FAMILY_KEYS:
        raise ValueError("Invalid thinking recipes")
    name, models, levels = value["family"], value["models"], value["levels"]
    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(value["source"], str)
        or not value["source"].strip()
        or not isinstance(models, list)
        or not models
        or not all(
            isinstance(pattern, str) and pattern and pattern == pattern.lower()
            for pattern in models
        )
        or len(set(models)) != len(models)
        or not isinstance(levels, dict)
        or not levels
        or list(levels) != [level for level in LEVELS if level in levels]
    ):
        raise ValueError("Invalid thinking recipes")
    for variables in levels.values():
        validate_template_kwargs(variables)
    return Family(name, tuple(models), dict(levels))


@cache
def recipes() -> tuple[Family, ...]:
    """The packaged recipe table, checked; ValueError for a malformed one."""
    return parse(files("victors_werkbank").joinpath(RECIPES).read_bytes())


def parse(data: bytes) -> tuple[Family, ...]:
    """A recipe table's families; ValueError unless it is strict JSON of this
    format, every family named once with its patterns and every level's
    variables within the template-variable rule."""
    try:
        value = json.loads(data, object_pairs_hook=_unique_keys)
    except (ValueError, UnicodeError):
        raise ValueError("Invalid thinking recipes") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "families"}
        or value["format"] != FORMAT
        or not isinstance(value["families"], list)
        or not value["families"]
    ):
        raise ValueError("Invalid thinking recipes")
    try:
        found = tuple(_family(item) for item in value["families"])
    except ValueError:
        raise ValueError("Invalid thinking recipes") from None
    if len({family.name for family in found}) != len(found):
        raise ValueError("Invalid thinking recipes")
    return found


def _families_text(families: tuple[Family, ...]) -> str:
    return "; ".join(
        f"{family.name}: {', '.join(family.levels)}" for family in families
    )


def resolve(model: str, level: str) -> dict[str, Any]:
    """The template variables `level` sends to the served model `model`, a
    fresh copy; RecipeError where no single recipe matches or it lacks the
    level (ValueError for a malformed table)."""
    families = recipes()
    if level not in LEVELS:
        raise RecipeError(f"--thinking takes one of {', '.join(LEVELS)}")
    found = [family for family in families if family.matches(model)]
    advice = "; set the template variables with --chat-template-kwargs instead"
    if len(found) != 1:
        raise RecipeError(
            "--thinking knows no recipe for this served model name"
            f" (recipes: {_families_text(families)}){advice}"
        )
    family = found[0]
    if level not in family.levels:
        raise RecipeError(
            f"--thinking {level} is not a level of the {family.name} recipe"
            f" (levels: {', '.join(family.levels)}){advice}"
        )
    return json.loads(json.dumps(family.levels[level]))
