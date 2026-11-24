"""Validation of an evaluation directory (`werkbank validate EVAL`).

Offline and standard library only, so a wheel without Harbor or `jsonschema`
validates. Every evaluation is checked for its format and partial-credit
method, the shape schema 10.0.0 describes (`SHAPE` mirrors `schemas/evaluation.schema.json`; tests
hold the two together), exactly its files with their digests, and internal
consistency. Only the build that wrote it (`evaluator.harness_sha256`,
with the same measured context needs and weight profiles, which lie outside
the harness digest) re-renders its files and, with its source runs,
recomputes it: only that is a complete check. Like run validation it checks
completeness and consistency, not authenticity.
"""

import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

from ..agent.episode import harness_digest
from ..agent.loop import NO_PROGRESS_RESPONSES
from ..agent.measurements import (
    IDENTITY_SAMPLING_KEYS,
    MIN_REPETITIONS,
    validate_sampling,
)
from ..core_profile import design_profile_sha256
from ..domains.variants import VALUE
from ..grading.evidence import (
    EvidenceError,
    canonical,
    encode,
    read_bounded,
    sha256,
    strict_record,
)
from ..registry import module_of, planned_counts, scored_set
from . import comparison, partial, weights
from .evaluation import (
    BUDGET_POINTS,
    CLASSES,
    EVALUATION,
    EVALUATION_FORMAT,
    PROTOCOL_CLASSES,
    UNEVALUATED,
    below_recommendation,
    case_sets_sha256,
    context_needs_sha256,
    evaluate,
    input_digests,
    read_inputs,
)
from .run import RecordError

MAX_EVALUATION_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024

_PATTERNS = {
    "hex": re.compile(r"[0-9a-f]{64}"),
    "case": re.compile(r"[A-Z]{1,2}[0-9]{2}"),
    "run": re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
    "pool": re.compile(r"[0-9a-f]{64}"),
    "profile": re.compile(r"standard|[a-z0-9][a-z0-9-]{0,62}"),
    "set": re.compile(r"[a-z0-9][a-z0-9-]{0,62}"),
    "revision": re.compile(r"[0-9a-f]{40}|sha256:[0-9a-f]{64}"),
    "image": re.compile(r"sha256:[0-9a-f]{64}"),
    "outcome": re.compile(r"[a-z_]+/[A-Za-z0-9_]+"),
    "tool": re.compile(r"[A-Za-z0-9_-]{1,64}"),
    "instance": VALUE,
}


def _number(value: Any, low: float | None = None, high: float | None = None) -> bool:
    # An integer beyond the float range is no finite JSON number either.
    if type(value) is int:
        value = float(value) if abs(value) <= sys.float_info.max else math.inf
    return (
        type(value) is float
        and math.isfinite(value)
        and (low is None or value >= low)
        and (high is None or value <= high)
    )


def _integer(value: Any, low: int = 0, high: int | None = None) -> bool:
    return type(value) is int and value >= low and (high is None or value <= high)


def _pattern(name: str) -> Any:
    return lambda value: (
        isinstance(value, str) and _PATTERNS[name].fullmatch(value) is not None
    )


SCALARS: dict[str, Any] = {
    "string": lambda value: isinstance(value, str),
    "text": lambda value: isinstance(value, str) and 0 < len(value),
    "short": lambda value: isinstance(value, str) and 0 < len(value) <= 256,
    "bool": lambda value: type(value) is bool,
    "count": _integer,
    "positive": lambda value: _integer(value, 1),
    "weight": lambda value: _integer(value, 1, 100),
    "number": _number,
    "seconds": lambda value: _number(value, 0),
    "share": lambda value: _number(value, 0, 1),
    "points": lambda value: _number(value, 0, 100),
    "difference": lambda value: _number(value, -1, 1),
    "points_difference": lambda value: _number(value, -100, 100),
    "hex": _pattern("hex"),
    "case": _pattern("case"),
    "run": _pattern("run"),
    "pool": _pattern("pool"),
    "profile": _pattern("profile"),
    "set": _pattern("set"),
    "revision": _pattern("revision"),
    "image": _pattern("image"),
    "outcome": _pattern("outcome"),
    "tool": _pattern("tool"),
    "instance": _pattern("instance"),
}
# The weighted categories, then the modules: categories without weight
# whose pools stand outside every score.
WEIGHTED = (
    "agentic",
    "security",
    "math",
    "coding",
    "retrieval",
    "reliability",
    "data",
    "multi-turn",
)
MODULES = ("german",)
CATEGORIES = WEIGHTED + MODULES
STATUSES = (
    "passed",
    "failed",
    "timeout",
    "output_limit",
    "invalid",
    "unsupported",
    "not_run",
)
FLAGS = (
    "invalid_result",
    "too_few_repetitions",
    "given_instance",
    "below_recommendation",
)


def null(node: Any) -> tuple[str, Any]:
    return ("null", node)


def items(node: Any, minimum: int = 0) -> tuple[str, Any, int]:
    return ("list", node, minimum)


def unique(node: Any, minimum: int = 0) -> tuple[str, Any, int]:
    """A list without repeated items, as the schema's `uniqueItems`."""
    return ("unique", node, minimum)


def per_category(node: Any) -> tuple[str, Any]:
    return ("map", node)


def enum(*values: Any) -> tuple[str, tuple[Any, ...]]:
    return ("enum", values)


INTERVAL = null(("pair",))
EFFICIENCY = {
    "attempts": "count",
    "solved": "count",
    "seconds": "seconds",
    "output_tokens": "count",
    "requests": "count",
    "seconds_per_solved": null("seconds"),
    "output_tokens_per_solved": null("seconds"),
    "requests_per_solved": null("seconds"),
    "prompt_tokens": "count",
    "prompt_tokens_complete": "bool",
    "timing": {
        "model_seconds": "number",
        "tool_seconds": "number",
        "verifier_seconds": "number",
        "other_seconds": "number",
    },
    "censored": {
        "attempts": "count",
        "seconds": "seconds",
        "output_tokens": "count",
        "requests": "count",
    },
}
CURVE = {
    "attempts": "count",
    "solved_within": ("fixed", "count", len(BUDGET_POINTS)),
    "output_limit_attempts": "count",
    "output_limit_share": null("share"),
    "output_limit_reasons": ("names", "positive"),
    "largest_solved_response": null("count"),
    "possibly_binding": "bool",
}
FAILURES = {
    "classes": {name: "count" for name in CLASSES},
    "outcomes": ("outcomes", "positive"),
}
PROTOCOL = {"episodes": "count", **{name: "count" for name in PROTOCOL_CLASSES}}
CATEGORY_SCORE = {
    "weight_percent": "count",
    "planned": "positive",
    "cases": items("case"),
    "measured_families": "count",
    "measured_attempts": "count",
    "passed_attempts": "count",
    "coverage": "share",
    "lower": "share",
    "upper": "share",
    "score": null("share"),
    "rate": null("share"),
    "interval": INTERVAL,
}
POINTS = {
    "lower": "points",
    "upper": "points",
    "score": null("points"),
    "interval": INTERVAL,
}
PARTIAL = {
    "method": ("const", partial.METHOD),
    "categories": per_category(
        {
            "measured_attempts": "count",
            "censored_attempts": "count",
            "lower": "share",
            "upper": "share",
            "score": null("share"),
            "rate": null("share"),
            "fulfilment": null("share"),
        }
    ),
    "overall": {
        "lower": null("points"),
        "upper": null("points"),
        "score": null("points"),
        "fulfilment": null("share"),
    },
}
CONFIGURATION = {
    "sampling": ("sampling",),
    "max_tokens_per_call": ("range", 1, 65536),
    "harness_sha256": "hex",
    "streaming": "bool",
}
# A ranking's and a Pareto view's grouped run configuration, in the pools'
# value shapes.
GROUP = {key: CONFIGURATION[key] for key in comparison.GROUP_KEYS}
HARDWARE = {"source": enum(*comparison.HARDWARE_SOURCES), "text": "text"}
POOL = {
    "key": "pool",
    "identity_complete": "bool",
    "hardware": null(HARDWARE),
    "model_id": "text",
    "serving": {
        "serving_runtime": null("short"),
        "serving_version": null("short"),
        "context_profile": null("positive"),
        "model_revision": null("revision"),
        "quantization": null("short"),
        "kv_cache_dtype": null("short"),
        "speculative_decoding": null("short"),
        "accelerator": null("short"),
        "accelerator_memory_gib": null("positive"),
        "memory_type": null(enum("discrete", "unified")),
    },
    "configuration": CONFIGURATION,
    "image": "image",
    "runs": unique("run", 1),
    "seeds": items(null("count")),
    "instances": items("instance", 1),
    "repetitions": "positive",
    "case_ids": unique("case", 1),
    "case_set": null("set"),
    "module": null(enum(*MODULES)),
    "flags": unique(enum(*FLAGS)),
    "recommendation": {
        "max_tokens_per_call": "positive",
        "context_tokens": "positive",
        "reported_context_tokens": null("positive"),
        "cases_needing_more_context": unique("case"),
    },
    "evaluable": "bool",
    "scores": null(
        {
            "case_set": null("set"),
            "categories": per_category(CATEGORY_SCORE),
            "overall": {"coverage": "share", "points": POINTS},
        }
    ),
    "rescores": null(items({"profile": "profile", **POINTS}, 1)),
    "efficiency": null({"categories": per_category(EFFICIENCY), "overall": EFFICIENCY}),
    "budget_curve": null({"categories": per_category(CURVE), "overall": CURVE}),
    "partial_credit": null(PARTIAL),
    "failures": {"categories": per_category(FAILURES), "overall": FAILURES},
    "protocol_errors": {"categories": per_category(PROTOCOL), "overall": PROTOCOL},
    "reasoning": {
        "responses": "count",
        "with_reasoning": "count",
        "runs": items({"responses": "count", "with_reasoning": "count"}, 1),
    },
    "security": (
        "cases",
        {
            "runs": "positive",
            "runs_with_violation": "count",
            "forbidden_attempts": "count",
            "harmful_effects": "count",
            "benign_overrefusals": "count",
        },
    ),
    "cases": items(
        {
            "case_id": "case",
            "category": enum(*CATEGORIES),
            "outcomes": items(
                {
                    "run_id": "run",
                    "status": enum(*STATUSES),
                    "reason": "text",
                    "elapsed_seconds": "seconds",
                    "output_tokens": "count",
                    "observed_security_violation": null("bool"),
                    "checks_passed": null("count"),
                    "checks_counted": null("positive"),
                    "no_progress": items(
                        {
                            "branch_id": "text",
                            "response": "positive",
                            "tools": unique("tool"),
                            "unoffered": "bool",
                        }
                    ),
                },
                1,
            ),
            "runs": "positive",
            "passed": "count",
            "censored": "count",
            "efficiency": null(EFFICIENCY),
            "partial_credit": null("share"),
            "checks": items(
                {
                    "branch_id": "text",
                    "check": "text",
                    "precondition": "bool",
                    "attempts": "positive",
                    "recorded": "count",
                    "passed": "count",
                }
            ),
            "subgoals": items(
                {
                    "branch_id": "text",
                    "attempts": "positive",
                    "goals": "positive",
                    "reached": "count",
                    "boundaries": "count",
                    "held": "count",
                    "broken": "count",
                }
            ),
            "uncensored_attempts": "count",
            "uncensored_passed": "count",
            "nondeterministic": "bool",
        }
    ),
}
# A module's pool is never ranked or placed: rankings and the Pareto view
# name weighted categories alone.
CATEGORY_OR_OVERALL = enum(*WEIGHTED, "overall")
# What `comparison.paired` gives a pair and a difference.
PAIRED = {
    "categories": per_category(
        {
            "families": "positive",
            "paired_families": "count",
            "unpaired_families": "count",
            "delta": null("difference"),
            "interval": INTERVAL,
            "share_above_zero": null("share"),
            "first_better": "count",
            "second_better": "count",
            "ties": "count",
            "first_censored": "count",
            "second_censored": "count",
            "cases": items(
                {
                    "case_id": "case",
                    "first_rate": "share",
                    "second_rate": "share",
                    "difference": "difference",
                }
            ),
        }
    ),
    "overall": {
        "complete": "bool",
        "points": null("points_difference"),
        "interval": INTERVAL,
        "share_above_zero": null("share"),
        "first_better": "count",
        "second_better": "count",
        "ties": "count",
    },
}
SHAPE = {
    "format": ("const", EVALUATION_FORMAT),
    "comparison_eligible": ("const", False),
    "evaluator": {
        "harness_sha256": "hex",
        "context_needs_sha256": "hex",
        "case_sets_sha256": "hex",
        "version": "text",
    },
    "profile": {
        "id": "string",
        "design_profile_sha256": "hex",
        "weights_percent": per_category("count"),
        "planned_counts": per_category("count"),
    },
    "parameters": {
        "resamples": ("range", 100, 100_000),
        "bootstrap_seed": ("range", 0, 4_294_967_295),
        "interval_percent": ("const", 95),
        "min_families": ("const", 2),
        "min_repetitions": ("const", MIN_REPETITIONS),
        "budget_points": ("const", list(BUDGET_POINTS)),
        "case_set": null({"name": "set", "case_ids": unique("case", 1)}),
    },
    "weight_profiles": items(
        {
            "id": "profile",
            "title": "short",
            "reviewed": "bool",
            "sha256": "hex",
            "weights_percent": {name: "weight" for name in WEIGHTED},
        },
        1,
    ),
    "inputs": items(
        {"run_id": "run", "plan_sha256": "hex", "run_sha256": "hex", "pool": "pool"},
        1,
    ),
    "pools": items(POOL, 1),
    "separability": items(
        {
            "configuration": GROUP,
            "case_ids": items("case", 1),
            "category": CATEGORY_OR_OVERALL,
            "ranking": items(
                {"pool": "pool", "value": "number", "interval": ("pair",)}
            ),
            "not_separable": items(("fixed", "pool", 2)),
            "not_ranked": items("pool"),
        }
    ),
    "differences": items(
        {
            "configuration": GROUP,
            "case_ids": items("case", 1),
            "first": "pool",
            "second": "pool",
            **PAIRED,
        }
    ),
    "comparisons": items(
        {
            "first": "hex",
            "second": "hex",
            "kind": enum(*comparison.KINDS),
            "differs": unique(enum(*comparison.PAIR_FIELDS)),
            "unknown": unique(enum(*comparison.UNKNOWN_FIELDS)),
            "case_ids": unique("case", 1),
            "first_flags": unique(enum(*FLAGS)),
            "second_flags": unique(enum(*FLAGS)),
            **PAIRED,
        }
    ),
    "pareto": items(
        {
            "hardware": HARDWARE,
            "configuration": GROUP,
            "case_ids": items("case", 1),
            "category": CATEGORY_OR_OVERALL,
            "value": enum("rate", "lower_points"),
            "placed": items(
                {
                    "pool": "pool",
                    "seconds_per_solved": "seconds",
                    "value": "seconds",
                    "interval": INTERVAL,
                    "seconds_per_attempt": "seconds",
                    "front": "bool",
                    "dominated_by": items({"pool": "pool", "separable": null("bool")}),
                }
            ),
            "not_placed": items(
                {
                    "pool": "pool",
                    "reason": enum(
                        "not_evaluable", "incomplete_coverage", "no_solution"
                    ),
                }
            ),
        }
    ),
    "limitations": items("text", 1),
    "files": ("files",),
}


def _fail() -> None:
    raise RecordError("Evaluation record does not have its documented shape")


# An evaluation of another format is never read: an earlier one pooled and
# evaluated by other rules, so its values would read as this build's.
# `werkbank compare` writes it again from its runs in this build's format.
OTHER_FORMAT = (
    f"The evaluation has another format than this build's {EVALUATION_FORMAT};"
    " write it again from its runs with werkbank compare"
)
# Nor is partial credit of another rule set: its values would read as this
# build's.
OTHER_METHOD = (
    "The evaluation's partial credit follows another method than this build's"
    f" {partial.METHOD}; write it again from its runs with werkbank compare"
)


def _other_method(record: dict[str, Any]) -> bool:
    """Whether a pool's partial credit names another method than this build's."""
    pools = record.get("pools")
    return isinstance(pools, list) and any(
        isinstance(entry, dict)
        and isinstance(entry.get("partial_credit"), dict)
        and entry["partial_credit"].get("method") != partial.METHOD
        for entry in pools
    )


def check_shape(value: Any, node: Any = SHAPE) -> None:
    """Raise RecordError unless `value` has the shape `node` describes."""
    if isinstance(node, dict):
        if not isinstance(value, dict) or set(value) != set(node):
            _fail()
        for key, sub in node.items():
            check_shape(value[key], sub)
        return
    if isinstance(node, str):
        if not SCALARS[node](value):
            _fail()
        return
    tag = node[0]
    if tag == "null":
        if value is not None:
            check_shape(value, node[1])
    elif tag in {"list", "unique"}:
        if not isinstance(value, list) or len(value) < node[2]:
            _fail()
        for item in value:
            check_shape(item, node[1])
        if tag == "unique" and len({canonical(item) for item in value}) != len(value):
            _fail()
    elif tag == "fixed":
        if not isinstance(value, list) or len(value) != node[2]:
            _fail()
        for item in value:
            check_shape(item, node[1])
    elif tag == "pair":
        if not isinstance(value, list) or len(value) != 2:
            _fail()
        if not all(_number(item) for item in value):
            _fail()
    elif tag == "map":
        if not isinstance(value, dict) or not set(value) <= set(CATEGORIES):
            _fail()
        for item in value.values():
            check_shape(item, node[1])
    elif tag in {"names", "outcomes", "cases"}:
        keys = {"names": "string", "outcomes": "outcome", "cases": "case"}[tag]
        if not isinstance(value, dict) or not all(SCALARS[keys](k) for k in value):
            _fail()
        for item in value.values():
            check_shape(item, node[1])
    elif tag == "enum":
        if isinstance(value, bool) or value not in node[1]:
            _fail()
    elif tag == "const":
        if type(value) is not type(node[1]) or value != node[1]:
            _fail()
    elif tag == "range":
        if not _integer(value, node[1], node[2]):
            _fail()
    elif tag == "sampling":
        try:
            validate_sampling(value)
        except (ValueError, TypeError):
            _fail()
        if not set(value) <= set(IDENTITY_SAMPLING_KEYS):
            _fail()
    elif tag == "files":
        from .tables import outputs_names

        if not isinstance(value, dict) or list(value) != outputs_names():
            _fail()
        if not all(SCALARS["hex"](item) for item in value.values()):
            _fail()
    else:  # pragma: no cover - a SHAPE typo
        raise AssertionError(tag)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _check_reasoning(entry: dict[str, Any]) -> None:
    """One count per run, none above its responses, summing to the pool's."""
    reasoning = entry["reasoning"]
    runs = reasoning["runs"]
    if (
        len(runs) != len(entry["runs"])
        or any(run["with_reasoning"] > run["responses"] for run in runs)
        or sum(run["responses"] for run in runs) != reasoning["responses"]
        or sum(run["with_reasoning"] for run in runs) != reasoning["with_reasoning"]
    ):
        _inconsistent()


def check_consistency(record: dict[str, Any]) -> None:
    """References between the record's sections and the sums its overall
    values are made of."""
    pools = {entry["key"]: entry for entry in record["pools"]}
    if len(pools) != len(record["pools"]) or list(pools) != sorted(pools):
        _inconsistent()
    runs = [item["run_id"] for item in record["inputs"]]
    if runs != sorted(set(runs)):
        _inconsistent()
    for key, entry in pools.items():
        if sorted(entry["runs"]) != sorted(
            item["run_id"] for item in record["inputs"] if item["pool"] == key
        ) or entry["repetitions"] != len(entry["runs"]):
            _inconsistent()
        _check_instances(entry)
    if {item["pool"] for item in record["inputs"]} != set(pools):
        _inconsistent()
    try:
        profiles = weights.check_entries(record["weight_profiles"])
    except weights.WeightError:
        _inconsistent()
    standard = record["profile"]["weights_percent"]
    if profiles[0]["weights_percent"] != standard:
        _inconsistent()
    ids = [item["id"] for item in profiles]
    case_set = record["parameters"]["case_set"]
    for entry in pools.values():
        # Under a case set every pool is evaluated on exactly its cases.
        if case_set is not None and (
            set(entry["case_ids"]) != set(case_set["case_ids"])
            or len(entry["case_ids"]) != len(case_set["case_ids"])
            or entry["case_set"] != case_set["name"]
        ):
            _inconsistent()
        _check_flags(entry)
        _check_reasoning(entry)
        scores = entry["scores"]
        try:
            module = module_of(entry["case_ids"])
        except (KeyError, ValueError):
            _inconsistent()
        # Every evaluated section exists exactly for an evaluated pool; a
        # module's pool has no scores in any case.
        if entry["module"] != module or any(
            (entry[name] is None) == (entry["evaluable"] and module is None)
            for name in ("scores", "rescores")
        ):
            _inconsistent()
        if any(
            (entry[name] is None) == entry["evaluable"]
            for name in ("efficiency", "budget_curve", "partial_credit")
        ):
            _inconsistent()
        _check_cases(entry)
        if module is not None and entry["evaluable"]:
            _check_module_credit(entry, module)
        if scores is None:
            continue
        _check_partial(entry["partial_credit"], scores, standard)
        _check_fulfilment(entry, standard)
        if [item["profile"] for item in entry["rescores"]] != ids:
            _inconsistent()
        categories = scores["categories"]
        if list(categories) != list(standard):
            _inconsistent()
        # The planned counts are those of the scored case set the scores name,
        # or the profile's, and only a category with every planned case has a
        # score.
        planned = planned_counts(record["profile"], entry["case_ids"])
        if scores["case_set"] != scored_set(entry["case_ids"]) or any(
            categories[k]["planned"] != planned[k]
            or (
                categories[k]["score"] is not None
                and len(categories[k]["cases"]) != planned[k]
            )
            for k in standard
        ):
            _inconsistent()
        for item, rescore in zip(profiles, entry["rescores"], strict=True):
            for bound in ("lower", "upper"):
                total = sum(
                    item["weights_percent"][k] * categories[k][bound] for k in standard
                )
                if not _close(total, rescore[bound]):
                    _inconsistent()
        if canonical(entry["rescores"][0]) != canonical(
            {"profile": weights.STANDARD, **scores["overall"]["points"]}
        ):
            _inconsistent()
    # A module's pool is never paired, ranked or placed.
    scored = {key for key, entry in pools.items() if entry["module"] is None}
    for item in record["comparisons"]:
        if not {item["first"], item["second"]} <= scored or not (
            item["first"] < item["second"]
        ):
            _inconsistent()
        _check_paired(item)
    _check_differences(record, pools)
    for section, key in (("separability", "ranking"), ("pareto", "placed")):
        for item in record[section]:
            named = {row["pool"] for row in item[key]}
            named |= set(
                item["not_ranked"]
                if section == "separability"
                else [row["pool"] for row in item["not_placed"]]
            )
            if not named <= scored:
                _inconsistent()


def _check_paired(item: dict[str, Any]) -> None:
    """A pair's or a difference's counts add up: each category's case wins
    over its paired families, the overall wins over the categories, and a
    share above 0 exactly beside an interval."""
    rows = item["categories"].values()
    wins = ("first_better", "second_better", "ties")
    overall = item["overall"]
    if any(
        sum(row[key] for key in wins) != row["paired_families"]
        or row["paired_families"] + row["unpaired_families"] != row["families"]
        or (row["interval"] is None) != (row["share_above_zero"] is None)
        for row in rows
    ) or (overall["interval"] is None) != (overall["share_above_zero"] is None):
        _inconsistent()
    if any(overall[key] != sum(row[key] for row in rows) for key in wins):
        _inconsistent()


def _check_differences(record: dict[str, Any], pools: dict[str, Any]) -> None:
    """Every two evaluated pools of one ranking group have exactly one
    difference, in the group's order, the first ranked first, with overall
    points only on a scored case set."""
    expected = []
    for _, case_ids, members in comparison.groups(record["pools"]):
        ranked = sorted(
            (entry for entry in members if entry["scores"] is not None),
            key=comparison.rank_order,
        )
        expected += [
            (list(case_ids), first["key"], second["key"])
            for index, first in enumerate(ranked)
            for second in ranked[index + 1 :]
        ]
    found = [
        (item["case_ids"], item["first"], item["second"])
        for item in record["differences"]
    ]
    if found != expected:
        _inconsistent()
    for item in record["differences"]:
        first = pools[item["first"]]
        if item["configuration"] != comparison.group_configuration(first):
            _inconsistent()
        if scored_set(item["case_ids"]) is None and (
            item["overall"]["points"] is not None
        ):
            _inconsistent()
        _check_paired(item)


def _check_instances(entry: dict[str, Any]) -> None:
    """One seed and one instance value per run; the runs of an entry with a
    given instance value share that value. The repetition rule is asked of
    every pool, whatever its sampling, and never of such an entry."""
    runs = len(entry["runs"])
    if len(entry["seeds"]) != runs or len(entry["instances"]) != runs:
        _inconsistent()
    given = "given_instance" in entry["flags"]
    if given and len(set(entry["instances"])) != 1:
        _inconsistent()
    few = not given and runs < MIN_REPETITIONS
    if few != ("too_few_repetitions" in entry["flags"]):
        _inconsistent()


def _check_flags(entry: dict[str, Any]) -> None:
    """A pool is evaluated exactly without a flag that leaves it unevaluated;
    the cases needing more context are the pool's, none without a reported
    context under the recommended one (the largest need rounded up), and the
    hint flag stands exactly where the cap is below the recommended one or a
    case needs more context."""
    if entry["evaluable"] == bool(set(UNEVALUATED) & set(entry["flags"])):
        _inconsistent()
    item = entry["recommendation"]
    short = item["cases_needing_more_context"]
    context = item["reported_context_tokens"]
    if not set(short) <= set(entry["case_ids"]) or (
        short and (context is None or context >= item["context_tokens"])
    ):
        _inconsistent()
    below = below_recommendation(item, entry["configuration"]["max_tokens_per_call"])
    if below != ("below_recommendation" in entry["flags"]):
        _inconsistent()


def _check_cases(entry: dict[str, Any]) -> None:
    """Checks never exceed those counted, a pass passed every one, a
    no-progress stop belongs to a measured attempt that did not pass, at most
    one per episode, after at least the responses it needs and with a call it
    repeated, a case's partial credit exists exactly where its pool is
    evaluated and it has a measured attempt, and its subgoal rows stand for
    exactly the branches and attempts of its check rows, within the case's
    subgoals and boundaries."""
    for case in entry["cases"]:
        measured = False
        for outcome in case["outcomes"]:
            measured = measured or outcome["status"] != "not_run"
            stops = outcome["no_progress"]
            if stops and outcome["status"] in {"passed", "not_run"}:
                _inconsistent()
            if len({stop["branch_id"] for stop in stops}) != len(stops) or any(
                stop["response"] < NO_PROGRESS_RESPONSES
                or not (stop["tools"] or stop["unoffered"])
                for stop in stops
            ):
                _inconsistent()
            passed, counted = outcome["checks_passed"], outcome["checks_counted"]
            if (passed is None) != (counted is None) or (
                counted is not None and passed > counted
            ):
                _inconsistent()
            if outcome["status"] == "passed" and (counted is None or passed != counted):
                _inconsistent()
        if (case["partial_credit"] is None) == (entry["evaluable"] and measured):
            _inconsistent()
        for row in case["checks"]:
            if not row["passed"] <= row["recorded"] <= row["attempts"]:
                _inconsistent()
        _check_subgoals(case)


def _check_subgoals(case: dict[str, Any]) -> None:
    """A case's subgoal rows against its check rows and this build's table."""
    rows = case["subgoals"]
    counts = partial.subgoal_counts(case["case_id"])
    branches = [row["branch_id"] for row in rows]
    if (
        counts is None
        or len(set(branches)) != len(branches)
        or {(r["branch_id"], r["attempts"]) for r in rows}
        != {(r["branch_id"], r["attempts"]) for r in case["checks"]}
    ):
        _inconsistent()
    for row in rows:
        if (
            (row["goals"], row["boundaries"]) != counts
            or row["attempts"] > case["runs"]
            or row["reached"] > row["goals"] * row["attempts"]
            or row["held"] + row["broken"] > row["boundaries"] * row["attempts"]
        ):
            _inconsistent()


def _check_partial(
    credit: dict[str, Any], scores: dict[str, Any], standard: dict[str, int]
) -> None:
    """Partial credit per category within its bounds and never below the
    score's, a value only as the lower bound; overall the weighted sum."""
    categories = credit["categories"]
    if list(categories) != list(standard):
        _inconsistent()
    for name, item in categories.items():
        score = scores["categories"][name]
        if (
            item["lower"] > item["upper"] + 1e-9
            or item["lower"] < score["lower"] - 1e-9
            or item["upper"] < score["upper"] - 1e-9
            or item["score"] not in (None, item["lower"])
            or (item["score"] is None) != (score["score"] is None)
        ):
            _inconsistent()
    overall = credit["overall"]
    for bound in ("lower", "upper"):
        total = sum(standard[k] * categories[k][bound] for k in standard)
        if overall[bound] is None or not _close(total, overall[bound]):
            _inconsistent()
    complete = all(item["score"] is not None for item in categories.values())
    if overall["score"] != (overall["lower"] if complete else None):
        _inconsistent()


def _check_module_credit(entry: dict[str, Any], module: str) -> None:
    """A module's pool: its credit has the module's category alone, over the
    pool's own cases, and overall its subgoal fulfilment with no points."""
    credit = entry["partial_credit"]
    categories = credit["categories"]
    if list(categories) != [module]:
        _inconsistent()
    item, overall = categories[module], credit["overall"]
    if (
        item["lower"] > item["upper"] + 1e-9
        or item["score"] not in (None, item["lower"])
        or (overall["lower"], overall["upper"], overall["score"]) != (None, None, None)
        or overall["fulfilment"] != item["fulfilment"]
    ):
        _inconsistent()
    _check_fulfilment(entry, {module: 1})


def _check_fulfilment(entry: dict[str, Any], standard: dict[str, int]) -> None:
    """Each category's subgoal fulfilment the mean over the pool's cases of
    its credit over every attempt (an unmeasured one 0), null without a case;
    overall the categories' values with the standard weights."""
    credit = entry["partial_credit"]
    for category in standard:
        values = []
        for case in entry["cases"]:
            if case["category"] != category:
                continue
            measured = sum(o["status"] != "not_run" for o in case["outcomes"])
            values.append((case["partial_credit"] or 0.0) * measured / case["runs"])
        found = credit["categories"][category]["fulfilment"]
        if (found is None) != (not values) or (
            values and not _close(found, sum(values) / len(values))
        ):
            _inconsistent()
    expected = partial.weighted_fulfilment(standard, credit["categories"])
    found = credit["overall"]["fulfilment"]
    if (found is None) != (expected is None) or (
        expected is not None and not _close(found, expected)
    ):
        _inconsistent()


def _inconsistent() -> None:
    raise RecordError("Evaluation record is inconsistent")


def _files(root: Path) -> dict[str, bytes]:
    """Every regular file under an evaluation directory by relative path;
    RecordError for a link, any other kind of entry or another directory."""
    from .tables import DATA, DATASET

    found: dict[str, bytes] = {}
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in names:
            path = base / name
            if path.is_symlink() or path.relative_to(root).as_posix() not in {
                DATASET,
                DATA,
            }:
                raise RecordError("The evaluation directory differs from its file list")
        for name in files:
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise RecordError("The evaluation directory holds a link or device")
            try:
                data = read_bounded(path, MAX_FILE_BYTES)
            except EvidenceError as error:
                raise RecordError(str(error)) from None
            found[path.relative_to(root).as_posix()] = data
    return found


def this_build(record: dict[str, Any]) -> bool:
    """Whether this build wrote the evaluation: the same harness, the same
    measured context needs, case sets and profile, and the same weight
    profiles, the standard one and every reviewed one it ships (the needs,
    the sets, the profile and the weight profiles are data outside the
    harness digest, so an evaluation written before they changed is written
    again from the same runs)."""
    evaluator = record["evaluator"]
    if (
        evaluator["harness_sha256"] != harness_digest()
        or evaluator["context_needs_sha256"] != context_needs_sha256()
        or evaluator["case_sets_sha256"] != case_sets_sha256()
        or record["profile"]["design_profile_sha256"] != design_profile_sha256()
    ):
        return False
    applied = record["weight_profiles"]
    shipped = [item for item in applied[1:] if item["reviewed"]]
    return applied[0] == weights.standard() and shipped == weights.packaged()


def check_evaluation(
    root: Path, sources: list[Path] | None = None
) -> tuple[dict[str, Any], bool]:
    """Validate an evaluation directory; returns its record and whether the
    check was complete (this build wrote it and its sources recompute it).
    RecordError for anything invalid."""
    from .tables import outputs

    if root.is_symlink() or not root.is_dir():
        raise RecordError("An evaluation is a directory")
    try:
        data = read_bounded(root / EVALUATION, MAX_EVALUATION_BYTES)
        record = strict_record(data)
    except EvidenceError as error:
        raise RecordError(str(error)) from None
    if not isinstance(record, dict):
        _fail()
    found = record.get("format")
    if (
        isinstance(found, str)
        and found.startswith("werkbank-evaluation-")
        and found != EVALUATION_FORMAT
    ):
        raise RecordError(OTHER_FORMAT)
    if found == EVALUATION_FORMAT and _other_method(record):
        raise RecordError(OTHER_METHOD)
    check_shape(record)
    check_consistency(record)
    present = _files(root)
    if set(present) != {EVALUATION, *record["files"]}:
        raise RecordError("The evaluation directory differs from its file list")
    for name, digest in record["files"].items():
        if sha256(present[name]) != digest:
            raise RecordError("An evaluation file differs from its digest")
    body = {key: value for key, value in record.items() if key != "files"}
    same = this_build(record)
    if same:
        if encode(record) != data:
            raise RecordError("evaluation.json is not in its written form")
        try:
            rendered = outputs(body)
        except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
            # A record the checks above admit always renders; fail closed.
            raise RecordError("Evaluation record cannot be rendered") from None
        if list(rendered) != list(record["files"]) or any(
            rendered[name] != present[name] for name in rendered
        ):
            raise RecordError("An evaluation file differs from its rendering")
    if sources is None:
        return record, False
    views = read_inputs(sources)
    if [input_digests(view) for view in views] != [
        {key: item[key] for key in ("run_id", "plan_sha256", "run_sha256")}
        for item in record["inputs"]
    ]:
        raise RecordError("The sources are not the evaluation's runs")
    if not same:
        return record, False
    parameters = record["parameters"]
    case_set = parameters["case_set"]
    expected = evaluate(
        views,
        resamples=parameters["resamples"],
        seed=parameters["bootstrap_seed"],
        weight_profiles=record["weight_profiles"],
        case_set=None if case_set is None else case_set["name"],
    )
    # The version number is outside the harness digest: a patch release keeps
    # the build, and the evaluation keeps the number it was written with.
    expected["evaluator"]["version"] = record["evaluator"]["version"]
    if canonical(expected) != canonical(body):
        raise RecordError("The evaluation differs from its sources")
    return record, True
