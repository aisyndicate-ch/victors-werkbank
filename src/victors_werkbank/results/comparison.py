"""Groups, paired comparisons and the Pareto view of an evaluation.

Pools compare when they ran one build (harness digest) and one case set:
rankings group by these two alone, so the model, serving software,
quantization, hardware, sampling, context, per-response cap and streaming
mode are what a ranking compares, and every pool shows them. Seconds compare
only on one stated hardware: the Pareto view also groups by the hardware
statement (`hardware`), and a pool without one is in no Pareto view.

Two pools pair when their model weights (model revision and quantization)
are known and the same and they share the run configuration, the container
image, the ordered case list and the hardware statement; everything else
that differs is named, and the pair's kind is `engine`, `kv_cache` or
`mixed`, always `mixed` where a setting is unknown on either side. A pair's category delta
is the mean of the paired case families' rate differences, with a paired
cluster bootstrap over those differences; there is no test, no multiplicity
correction and no ranking. The Pareto view puts seconds per solved case
against the rate on measured cases (per category) or the points on the
measured case set (overall) and never combines them into one number. Every
value here is a budget-score value: censored attempts count as not solved,
and nothing is difficulty evidence.
"""

import json
from typing import Any

from ..registry import CASES, planned_counts
from ..serving import ENGINE_SETTINGS, MAX_TEXT, TEMPLATE_DEFAULTS
from .uncertainty import MIN_FAMILIES, bootstrap, interval

# Serving settings a pair may differ in, by their identity names; a pseudo
# field for the models listing and the engine's model path.
PAIR_FIELDS = (
    "serving_runtime",
    "serving_version",
    "engine_commit",
    "kv_cache_dtype",
    "speculative_decoding",
    "context_profile",
    "chat_template",
    TEMPLATE_DEFAULTS,
    "tool_call_parser",
    "driver_version",
    "compute_runtime",
    "serving_host_os",
    "model_listing",
)
# Settings that come with an engine: its version and commit, and the tool-call
# parser, whose names are engine-specific.
ENGINE_BOUND = frozenset({"serving_version", "engine_commit", "tool_call_parser"})
# Serving-host software and the models listing (served name, owning engine,
# model location): they may differ in any pair and are always named.
HOST = frozenset(
    {"driver_version", "compute_runtime", "serving_host_os", "model_listing"}
)
# Settings a pair's kind needs on both sides: every one but serving-host
# software, the listing, the engine commit, which release builds lack, and
# the server's default template variables, which only SGLang reports. Those
# defaults, where they differ, still make a pair `mixed`.
KNOWN_FIELDS = tuple(
    key
    for key in PAIR_FIELDS
    if key not in HOST and key not in {"engine_commit", TEMPLATE_DEFAULTS}
)
# What a pair may not know on either side: those settings and the hardware.
UNKNOWN_FIELDS = (*KNOWN_FIELDS, "hardware")
# The weights a pair must know and share.
WEIGHT_KEYS = ("model_revision", "quantization")
KINDS = ("engine", "kv_cache", "mixed")
# The run configuration a ranking and a Pareto view group pools by, beside
# the case set: the build. The per-response cap, streaming mode and sampling
# are no grouping keys: pools that differ in them share a ranking and show
# them in their pool lines.
GROUP_KEYS = ("harness_sha256",)
# Where a hardware statement comes from, and how renderings name each
# source: a hardware label, the serving note (its accelerator before the local
# read, as a loopback endpoint can be a tunnel to another machine; its memory
# and memory type alone where no GPU was read) and the local GPUs detected for
# a loopback endpoint, which the note's memory and memory type complete.
SOURCE_TEXTS = {
    "label": "hardware label",
    "note": "serving note",
    "detected": "detected local GPUs",
}
HARDWARE_SOURCES = tuple(SOURCE_TEXTS)
# The serving note's keys that state the hardware.
NOTE_HARDWARE = ("accelerator", "accelerator_memory_gib", "memory_type")
# The identity fields an evaluation exports per pool, and nothing else.
SUMMARY_KEYS = (
    "serving_runtime",
    "serving_version",
    "context_profile",
    "model_revision",
    "quantization",
    "kv_cache_dtype",
    "speculative_decoding",
    "accelerator",
    "accelerator_memory_gib",
    "memory_type",
)


def _detected_accelerator(serving: dict[str, Any]) -> tuple[str | None, int | None]:
    """The local accelerators of a loopback run as one setting: a single
    GPU's name, `n x name` for identical ones, mixed ones joined with ` + `;
    memory summed where every GPU reports it."""
    host = serving.get("host")
    found = host["accelerators"] if host is not None else None
    if not found:
        return None, None
    counts: dict[str, int] = {}
    for item in found:
        counts[item["name"]] = counts.get(item["name"], 0) + 1
    name = " + ".join(
        text if count == 1 else f"{count} x {text}" for text, count in counts.items()
    )
    memory = [item["memory_gib"] for item in found]
    total = None if None in memory else sum(memory)
    return (name if len(name) <= MAX_TEXT else None), total


def _local_read(serving: dict[str, Any]) -> tuple[str | None, int | None]:
    """The local accelerators and their memory (`_detected_accelerator`)
    where neither a hardware label nor the serving note's accelerator states
    the hardware, as a loopback endpoint can be a tunnel to the machine they
    name; (None, None) otherwise. A note's memory or memory type alone never
    sets the read aside: it completes it, as a GPU tool reports no memory
    for unified memory."""
    host = serving.get("host")
    note = serving["owner_note"] or {}
    if (host is not None and host["label"] is not None) or "accelerator" in note:
        return None, None
    return _detected_accelerator(serving)


def template_defaults(engine: dict[str, Any]) -> Any:
    """The default template variables a server reported, as one value: the
    variables, or the digest of a value outside their rule; None where none
    is reported or set, and in records of earlier formats."""
    found = engine.get(TEMPLATE_DEFAULTS)
    if found is None:
        return None
    return found["variables"] if found["variables"] is not None else found["sha256"]


def effective(serving: dict[str, Any]) -> dict[str, Any]:
    """A plan's serving settings as its identity reconciles them: runtime,
    version and context length from the probes where reported, otherwise
    from the owner note; quantization, KV cache and speculative decoding
    from the note where stated, otherwise as the engine reported them; the
    server's default template variables as the engine reported them; the
    accelerator from the note, or from the local GPU tool only where neither
    a hardware label nor the note's accelerator states the hardware, and its
    memory from the note where stated, otherwise from the same read
    (`hardware`), so a pool never names two accelerators; None where
    nothing gives one."""
    note = serving["owner_note"] or {}
    engine = serving["engine"]
    entry = serving["models"]["entry"]
    lengths = {
        value
        for value in (
            entry["max_model_len"] if entry is not None else None,
            engine["context_tokens"],
            note.get("context_profile"),
        )
        if value is not None
    }
    values = {key: note.get(key) for key in PAIR_FIELDS[:-1]}
    values.update(
        serving_runtime=engine["runtime"] or note.get("serving_runtime"),
        serving_version=engine["version"] or note.get("serving_version"),
        context_profile=lengths.pop() if len(lengths) == 1 else None,
        model_listing=json.dumps(
            [entry, engine["model_path"]], sort_keys=True, ensure_ascii=True
        ),
    )
    values[TEMPLATE_DEFAULTS] = template_defaults(engine)
    for key in ("model_revision", "quantization", *NOTE_HARDWARE):
        values[key] = note.get(key)
    # Earlier records carry neither engine settings nor a host section.
    for key in ENGINE_SETTINGS:
        if key not in note:
            values[key] = engine.get(key)
    accelerator, memory = _local_read(serving)
    if "accelerator" not in note:
        values["accelerator"] = accelerator
    if "accelerator_memory_gib" not in note:
        values["accelerator_memory_gib"] = memory
    return values


def accelerator_text(values: dict[str, Any]) -> str:
    """An accelerator, its memory and memory type as one text; empty where
    none is given."""
    memory = values.get("accelerator_memory_gib")
    size = " ".join(
        part
        for part in (
            None if memory is None else f"{memory} GiB",
            values.get("memory_type"),
        )
        if part is not None
    )
    return ", ".join(part for part in (values.get("accelerator"), size) if part)


def unused_read(serving: dict[str, Any]) -> dict[str, Any] | None:
    """The accelerator and memory the local read of a loopback run gave
    where a hardware label or the serving note's accelerator sets the read
    aside (`hardware`); None where the read is used or gave no accelerator.
    The read stays part of the setup fingerprint."""
    name, memory = _detected_accelerator(serving)
    if name is None or _local_read(serving)[0] is not None:
        return None
    return {"accelerator": name, "accelerator_memory_gib": memory}


def hardware(serving: dict[str, Any]) -> dict[str, str] | None:
    """A plan's hardware statement: the stated hardware label; otherwise the
    note's accelerator, memory and memory type where the note names the
    accelerator, as a loopback endpoint can be a tunnel to another machine;
    otherwise the local accelerators read for a loopback endpoint, with the
    note's memory and memory type where it states them and the read memory
    otherwise; otherwise the note's memory and memory type alone; None where
    none states any. Seconds compare only between pools with one
    statement."""
    host = serving.get("host")
    if host is not None and host["label"] is not None:
        return {"source": "label", "text": host["label"]}
    if _local_read(serving)[0] is not None:
        return {"source": "detected", "text": accelerator_text(effective(serving))}
    text = accelerator_text(serving["owner_note"] or {})
    return {"source": "note", "text": text} if text else None


def summary(serving: dict[str, Any]) -> dict[str, Any]:
    """The exported serving settings of a pool: what pairing and a model card
    need, never the models listing, a model path, host software, the chat
    template or the parser."""
    values = effective(serving)
    return {key: values[key] for key in SUMMARY_KEYS}


def _comparable(key: str, value: Any) -> Any:
    # A probed runtime name and a noted one differ only in case.
    if key == "serving_runtime" and isinstance(value, str):
        return value.casefold()
    return value


def case_order(entry: dict[str, Any], view: Any) -> tuple[str, ...]:
    """The order a run of a pool ran the pool's evaluated cases in: its
    plan's order without the cases outside them."""
    evaluated = set(entry["case_ids"])
    return tuple(c for c in view.plan["case_ids"] if c in evaluated)


def pairing_key(entry: dict[str, Any], members: list[Any]) -> str | None:
    """The key two pools share when they serve the same known weights (the
    model revision from the serving note, the quantization noted or reported
    by the engine) under the same run configuration, container image, order
    of the evaluated cases (`case_order`) and hardware statement; None for a
    pool that is not evaluable, whose weights are not known or whose runs
    differ in case order."""
    if not entry["evaluable"]:
        return None
    orders = {case_order(entry, view) for view in members}
    if len(orders) != 1:
        return None
    values = effective(members[0].plan["serving"])
    known = {key: values[key] for key in WEIGHT_KEYS}
    if None in known.values():
        return None
    return json.dumps(
        {
            **known,
            "configuration": entry["configuration"],
            "image": entry["image"],
            "case_ids": list(orders.pop()),
            "hardware": entry["hardware"],
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def differs(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """The names of the serving settings two plans differ in."""
    a, b = effective(first), effective(second)
    return [
        key
        for key in PAIR_FIELDS
        if _comparable(key, a[key]) != _comparable(key, b[key])
    ]


def unknown(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """The settings of `KNOWN_FIELDS` and the hardware statement that
    neither probes, note nor host give for at least one of two plans."""
    a, b = effective(first), effective(second)
    found = [key for key in KNOWN_FIELDS if a[key] is None or b[key] is None]
    if hardware(first) is None or hardware(second) is None:
        found.append("hardware")
    return found


def kind(names: list[str], missing: list[str] | tuple[str, ...] = ()) -> str:
    """`mixed` when a setting is unknown on either side, as a missing value
    never names a single cause; otherwise `kv_cache` when the KV-cache type
    is the only differing setting but serving-host software and the listing,
    `engine` when the runtime differs with at most its bound settings, host
    software and the listing, and `mixed` for a delta that no single setting
    explains."""
    if missing:
        return "mixed"
    other = set(names) - HOST
    if other == {"kv_cache_dtype"}:
        return "kv_cache"
    if "serving_runtime" in other and other - ENGINE_BOUND <= {"serving_runtime"}:
        return "engine"
    return "mixed"


def _family(attempts: list[Any]) -> tuple[float | None, bool]:
    """The rate of one family in one pool (passes over measured attempts;
    censored attempts count as not solved) and whether every repetition was
    measured."""
    done = [a for a in attempts if a.measured]
    if not done:
        return None, False
    return sum(a.passed for a in done) / len(done), len(done) == len(attempts)


def paired(
    profile: dict[str, Any],
    case_ids: list[str],
    first: dict[str, list[Any]],
    second: dict[str, list[Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Per category the paired families' differences (first minus second),
    their mean, a paired cluster-bootstrap interval and the paired case wins;
    overall the difference in points on the case set, only where both pools
    measured every attempt of every family."""
    weights, planned = profile["weights_percent"], planned_counts(profile, case_ids)
    rows: dict[str, dict[str, Any]] = {}
    strata: dict[str, list[float]] = {}
    complete = True
    for category in weights:
        families = [c for c in case_ids if CASES[c].category == category]
        if not families:
            continue
        found, values = [], []
        for case_id in families:
            rate_a, full_a = _family(first[case_id])
            rate_b, full_b = _family(second[case_id])
            complete = complete and full_a and full_b
            if rate_a is None or rate_b is None:
                continue
            values.append(rate_a - rate_b)
            found.append(
                {
                    "case_id": case_id,
                    "first_rate": rate_a,
                    "second_rate": rate_b,
                    "difference": rate_a - rate_b,
                }
            )
        rows[category] = {
            "families": len(families),
            "paired_families": len(found),
            "unpaired_families": len(families) - len(found),
            "delta": sum(values) / len(values) if values else None,
            "interval": None,
            "first_better": sum(v > 0 for v in values),
            "second_better": sum(v < 0 for v in values),
            "ties": sum(v == 0 for v in values),
            "first_censored": sum(
                a.censored for c in families for a in first[c] if a.measured
            ),
            "second_censored": sum(
                a.censored for c in families for a in second[c] if a.measured
            ),
            "cases": found,
        }
        if len(values) >= MIN_FAMILIES:
            strata[category] = values
    draws = bootstrap(strata, resamples=resamples, seed=seed) if strata else {}
    for category, means in draws.items():
        rows[category]["interval"] = list(interval(means))
    overall: dict[str, Any] = {"complete": complete, "points": None, "interval": None}
    if complete:
        # Both pools measured every attempt, so their unmeasured parts are
        # the same planned cases: lower and upper bounds differ by this.
        overall["points"] = sum(
            weights[k] * sum(f["difference"] for f in row["cases"]) / planned[k]
            for k, row in rows.items()
        )
        if set(draws) == set(rows):
            totals = [
                sum(
                    weights[k] * draws[k][index] * row["paired_families"] / planned[k]
                    for k, row in rows.items()
                )
                for index in range(resamples)
            ]
            overall["interval"] = list(interval(totals))
    return {"categories": rows, "overall": overall}


def comparisons(
    profile: dict[str, Any],
    entries: list[dict[str, Any]],
    members: dict[str, list[Any]],
    tries: dict[str, dict[str, list[Any]]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Every pair of pools with one pairing key, in pool-key order."""
    keyed: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        key = pairing_key(entry, members[entry["key"]])
        if key is not None:
            keyed.setdefault(key, []).append(entry)
    found = []
    for group in keyed.values():
        group = sorted(group, key=lambda item: item["key"])
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                plans = (
                    members[first["key"]][0].plan["serving"],
                    members[second["key"]][0].plan["serving"],
                )
                names, missing = differs(*plans), unknown(*plans)
                case_ids = list(case_order(first, members[first["key"]][0]))
                found.append(
                    {
                        "first": first["key"],
                        "second": second["key"],
                        "kind": kind(names, missing),
                        "differs": names,
                        "unknown": missing,
                        "case_ids": case_ids,
                        "first_flags": first["flags"],
                        "second_flags": second["flags"],
                        **paired(
                            profile,
                            first["case_ids"],
                            tries[first["key"]],
                            tries[second["key"]],
                            resamples=resamples,
                            seed=seed,
                        ),
                    }
                )
    return sorted(found, key=lambda item: (item["first"], item["second"]))


def group_configuration(entry: dict[str, Any]) -> dict[str, Any]:
    """The part of a pool's run configuration its ranking and Pareto view
    group by (`GROUP_KEYS`)."""
    return {key: entry["configuration"][key] for key in GROUP_KEYS}


def _text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def groups(
    entries: list[dict[str, Any]], *, timed: bool = False
) -> list[tuple[dict[str, str] | None, tuple[str, ...], list[dict[str, Any]]]]:
    """The pools that share a build (harness digest) and a case set, and
    with `timed` also a hardware statement, as (statement, case set, pools
    in input order), for every group of at least two pools; ordered by the
    statement's, the case set's and the build's text, so the groups' order
    never depends on the input order. Without `timed` the statement is None:
    rankings compare across hardware. With `timed` a pool without a hardware
    statement is never grouped: seconds compare only on one stated hardware.
    A pool's case list is in category order, so one case set is one list."""
    found: dict[tuple[str, tuple[str, ...], str], list[dict[str, Any]]] = {}
    for entry in entries:
        statement = entry["hardware"] if timed else None
        if timed and statement is None:
            continue
        group = (
            _text(statement),
            tuple(entry["case_ids"]),
            _text(group_configuration(entry)),
        )
        found.setdefault(group, []).append(entry)
    return [
        (members[0]["hardware"] if timed else None, case_ids, members)
        for (_, case_ids, _), members in sorted(found.items())
        if len(members) >= 2
    ]


def _overlap(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _point(entry: dict[str, Any], category: str) -> tuple[Any, ...]:
    """(x, y, y interval, seconds per attempt) of a pool, or a reason it is
    not placed: it is not evaluable, did not measure every attempt of the
    case set, or solved nothing (seconds per solved case undefined)."""
    if entry["scores"] is None:
        return ("not_evaluable",)
    runs = entry["repetitions"]
    scores, efficiency = entry["scores"], entry["efficiency"]
    if category == "overall":
        items = scores["categories"].values()
        measured = sum(item["measured_attempts"] for item in items)
        planned = sum(len(item["cases"]) for item in items) * runs
        points = scores["overall"]["points"]
        y, bounds = points["lower"], points["interval"]
        speed = efficiency["overall"]
    else:
        item = scores["categories"][category]
        measured, planned = item["measured_attempts"], len(item["cases"]) * runs
        y, bounds = item["rate"], item["interval"]
        speed = efficiency["categories"][category]
    if measured != planned:
        return ("incomplete_coverage",)
    if speed["seconds_per_solved"] is None or y is None:
        return ("no_solution",)
    per_attempt = speed["seconds"] / speed["attempts"]
    return (speed["seconds_per_solved"], y, bounds, per_attempt)


def pareto(
    profile: dict[str, Any], entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Within one timed group (`groups`: build, case set and hardware
    statement), per category and overall: the pools no other pool dominates
    (at least as high a rate on measured cases, or lower bound in points
    overall, and at most as many seconds per solved case, one strictly), and
    for every dominated pool all that dominate it, each marked separable only
    where both y intervals exist and do not overlap."""
    found = []
    for statement, case_ids, members in groups(entries, timed=True):
        for category in [*profile["weights_percent"], "overall"]:
            if category != "overall" and not any(
                CASES[c].category == category for c in case_ids
            ):
                continue
            placed, absent = [], []
            for entry in sorted(members, key=lambda item: item["key"]):
                point = _point(entry, category)
                if len(point) == 1:
                    absent.append({"pool": entry["key"], "reason": point[0]})
                else:
                    placed.append((entry["key"], *point))
            rows = []
            for key, x, y, bounds, per_attempt in placed:
                above = [
                    {
                        "pool": other,
                        "separable": None
                        if bounds is None or other_bounds is None
                        else not _overlap(bounds, other_bounds),
                    }
                    for other, ox, oy, other_bounds, _ in placed
                    if other != key and oy >= y and ox <= x and (oy > y or ox < x)
                ]
                rows.append(
                    {
                        "pool": key,
                        "seconds_per_solved": x,
                        "value": y,
                        "interval": bounds,
                        "seconds_per_attempt": per_attempt,
                        "front": not above,
                        "dominated_by": above,
                    }
                )
            found.append(
                {
                    "hardware": statement,
                    "configuration": group_configuration(members[0]),
                    "case_ids": list(case_ids),
                    "category": category,
                    "value": "lower_points" if category == "overall" else "rate",
                    "placed": rows,
                    "not_placed": absent,
                }
            )
    return found
