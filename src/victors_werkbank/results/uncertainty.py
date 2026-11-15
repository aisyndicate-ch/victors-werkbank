"""Uncertainty for evaluation across runs: a seeded cluster bootstrap over
case families, stratified by category (docs/METHODOLOGY.md, section
Sampling, repetitions and uncertainty).

A family is one case ID (a security case's two episodes are one family), and
its value is the mean over the repetitions of a pool, so repetitions are never
counted as independent families. Standard library only; the same values,
resamples and seed always give the same interval.
"""

import random

RESAMPLES = 10_000
SEED = 0
INTERVAL_PERCENT = 95
# A category with fewer measured families has no interval.
MIN_FAMILIES = 2


def bootstrap(
    strata: dict[str, list[float]], *, resamples: int, seed: int
) -> dict[str, list[float]]:
    """Per stratum, `resamples` means of its family values drawn with
    replacement within the stratum. Draws follow the strata's order and each
    stratum's value order, so callers fix both (categories in profile order,
    families by case ID)."""
    if resamples < 1 or any(not values for values in strata.values()):
        raise ValueError("A bootstrap needs resamples and values in every stratum")
    rng = random.Random(seed)
    means: dict[str, list[float]] = {key: [] for key in strata}
    for _ in range(resamples):
        for key, values in strata.items():
            size = len(values)
            means[key].append(sum(values[rng.randrange(size)] for _ in values) / size)
    return means


def interval(
    values: list[float], percent: int = INTERVAL_PERCENT
) -> tuple[float, float]:
    """Nearest-rank percentile interval holding `percent` of `values`; the
    ranks are computed in integers, so no rounding moves a bound."""
    if not values or not 0 < percent < 100:
        raise ValueError("An interval needs values and a percent in 1..99")
    ordered = sorted(values)
    size = len(ordered)
    tail = 100 - percent
    low = -(-size * tail // 200)
    high = -(-size * (200 - tail) // 200)
    return ordered[max(low, 1) - 1], ordered[high - 1]
