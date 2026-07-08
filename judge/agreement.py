"""agreement.py — judge<->human agreement metrics in PURE PYTHON STDLIB.

No numpy, no sklearn. Implements:
  - raw agreement %
  - Cohen's kappa (chance-corrected agreement)
  - pass-rate with a 95% bootstrap confidence interval (seeded `random`)

These are the "meta-eval" metrics: they measure whether the JUDGE agrees with
HUMAN labels, not whether responses are good. Labels are binary "pass"/"fail".
"""

import random
from collections import Counter


def agreement_pct(judge_labels, human_labels):
    """Fraction of items where judge label == human label (0.0..1.0)."""
    if not judge_labels:
        return 0.0
    if len(judge_labels) != len(human_labels):
        raise ValueError("label lists must be the same length")
    hits = sum(1 for j, h in zip(judge_labels, human_labels) if j == h)
    return hits / len(judge_labels)


def cohens_kappa(judge_labels, human_labels):
    """Cohen's kappa for two raters over a categorical set (here: pass/fail).

    kappa = (p_o - p_e) / (1 - p_e)
      p_o = observed agreement
      p_e = expected agreement by chance, from the marginal distributions.

    Returns a float. 1.0 = perfect, 0.0 = chance-level, <0 = worse than chance.
    Handles the degenerate case (all labels identical -> p_e == 1) by returning
    1.0 when observed agreement is also perfect, else 0.0.
    """
    n = len(judge_labels)
    if n == 0:
        return 0.0
    if n != len(human_labels):
        raise ValueError("label lists must be the same length")

    categories = set(judge_labels) | set(human_labels)
    p_o = agreement_pct(judge_labels, human_labels)

    judge_counts = Counter(judge_labels)
    human_counts = Counter(human_labels)
    p_e = sum(
        (judge_counts[c] / n) * (human_counts[c] / n) for c in categories
    )

    if p_e == 1.0:
        # Both raters used a single category for everything.
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def pass_rate(judge_labels):
    """Fraction of items the judge labelled 'pass' (0.0..1.0)."""
    if not judge_labels:
        return 0.0
    return sum(1 for j in judge_labels if j == "pass") / len(judge_labels)


def bootstrap_ci(values, statistic=None, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a statistic over `values`.

    Default statistic = mean (works for a 0/1 indicator list -> a proportion CI).
    Resamples WITH replacement using a seeded `random.Random` for reproducibility.

    Returns (low, high) at the (alpha/2, 1-alpha/2) percentiles.
    """
    if not values:
        return (0.0, 0.0)
    if statistic is None:
        statistic = lambda xs: sum(xs) / len(xs)

    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()

    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    hi_idx = max(lo_idx, min(hi_idx, n_boot - 1))
    return (stats[lo_idx], stats[hi_idx])


def pass_rate_ci(judge_labels, n_boot=2000, alpha=0.05, seed=0):
    """95% bootstrap CI for the pass-rate. Returns (rate, low, high)."""
    indicators = [1 if j == "pass" else 0 for j in judge_labels]
    rate = pass_rate(judge_labels)
    low, high = bootstrap_ci(indicators, n_boot=n_boot, alpha=alpha, seed=seed)
    return rate, low, high
