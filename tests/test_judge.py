"""test_judge.py — CI-runnable test suite for the judge harness.

Runnable two ways:
    python3 -m pytest tests/test_judge.py      (if pytest is installed)
    python3 tests/test_judge.py                (plain stdlib, no pytest needed)

Every check is a plain `assert`, so the file doubles as a standalone script. The
offline mock judge is deterministic and key-free, so the whole suite runs in CI
with no network and no ANTHROPIC_API_KEY.
"""

import json
import os
import sys

# Make the repo root importable whether run via pytest or as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from judge.agreement import (  # noqa: E402
    agreement_pct,
    bootstrap_ci,
    cohens_kappa,
    pass_rate,
    pass_rate_ci,
)
from judge.judge import get_judge, mock_score, offline_judge  # noqa: E402
from judge.pairwise import default_pairs, position_bias_check  # noqa: E402
from judge.rubric import PASS_THRESHOLD, normalize_verdict  # noqa: E402
from judge.stability import run_stability_probe  # noqa: E402

GOLD = os.path.join(ROOT, "gold", "labeled.jsonl")


def _load_gold():
    with open(GOLD, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# agreement / kappa primitives
# --------------------------------------------------------------------------
def test_agreement_perfect_and_none():
    assert agreement_pct(["pass", "fail"], ["pass", "fail"]) == 1.0
    assert agreement_pct(["pass", "fail"], ["fail", "pass"]) == 0.0


def test_kappa_perfect_is_one():
    # Mixed labels, perfect agreement -> kappa 1.0
    j = ["pass", "fail", "pass", "fail"]
    assert abs(cohens_kappa(j, j) - 1.0) < 1e-9


def test_kappa_chance_is_near_zero():
    # Judge ignores truth entirely (alternating) vs balanced humans -> ~0.
    judge = ["pass", "fail", "pass", "fail"]
    human = ["pass", "pass", "fail", "fail"]
    k = cohens_kappa(judge, human)
    assert -0.6 < k < 0.6  # not strongly positive; chance-level


def test_kappa_degenerate_all_same():
    # Both raters all "pass" -> p_e == 1; observed perfect -> defined as 1.0
    assert cohens_kappa(["pass"] * 5, ["pass"] * 5) == 1.0


def test_kappa_worse_than_chance_is_negative():
    judge = ["pass", "fail", "pass", "fail"]
    human = ["fail", "pass", "fail", "pass"]
    assert cohens_kappa(judge, human) < 0.0


# --------------------------------------------------------------------------
# bootstrap CI
# --------------------------------------------------------------------------
def test_bootstrap_ci_is_seeded_and_bracketing():
    vals = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]  # mean 0.7
    lo1, hi1 = bootstrap_ci(vals, seed=0)
    lo2, hi2 = bootstrap_ci(vals, seed=0)
    assert (lo1, hi1) == (lo2, hi2)  # reproducible
    assert lo1 <= 0.7 <= hi1  # CI brackets the point estimate
    assert 0.0 <= lo1 <= hi1 <= 1.0


def test_pass_rate_ci_shape():
    labels = ["pass", "pass", "fail", "pass"]
    rate, lo, hi = pass_rate_ci(labels, seed=0)
    assert rate == 0.75
    assert lo <= rate <= hi


# --------------------------------------------------------------------------
# rubric / verdict normalization
# --------------------------------------------------------------------------
def test_normalize_clamps_score():
    v = normalize_verdict({"score": 9, "label": "pass", "reason": "x"})
    assert v["score"] == 5
    v = normalize_verdict({"score": -3, "label": "fail", "reason": "x"})
    assert v["score"] == 1


def test_normalize_derives_label_when_missing():
    v = normalize_verdict({"score": 4})
    assert v["label"] == "pass"  # 4 >= PASS_THRESHOLD
    v = normalize_verdict({"score": 2})
    assert v["label"] == "fail"


def test_normalize_rejects_garbage():
    try:
        normalize_verdict({"label": "pass"})  # no score
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on missing score")


# --------------------------------------------------------------------------
# offline judge behavior
# --------------------------------------------------------------------------
def test_offline_judge_is_deterministic():
    p, r = "How do I reverse a list?", "Use my_list[::-1] to reverse it."
    a = offline_judge(p, r, seed=0)
    b = offline_judge(p, r, seed=0)
    assert a == b


def test_offline_judge_seed_changes_output_sometimes():
    # Across many seeds the score should not be constant (controlled noise).
    p, r = "Explain recursion.", "Recursion is when a function calls itself."
    scores = {mock_score(p, r, seed=s) for s in range(20)}
    assert len(scores) > 1


def test_offline_judge_passes_good_fails_bad():
    good = offline_judge(
        "How do I read a file?",
        "Use open() and read(): with open('f') as f: data = f.read(). "
        "For example you can then print the contents.",
        seed=0,
    )
    bad = offline_judge("How do I read a file?", "I don't know, google it.", seed=0)
    assert good["label"] == "pass"
    assert bad["label"] == "fail"


def test_offline_path_does_not_import_anthropic():
    # The whole offline pipeline must run without anthropic/pydantic present.
    judge = get_judge("offline")
    judge("q", "a clear helpful answer with an example here", 0)
    assert "anthropic" not in sys.modules, "offline path imported anthropic!"
    assert "pydantic" not in sys.modules, "offline path imported pydantic!"


# --------------------------------------------------------------------------
# stability probe
# --------------------------------------------------------------------------
def test_stability_probe_shape_and_bounds():
    items = _load_gold()
    res = run_stability_probe(get_judge("offline"), items, n_runs=8, base_seed=0)
    assert res["n_runs"] == 8
    assert len(res["per_run_pass_rate"]) == 8
    assert res["variance"] >= 0.0
    assert res["min_pass_rate"] <= res["mean_pass_rate"] <= res["max_pass_rate"]
    assert 0.0 <= res["ci_low"] <= res["ci_high"] <= 1.0


# --------------------------------------------------------------------------
# pairwise position bias
# --------------------------------------------------------------------------
def test_position_bias_rate_in_unit_interval():
    items = _load_gold()
    pairs = default_pairs(items)
    res = position_bias_check(get_judge("offline"), pairs, seed=0)
    assert res["n_pairs"] == len(pairs)
    assert 0.0 <= res["flip_rate"] <= 1.0
    assert res["flips"] == sum(1 for d in res["details"] if d["flipped"])


# --------------------------------------------------------------------------
# end-to-end headline metric on the bundled gold set
# --------------------------------------------------------------------------
def test_headline_metric_is_credible():
    """Mock should agree with humans MOSTLY but not perfectly (honest kappa)."""
    items = _load_gold()
    assert len(items) >= 40
    judge = get_judge("offline")
    jl = [judge(it["prompt"], it["response"], 0)["label"] for it in items]
    hl = [it["human_label"] for it in items]

    agree = agreement_pct(jl, hl)
    kappa = cohens_kappa(jl, hl)
    rate = pass_rate(jl)

    # Credible, not rigged: high-but-imperfect agreement, mid-high kappa.
    assert 0.65 <= agree < 1.0, f"agreement {agree} outside credible band"
    assert 0.45 <= kappa < 0.9, f"kappa {kappa} outside credible band"
    assert 0.0 < rate < 1.0, f"pass-rate {rate} degenerate"


def _run_all():
    """Run every test function in this module as a plain script."""
    fns = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed.")
    return passed == len(fns)


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
