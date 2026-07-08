"""eval_cli.py — the one command that prints the honest meta-eval metrics.

Run:
    python3 eval_cli.py                 # offline mock judge (no key needed)
    ANTHROPIC_API_KEY=... python3 eval_cli.py --mode live

It loads gold/labeled.jsonl, runs the judge, and reports:
    - judge<->human agreement %
    - Cohen's kappa
    - pass-rate with a 95% bootstrap CI
    - position-bias flip rate
    - a stability probe (pass-rate variance over N runs)
then writes scorecard.json.

This is the model-graded sibling of slop-scanner's rule-graded gate: same
(prompt, response) shape, but here we ALSO validate the grader against humans.
"""

import argparse
import json
import os
import sys

from judge.agreement import agreement_pct, cohens_kappa, pass_rate_ci
from judge.judge import get_judge
from judge.pairwise import default_pairs, position_bias_check
from judge.stability import run_stability_probe

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_PATH = os.path.join(HERE, "gold", "labeled.jsonl")
SCORECARD_PATH = os.path.join(HERE, "scorecard.json")


def load_gold(path=GOLD_PATH):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def compute_scorecard(mode=None, n_runs=10, seed=0):
    """Run the full meta-eval and return a scorecard dict."""
    judge = get_judge(mode)
    resolved_mode = mode or ("live" if os.environ.get("ANTHROPIC_API_KEY") else "offline")
    items = load_gold()

    # Single-pass verdicts (seed=0) for agreement + kappa vs human labels.
    judge_labels = [judge(it["prompt"], it["response"], seed)["label"] for it in items]
    human_labels = [it["human_label"] for it in items]

    agree = agreement_pct(judge_labels, human_labels)
    kappa = cohens_kappa(judge_labels, human_labels)
    rate, ci_low, ci_high = pass_rate_ci(judge_labels, seed=seed)

    stability = run_stability_probe(judge, items, n_runs=n_runs, base_seed=seed)
    pairs = default_pairs(items)
    bias = position_bias_check(judge, pairs, seed=seed)

    return {
        "mode": resolved_mode,
        "n_labels": len(items),
        "agreement_pct": agree,
        "cohens_kappa": kappa,
        "pass_rate": rate,
        "pass_rate_ci95": [ci_low, ci_high],
        "stability": {
            "n_runs": stability["n_runs"],
            "mean_pass_rate": stability["mean_pass_rate"],
            "min_pass_rate": stability["min_pass_rate"],
            "max_pass_rate": stability["max_pass_rate"],
            "variance": stability["variance"],
            "ci95": [stability["ci_low"], stability["ci_high"]],
            "per_run_pass_rate": stability["per_run_pass_rate"],
        },
        "position_bias": {
            "n_pairs": bias["n_pairs"],
            "flips": bias["flips"],
            "flip_rate": bias["flip_rate"],
        },
    }


def print_scorecard(sc):
    print("=" * 60)
    print(f"  judge-harness meta-eval  (mode: {sc['mode']})")
    print("=" * 60)
    print(f"  labels (humans):        {sc['n_labels']}")
    print(f"  judge<->human agreement: {sc['agreement_pct'] * 100:5.1f} %")
    print(f"  Cohen's kappa:           {sc['cohens_kappa']:+.3f}")
    pr = sc["pass_rate"]
    lo, hi = sc["pass_rate_ci95"]
    print(f"  pass-rate:               {pr:.3f}  (95% CI {lo:.3f} - {hi:.3f})")
    st = sc["stability"]
    half = (st["ci95"][1] - st["ci95"][0]) / 2
    print(
        f"  stability ({st['n_runs']} runs):     mean {st['mean_pass_rate']:.3f} "
        f"+/- {half:.3f}  (var {st['variance']:.5f})"
    )
    pb = sc["position_bias"]
    print(
        f"  position-bias flip rate: {pb['flip_rate'] * 100:5.1f} % "
        f"({pb['flips']}/{pb['n_pairs']} pairs)"
    )
    print("=" * 60)
    headline = (
        f"  HEADLINE: kappa={sc['cohens_kappa']:.2f} on {sc['n_labels']} labels; "
        f"pass-rate {pr:.2f} +/- {half:.2f} over {st['n_runs']} runs; "
        f"position-bias {pb['flip_rate'] * 100:.0f}%"
    )
    print(headline)
    print("=" * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description="judge-harness meta-eval")
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default=None,
        help="judge mode (default: auto — live iff ANTHROPIC_API_KEY set)",
    )
    parser.add_argument("--runs", type=int, default=10, help="stability runs")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    args = parser.parse_args(argv)

    sc = compute_scorecard(mode=args.mode, n_runs=args.runs, seed=args.seed)
    print_scorecard(sc)

    with open(SCORECARD_PATH, "w", encoding="utf-8") as f:
        json.dump(sc, f, indent=2)
    print(f"  wrote {SCORECARD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
