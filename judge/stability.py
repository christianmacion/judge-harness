"""stability.py — the stability probe.

LLM judges are non-deterministic. This probe quantifies that: run the judge N
times on the same (prompt, response) set and measure how much the PASS-RATE
moves run to run. In offline mode the variation comes from seeded jitter (each
run uses a different seed); in live mode it comes from temperature>0.

Returns per-run pass-rates, their spread (min/max/variance), and a 95%
bootstrap CI on the pooled pass-rate. Pure stdlib.
"""

from judge.agreement import bootstrap_ci, pass_rate


def run_stability_probe(judge, items, n_runs=10, base_seed=0):
    """Run `judge` over `items` `n_runs` times and report pass-rate stability.

    judge: callable (prompt, response, seed) -> verdict dict
    items: list of dicts with at least "prompt" and "response"
    Each run r uses seed = base_seed + r (so runs differ but are reproducible).

    Returns a dict:
      {
        "n_runs", "per_run_pass_rate" [list], "mean_pass_rate",
        "min_pass_rate", "max_pass_rate", "variance",
        "ci_low", "ci_high"   # 95% bootstrap CI over per-run pass-rates
      }
    """
    if not items:
        return {
            "n_runs": n_runs,
            "per_run_pass_rate": [],
            "mean_pass_rate": 0.0,
            "min_pass_rate": 0.0,
            "max_pass_rate": 0.0,
            "variance": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    per_run = []
    for r in range(n_runs):
        seed = base_seed + r
        labels = [
            judge(it["prompt"], it["response"], seed)["label"] for it in items
        ]
        per_run.append(pass_rate(labels))

    mean_pr = sum(per_run) / len(per_run)
    var = sum((x - mean_pr) ** 2 for x in per_run) / len(per_run)
    # Bootstrap CI over the per-run pass-rates (resample the runs).
    ci_low, ci_high = bootstrap_ci(per_run, n_boot=2000, alpha=0.05, seed=base_seed)

    return {
        "n_runs": n_runs,
        "per_run_pass_rate": per_run,
        "mean_pass_rate": mean_pr,
        "min_pass_rate": min(per_run),
        "max_pass_rate": max(per_run),
        "variance": var,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
