"""judge-harness: an LLM-as-judge eval harness that validates the judge itself.

Public surface:
    judge.judge      — get_judge / offline_judge / live_judge / mock_score
    judge.rubric     — RUBRIC, VERDICT_SCHEMA, normalize_verdict, PASS_THRESHOLD
    judge.agreement  — agreement_pct, cohens_kappa, pass_rate_ci, bootstrap_ci
    judge.stability  — run_stability_probe
    judge.pairwise   — position_bias_check, default_pairs
"""
