"""app.py — Streamlit UI for judge-harness.

Two tabs:
  1. Meta-eval — runs the judge against the bundled human labels and shows the
     honest scorecard (agreement %, Cohen's kappa, pass-rate +/- 95% CI,
     position-bias flip rate, stability probe).
  2. Try the judge — score an ad-hoc (prompt, response) pair.

Offline mock judge runs with no key. If ANTHROPIC_API_KEY is set, a "live"
toggle appears that routes to claude-haiku-4-5-20251001. The offline path never
imports anthropic / pydantic.

Run: streamlit run app.py
"""

import os

import streamlit as st

from eval_cli import compute_scorecard, load_gold
from judge.judge import get_judge

st.set_page_config(page_title="judge-harness", page_icon="⚖️", layout="centered")

st.title("⚖️ judge-harness")
st.caption(
    "LLM-as-judge you can actually trust — it scores responses on a rubric AND "
    "validates the judge itself against human labels (the meta-eval)."
)

HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

with st.sidebar:
    st.header("Judge mode")
    if HAS_KEY:
        mode = st.radio(
            "Backend",
            ["offline", "live"],
            help="offline = seeded stdlib mock (reproducible). "
            "live = claude-haiku-4-5-20251001.",
        )
    else:
        mode = "offline"
        st.info(
            "Running **offline** (deterministic seeded mock judge). "
            "Set `ANTHROPIC_API_KEY` to unlock live mode "
            "(`claude-haiku-4-5-20251001`)."
        )
    st.markdown("---")
    st.markdown(
        "**Honest scope.** This is a *meta-eval*: the offline judge is a seeded "
        "mock so the metrics are reproducible without a key. LLM judges are "
        "biased and non-deterministic — this harness **measures** that "
        "(position-bias flip rate + bootstrap CI) rather than hiding it."
    )
    st.markdown(
        "Sibling project: **slop-scanner** is the *rule-graded* gate; this is "
        "its *model-graded* sibling, validated against humans."
    )

tab_eval, tab_try = st.tabs(["Meta-eval scorecard", "Try the judge"])

with tab_eval:
    st.subheader("Validate the judge against humans")
    n_runs = st.slider("Stability runs", 3, 30, 10)
    if st.button("Run meta-eval", type="primary"):
        with st.spinner(f"Running judge ({mode}) over the gold set..."):
            sc = compute_scorecard(mode=mode, n_runs=n_runs)

        c1, c2, c3 = st.columns(3)
        c1.metric("Judge↔human agreement", f"{sc['agreement_pct'] * 100:.1f}%")
        c2.metric("Cohen's kappa", f"{sc['cohens_kappa']:.2f}")
        lo, hi = sc["pass_rate_ci95"]
        c3.metric(
            "Pass-rate",
            f"{sc['pass_rate']:.2f}",
            f"95% CI {lo:.2f}–{hi:.2f}",
            delta_color="off",
        )

        c4, c5 = st.columns(2)
        pb = sc["position_bias"]
        c4.metric(
            "Position-bias flip rate",
            f"{pb['flip_rate'] * 100:.0f}%",
            f"{pb['flips']}/{pb['n_pairs']} A/B pairs",
            delta_color="off",
        )
        stb = sc["stability"]
        half = (stb["ci95"][1] - stb["ci95"][0]) / 2
        c5.metric(
            f"Stability ({stb['n_runs']} runs)",
            f"{stb['mean_pass_rate']:.2f} ± {half:.2f}",
            f"var {stb['variance']:.5f}",
            delta_color="off",
        )

        st.markdown(
            f"**Headline:** judge↔human kappa = **{sc['cohens_kappa']:.2f}** on "
            f"**{sc['n_labels']}** human labels; pass-rate "
            f"**{sc['pass_rate']:.2f} ± {half:.2f}** over {stb['n_runs']} runs; "
            f"position-bias flip rate **{pb['flip_rate'] * 100:.0f}%**."
        )
        st.line_chart(
            {"pass-rate per run": stb["per_run_pass_rate"]},
            height=200,
        )
        with st.expander("Raw scorecard.json"):
            st.json(sc)

with tab_try:
    st.subheader("Score one (prompt, response) pair")
    prompt = st.text_area("Prompt", "How do I reverse a list in Python?")
    response = st.text_area(
        "Response",
        "Use slicing: my_list[::-1] returns a reversed copy. For example "
        "[1,2,3][::-1] gives [3,2,1].",
    )
    seed = st.number_input("Seed (offline jitter)", 0, 9999, 0)
    if st.button("Judge it"):
        judge = get_judge(mode)
        verdict = judge(prompt, response, int(seed))
        cols = st.columns(2)
        cols[0].metric("Score (1–5)", verdict["score"])
        cols[1].metric("Verdict", verdict["label"].upper())
        st.write(f"**Reason:** {verdict['reason']}")
        st.caption(
            "Tip: in offline mode, change the seed to see the judge's "
            "non-determinism — that variance is what the stability probe "
            "quantifies."
        )

    st.markdown("---")
    with st.expander("Peek at the gold set"):
        st.dataframe(load_gold(), use_container_width=True)

st.markdown("---")
st.caption("Christian Macion — AI / Agent Engineer")
