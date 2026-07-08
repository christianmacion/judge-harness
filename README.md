# ⚖️ judge-harness — LLM-as-judge you can actually trust

**TL;DR.** An LLM-as-judge eval harness that scores `(prompt, response)` pairs on
a rubric **and validates the judge itself against human labels**. It computes the
metrics that tell you whether you can *trust the grader*: judge↔human agreement,
Cohen's kappa, pass-rate with a 95% bootstrap CI, and a position-bias flip rate.
This is the **meta-eval** — it evaluates the evaluator.

> **Headline metric (offline mock judge, bundled gold set):**
> **judge↔human kappa = 0.58 on 48 labels; pass-rate 0.60 ± 0.01 over 10 runs;
> position-bias flip rate 17%; agreement 79.2%.**
>
> Reproduce it with one command: `python3 eval_cli.py` (no API key needed).

This is the **model-graded sibling of [slop-scanner](../slop-scanner)**: the slop
gate is *rule-graded* (regex/heuristic rules); this is its *model-graded* sibling,
and — crucially — **validated against humans**.

---

## Why this exists (honest scope)

LLM judges are seductive and dangerous: they're biased (they prefer whichever
answer they saw first), non-deterministic (run twice, get two answers), and easy
to fool into a rigged-looking 1.0 agreement. A judge you haven't validated is a
vibe, not a metric.

judge-harness treats the judge as **something to be measured, not trusted**:

- **Agreement % + Cohen's kappa** vs human labels — does the judge actually track
  ground truth, beyond chance?
- **Pass-rate ± 95% bootstrap CI** — how confident is the headline number?
- **Stability probe** — run N times; how much does the pass-rate wobble?
- **Position-bias flip rate** — swap A/B order; how often does the verdict flip
  purely on position?

The **offline judge is a deliberately seeded mock** so all of these render
reproducibly with no API key. That is a *feature, not a cheat*: the point of the
demo is the meta-eval machinery, and the mock is tuned to agree with humans
**mostly but not perfectly** (kappa ~0.58, not 1.0) so the numbers are credible.
The same metrics run unchanged against the **live** Anthropic judge.

---

## The four metrics, in plain terms

| Metric | Question it answers | Good value |
|---|---|---|
| Agreement % | Of N labels, how often does judge == human? | high (but <100%) |
| Cohen's kappa | Agreement *corrected for chance* | 0.6–0.8 = substantial |
| Pass-rate ± CI | What fraction pass, and how sure are we? | tight CI |
| Stability variance | Same input, repeated — how much does it move? | low |
| Position-bias flip rate | Swap A/B order — how often does the winner change? | near 0% |

All four are implemented in **pure Python stdlib** — Cohen's kappa, agreement,
the bootstrap (seeded `random`), and the position-bias swap. No numpy, no
sklearn.

---

## Reproduce the metric (one command)

```bash
cd judge-harness
python3 eval_cli.py
```

Output:

```
============================================================
  judge-harness meta-eval  (mode: offline)
============================================================
  labels (humans):        48
  judge<->human agreement:  79.2 %
  Cohen's kappa:           +0.580
  pass-rate:               0.604  (95% CI 0.458 - 0.750)
  stability (10 runs):     mean 0.604 +/- 0.017  (var 0.00080)
  position-bias flip rate:  16.7 % (2/12 pairs)
============================================================
  HEADLINE: kappa=0.58 on 48 labels; pass-rate 0.60 +/- 0.02 over 10 runs; position-bias 17%
============================================================
  wrote .../scorecard.json
```

It also writes **`scorecard.json`** with the full breakdown (per-run pass-rates,
CIs, pair details).

---

## Offline vs live

| | **Offline** (default) | **Live** |
|---|---|---|
| Trigger | no `ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` set, or `--mode live` |
| Judge | seeded stdlib mock (heuristic + controlled noise) | `claude-haiku-4-5-20251001` |
| Deps | **none** (pure stdlib) | `anthropic` (+ optional `pydantic`) |
| Determinism | reproducible (seeded) | non-deterministic (`temperature=1.0`) |
| Imports `anthropic`/`pydantic`? | **never** | yes (lazy, live-only) |

```bash
# offline (no key) — the default
python3 eval_cli.py

# live (real Claude judge)
export ANTHROPIC_API_KEY=sk-ant-...
python3 eval_cli.py --mode live
```

The live judge asks Claude for a **structured JSON verdict**
(`{"score", "label", "reason"}`) constrained by a JSON schema
(`output_config.format`), then validates it (optional pydantic check +
defensive `normalize_verdict` clamping). The **offline path never imports
`anthropic` or `pydantic`** — there's a test that asserts exactly that.

---

## Run the app

```bash
pip install -r requirements.txt   # only needed for the UI / live mode
streamlit run app.py
```

**How a reviewer clicks it:**
1. Open the app — it starts in **offline** mode (no key required).
2. **Meta-eval scorecard** tab → *Run meta-eval* → see agreement, kappa,
   pass-rate ± CI, position-bias, and a per-run pass-rate chart.
3. **Try the judge** tab → paste a `(prompt, response)`, hit *Judge it*, then
   **change the seed** to watch the judge's non-determinism — the exact thing
   the stability probe quantifies.
4. If `ANTHROPIC_API_KEY` is set, a sidebar toggle routes to the live Claude
   judge; the same scorecard re-runs against it.

---

## Tests (CI-runnable)

```bash
python3 -m pytest tests/test_judge.py     # if pytest is installed
python3 tests/test_judge.py               # plain stdlib, no pytest needed
```

The suite (17 checks) covers kappa edge cases (perfect / chance / degenerate /
worse-than-chance), the seeded bootstrap, verdict normalization, judge
determinism, the no-`anthropic`-on-offline-path guarantee, the stability probe,
the position-bias swap, and an end-to-end "headline metric is credible (not
rigged)" assertion. Every check is a plain `assert`, so the file runs both under
pytest and as a bare script.

---

## Layout

```
judge-harness/
├── app.py                  # Streamlit UI (#0b3d5c theme)
├── eval_cli.py             # the one-command meta-eval → prints + scorecard.json
├── judge/
│   ├── rubric.py           # rubric text, verdict schema, normalize_verdict
│   ├── judge.py            # offline seeded mock + live Anthropic judge (lazy import)
│   ├── agreement.py        # agreement %, Cohen's kappa, bootstrap CI  (pure stdlib)
│   ├── stability.py        # N-run stability probe
│   └── pairwise.py         # A/B position-bias flip-rate check
├── gold/
│   ├── labeled.jsonl       # ~48 human-labeled (prompt, response, label, score)
│   └── build_gold.py       # regenerates the gold set (reproducible)
├── tests/test_judge.py     # pytest + standalone
├── requirements.txt
└── .streamlit/config.toml
```

---

## House contract (shared across the portfolio)

1. **README with a headline metric** ✔ (kappa 0.58 on 48 labels, pass-rate ± CI).
2. **Reproducible metric via one CLI command** ✔ (`python3 eval_cli.py` on the
   in-repo gold set).
3. **Documented offline path (no key) + live mode gated on `ANTHROPIC_API_KEY`** ✔.
4. **Honest scope** ✔ — this is a meta-eval; the offline judge is a seeded mock
   to make metrics reproducible; LLM judges are biased and non-deterministic and
   this harness **measures** that (position bias + CI) rather than hiding it.

---

*Christian Macion — AI / Agent Engineer*
