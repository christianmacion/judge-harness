# Deploy — judge-harness

**TL;DR:** a public URL in ~5 minutes on Streamlit Community Cloud (free). Works with **no API key** (offline seeded-mock judge makes every metric reproducible); add `ANTHROPIC_API_KEY` for a live Claude judge.

## 0. Prerequisites
- GitHub account + a Streamlit Community Cloud account ([share.streamlit.io](https://share.streamlit.io), sign in with GitHub — free).
- *(Optional, live mode only)* an Anthropic API key.

## 1. Own public GitHub repo
```bash
cd "06_projects/judge-harness"
git init && git add . && git commit -m "judge-harness: a validated LLM-as-judge meta-eval"
gh repo create judge-harness --public --source=. --push
```

## 2. Deploy on Streamlit Community Cloud
1. [share.streamlit.io](https://share.streamlit.io) → **Create app** → **Deploy from GitHub**.
2. Repo `<you>/judge-harness` · Branch `main` · **Main file path: `app.py`**.
3. *(Optional, live judge)* **Advanced settings → Secrets**:
   ```toml
   ANTHROPIC_API_KEY="sk-ant-..."
   ```
   (Streamlit exposes secrets as env vars; the app's `os.getenv` reads it.)
4. **Deploy** → permanent URL `https://<app>.streamlit.app`.

## What a reviewer sees
Click **Validate judge** → **judge↔human Cohen's κ = 0.58** on 48 human labels (agreement 79%). Click **Stability** → pass-rate **0.60 ± 0.02** over 10 runs with a 95% bootstrap CI. Click **Pairwise** → a **17% position-bias flip rate** — the harness proving the judge itself is biased and quantifying it.

## Run locally (and in CI)
```bash
pip install -r requirements.txt
python eval_cli.py              # prints all four metrics, writes scorecard.json
python tests/test_judge.py      # 17 checks (also works under: python -m pytest)
streamlit run app.py
```
This repo is **CI-friendly** — the test suite runs without an API key, so you can wire it into GitHub Actions as a regression gate.

## Alternative host: Hugging Face Spaces (SDK: Streamlit). Add the key under **Settings → Variables and secrets** for live mode.

---
*Christian Macion — AI / Agent Engineer.*
