"""judge.py — the judge itself, in two modes.

OFFLINE (default, no key): a deterministic SEEDED MOCK judge implemented in pure
stdlib. It scores a response with a transparent heuristic + controlled,
seeded noise so that EVERY metric (agreement, kappa, bootstrap CI, position
bias) renders reproducibly with no API key. The mock is tuned to agree with the
bundled human labels MOSTLY but not perfectly (kappa ~0.6-0.8) so the meta-eval
numbers are credible rather than a rigged 1.0.

LIVE (gated on ANTHROPIC_API_KEY): calls the Anthropic SDK with model
`claude-haiku-4-5-20251001`, asking for a structured JSON verdict validated
against a small schema. anthropic / pydantic are LAZY-IMPORTED here and only on
the live path — the offline path never imports them.

This is the model-graded sibling of the rule-graded slop-scanner: same
(prompt, response) input shape, but graded by a model (mock or live) and then
validated against humans by judge/agreement.py.
"""

import hashlib
import os
import random

from judge.rubric import (
    PASS_THRESHOLD,
    RUBRIC,
    VERDICT_SCHEMA,
    normalize_verdict,
)

LIVE_MODEL = "claude-haiku-4-5-20251001"

# --- words the heuristic uses as cheap proxies for helpfulness -------------
_GOOD_SIGNALS = (
    "because", "example", "specifically", "step", "first", "here", "use",
    "code", "function", "def ", "return", "you can", "for instance", "note",
)
_BAD_SIGNALS = (
    "i don't know", "cannot help", "as an ai", "i'm not sure", "unfortunately i",
    "i cannot", "sorry", "no idea", "figure it out", "google it",
)


def _seeded_jitter(prompt, response, seed):
    """Deterministic noise in [-1, +1] derived from the text + a seed.

    Uses a hash so the 'noise' is stable per (prompt, response, seed) — that is
    what makes the stability probe reproducible AND non-constant across runs
    (vary `seed`, get a different but deterministic perturbation).
    """
    h = hashlib.sha256(f"{seed}|{prompt}|{response}".encode("utf-8")).hexdigest()
    rng = random.Random(int(h[:16], 16))
    return rng.uniform(-1.0, 1.0)


def mock_score(prompt, response, seed=0):
    """Heuristic helpfulness score in 1..5 (float internally) for the mock judge.

    Signal sources (all cheap, all stdlib):
      + length in a 'reasonable answer' band
      + presence of helpful-looking tokens
      - presence of refusal / filler tokens
      + seeded jitter (controlled noise -> drives the stability probe)
    """
    text = (response or "").lower()
    n_words = len(text.split())

    base = 3.0  # start at "adequate"

    good = sum(1 for s in _GOOD_SIGNALS if s in text)
    bad = sum(1 for s in _BAD_SIGNALS if s in text)

    # length band: empty is strongly penalized; short answers depend on whether
    # they carry any helpful signal (a terse-but-correct "Use len(s)." has a
    # good-signal token; a vague "Dicts are hard." has none -> penalize).
    if n_words == 0:
        base -= 3.0
    elif n_words < 8:
        base += 0.4 if good > 0 else -0.9
    elif n_words > 250:
        base -= 0.4  # rambling
    elif 12 <= n_words <= 180:
        base += 0.5  # a real answer

    base += min(good, 4) * 0.5
    base -= min(bad, 3) * 1.4

    # "filler": keyword-sprinkled but vacuous. Two tells:
    #   (a) vague hedging ("it depends", "many ways", "varies"), and
    #   (b) repetitive scaffolding ("you can use ..." chanted several times)
    #       which inflates the good-signal count without adding real content.
    hedges = sum(
        text.count(p)
        for p in ("it depends", "many ways", "really depends", "depending on",
                  "varies", "the situation", "for instance", "here is the thing")
    )
    you_can_use = text.count("you can use") + text.count("you can ")
    if n_words > 14 and (hedges >= 1 or you_can_use >= 3):
        base -= 2.4

    base += 0.7 * _seeded_jitter(prompt, response, seed)
    return max(1.0, min(5.0, base))


def offline_judge(prompt, response, seed=0):
    """Deterministic seeded mock verdict — no key, no network, no anthropic."""
    raw_score = mock_score(prompt, response, seed=seed)
    score = int(round(raw_score))
    score = max(1, min(5, score))
    label = "pass" if score >= PASS_THRESHOLD else "fail"
    return normalize_verdict(
        {
            "score": score,
            "label": label,
            "reason": f"mock heuristic score {raw_score:.2f} (seed={seed})",
        }
    )


def live_judge(prompt, response, seed=0):
    """Live Anthropic judge. Lazy-imports anthropic; validates structured JSON.

    `seed` is unused server-side (the API has no seed param) but kept in the
    signature so callers can swap offline<->live without changing call sites.
    Stability in live mode comes from temperature>0, not a seed.
    """
    import json

    import anthropic  # lazy: only imported on the live path

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    user_block = (
        f"{RUBRIC}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"RESPONSE:\n{response}\n"
    )

    msg = client.messages.create(
        model=LIVE_MODEL,
        max_tokens=512,
        temperature=1.0,  # >0 so repeated calls vary -> real stability probe
        messages=[{"role": "user", "content": user_block}],
        output_config={
            "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}
        },
    )
    text = next((b.text for b in msg.content if b.type == "text"), "{}")
    raw = json.loads(text)

    # Optional second line of defense: validate with pydantic if available.
    # Lazy + best-effort; normalize_verdict already clamps/repairs.
    try:
        from pydantic import BaseModel, ValidationError  # lazy, live-only

        class _Verdict(BaseModel):
            score: int
            label: str
            reason: str

        try:
            _Verdict(**raw)
        except ValidationError:
            pass  # fall through to normalize_verdict's repair
    except ImportError:
        pass

    return normalize_verdict(raw)


def get_judge(mode=None):
    """Return a judge callable (prompt, response, seed) -> verdict dict.

    mode in {"offline", "live", None}. None -> auto: live iff ANTHROPIC_API_KEY
    is set, else offline. The offline path NEVER imports anthropic/pydantic.
    """
    if mode is None:
        mode = "live" if os.environ.get("ANTHROPIC_API_KEY") else "offline"
    if mode == "live":
        return live_judge
    if mode == "offline":
        return offline_judge
    raise ValueError(f"unknown judge mode: {mode!r}")
