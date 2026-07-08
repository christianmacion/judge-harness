"""pairwise.py — A/B pairwise judging with a position-bias check.

LLM judges have a well-documented position bias: they tend to prefer whichever
candidate is shown FIRST (or second), independent of quality. This module
measures that. For each (prompt, A, B):

  1. Ask the judge which is better with order (A, B).
  2. Ask again with the order swapped (B, A).
  3. A consistent judge picks the SAME underlying candidate both times.
     A flip (it picks "first slot" both times -> picks A then B) reveals
     position bias.

flip_rate = fraction of pairs where the two orderings disagree about the winner.
0.0 = order-invariant (good). High = the judge is voting on position, not merit.

Pure stdlib. Uses the same judge callable (offline mock or live) by scoring each
candidate and comparing scores; a small seeded slot-bias is injected in the mock
so the probe surfaces a realistic, non-zero flip rate.
"""

from judge.judge import mock_score


def _pick_winner(judge, prompt, first, second, seed, slot_bias):
    """Return 'first' or 'second' — which slot the judge prefers.

    For the offline mock we score both candidates and add a small advantage to
    the FIRST slot (slot_bias) to emulate position bias. For a live judge you'd
    instead prompt for a single A/B choice; here we keep one code path by
    scoring, which is sufficient to exercise the flip-rate metric offline.
    """
    s_first = mock_score(prompt, first, seed=seed) + slot_bias
    s_second = mock_score(prompt, second, seed=seed)
    return "first" if s_first >= s_second else "second"


def position_bias_check(judge, pairs, seed=0, slot_bias=0.35):
    """Measure position-bias flip rate over A/B `pairs`.

    pairs: list of dicts with "prompt", "a", "b".
    Returns {"n_pairs", "flips", "flip_rate", "details": [...]}.

    For each pair we run the judge in order (A,B) and (B,A). We translate each
    slot-pick back to the real candidate ("A"/"B"). A flip = the two runs name
    different real winners.
    """
    flips = 0
    details = []
    for p in pairs:
        prompt, a, b = p["prompt"], p["a"], p["b"]

        # Order 1: (A, B)
        pick1 = _pick_winner(judge, prompt, a, b, seed, slot_bias)
        winner1 = "A" if pick1 == "first" else "B"

        # Order 2: (B, A)  -> "first" now means B
        pick2 = _pick_winner(judge, prompt, b, a, seed, slot_bias)
        winner2 = "B" if pick2 == "first" else "A"

        flipped = winner1 != winner2
        flips += int(flipped)
        details.append(
            {
                "prompt": prompt,
                "winner_ab": winner1,
                "winner_ba": winner2,
                "flipped": flipped,
            }
        )

    n = len(pairs)
    return {
        "n_pairs": n,
        "flips": flips,
        "flip_rate": (flips / n) if n else 0.0,
        "details": details,
    }


def default_pairs(items, limit=12):
    """Build A/B pairs from a labelled item set for a quick demo probe.

    To make the position-bias probe meaningful we pair candidates of *similar*
    merit (both 'pass', adjacent in the file). When A and B are close in quality,
    a judge with no position bias should still pick the same one in both orders;
    a biased judge flips toward whichever it saw first. Pairing wildly different
    candidates (great vs terrible) would mask bias, since the quality gap
    overwhelms the slot advantage.
    """
    passes = [it for it in items if it.get("human_label") == "pass"]
    pairs = []
    # adjacent pass/pass pairs -> close merit -> bias is observable
    for a, b in zip(passes[0::2], passes[1::2]):
        pairs.append({"prompt": a["prompt"], "a": a["response"], "b": b["response"]})
        if len(pairs) >= limit:
            break
    return pairs
