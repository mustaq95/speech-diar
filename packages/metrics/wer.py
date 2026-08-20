"""Word and character error rates, with the alignment behind them.

Levenshtein with a backtrace, hand-written rather than pulled from `jiwer`. That
is not reinvention for its own sake: `jiwer` applies its own transform pipeline,
which is built for English and would fight
`packages/metrics/text_norm.py` — and the alignment it exposes is not the
per-word op list the UI needs to box the substituted words. Twenty lines of DP
with a backtrace gives both, with no dependency and no second opinion about what
normalization means.

The returned `alignment` is what the UI highlights. Those boxes are edit
operations against the reference — a text comparison. No aligner, no audio, no
per-word timings are involved.
"""

from dataclasses import dataclass, field
from typing import Literal

from packages.metrics.text_norm import normalize, tokenize

Op = Literal["equal", "sub", "del", "ins"]


@dataclass(frozen=True)
class AlignmentStep:
    """One edit operation. `hyp_index` points into the hypothesis token list so
    the UI can mark the word it actually rendered."""

    op: Op
    ref: str | None = None
    hyp: str | None = None
    hyp_index: int | None = None


@dataclass(frozen=True)
class ErrorRates:
    """One engine's scores against one reference.

    `wer` can exceed 1.0. That is not a bug to clamp: an engine that emits more
    wrong words than the reference has words has an error rate above 100%, and
    hiding that would flatter it.
    """

    wer: float
    cer: float
    sub: int
    delete: int
    ins: int
    ref_words: int
    hyp_words: int
    alignment: list[AlignmentStep] = field(default_factory=list)


def _levenshtein_backtrace(ref: list[str], hyp: list[str]) -> tuple[int, int, int, list[AlignmentStep]]:
    """Edit counts plus the operation path, over token lists.

    Full DP table rather than the two-row trick, because the backtrace needs the
    whole table. Reference lengths here are hundreds of words, so the memory is
    irrelevant next to being able to say WHICH words were wrong.
    """
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],  # substitute
                    cost[i - 1][j],      # delete (ref word missing from hyp)
                    cost[i][j - 1],      # insert (hyp word not in ref)
                )

    steps: list[AlignmentStep] = []
    subs = dels = inses = 0
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            steps.append(AlignmentStep("equal", ref[i - 1], hyp[j - 1], j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            steps.append(AlignmentStep("sub", ref[i - 1], hyp[j - 1], j - 1))
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            steps.append(AlignmentStep("del", ref[i - 1], None, None))
            dels += 1
            i -= 1
        else:
            steps.append(AlignmentStep("ins", None, hyp[j - 1], j - 1))
            inses += 1
            j -= 1
    steps.reverse()
    return subs, dels, inses, steps


def _char_error_rate(ref: str, hyp: str) -> float:
    """Edit distance per reference character. Distance only — CER has no
    alignment consumer, so this uses the two-row form and stays O(min(n,m)) in
    memory over strings that can be tens of thousands of characters."""
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_char in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_char != hyp_char),
            )
        previous = current
    return previous[-1] / len(ref)


def score(reference: str, hypothesis: str, *, normalized: bool = True) -> ErrorRates:
    """Score one hypothesis against one reference.

    An empty reference gives 0.0 rather than dividing by zero, and the caller is
    expected not to score against one at all — the API rejects an empty reference
    for exactly this reason, since every engine would otherwise look perfect or
    infinitely wrong depending on the guard.
    """
    ref_tokens = tokenize(reference, normalized=normalized)
    hyp_tokens = tokenize(hypothesis, normalized=normalized)
    subs, dels, inses, steps = _levenshtein_backtrace(ref_tokens, hyp_tokens)
    ref_count = len(ref_tokens)
    wer = (subs + dels + inses) / ref_count if ref_count else 0.0

    ref_chars = normalize(reference) if normalized else reference.strip()
    hyp_chars = normalize(hypothesis) if normalized else hypothesis.strip()
    return ErrorRates(
        wer=wer,
        cer=_char_error_rate(ref_chars, hyp_chars),
        sub=subs,
        delete=dels,
        ins=inses,
        ref_words=ref_count,
        hyp_words=len(hyp_tokens),
        alignment=steps,
    )
