"""WER/CER and the Arabic normalization they depend on.

Hand-checked fixtures throughout: an error rate is the product this feature
sells, so every expected value here is one that can be verified by counting on
paper, not one recorded from whatever the implementation happened to return.

The Arabic cases are the reason normalization exists. An engine writing
"إسماعيل" where the script said "اسماعيل" made no error a listener would notice,
and counting one would make the metric measure orthography instead of
recognition. Each fold is tested on its own, so a wrong one is a named failure.
"""

import pytest

from packages.metrics import text_norm
from packages.metrics.wer import score


# --- normalization, one rule at a time -------------------------------------

@pytest.mark.parametrize(
    "raw, expected, reason",
    [
        ("مُحَمَّد", "محمد", "tashkeel carries no consonantal information"),
        ("الــــنهاردة", "النهارده", "tatweel is typographic stretching only"),
        ("أحمد", "احمد", "hamza-above alef"),
        ("إسماعيل", "اسماعيل", "hamza-below alef"),
        ("آمن", "امن", "madda alef"),
        ("مدرسة", "مدرسه", "ta-marbuta and ha are written interchangeably"),
        # Written as escapes: these characters are visually identical in most
        # fonts, so a literal here would be unreviewable and could silently be
        # the wrong codepoint.
        ("\u0639\u0644\u0649", "\u0639\u0644\u064a", "alef maksura (U+0649) folds to yeh"),
        ("\u0639\u0644\u06cc", "\u0639\u0644\u064a", "farsi yeh (U+06CC) folds to yeh"),
        ("\u0645\u06a9\u062a\u0628", "\u0645\u0643\u062a\u0628", "keheh (U+06A9) folds to kaf"),
        ("مسؤول", "مسوول", "hamza carrier waw"),
        ("شيئ", "شيي", "hamza carrier ya"),
        ("٣٧", "37", "arabic-indic digits"),
        ("۴۵", "45", "extended arabic-indic digits"),
    ],
)
def test_each_normalization_rule(raw, expected, reason) -> None:
    assert text_norm.normalize(raw) == text_norm.normalize(expected), reason


def test_normalization_is_idempotent() -> None:
    """Scoring normalizes both sides; a second pass must not keep changing it."""
    once = text_norm.normalize("إسماعيل قال ٣٧٪ مدرسة")
    assert text_norm.normalize(once) == once


def test_punctuation_removal_does_not_join_words() -> None:
    """A comma is not a word, but removing it must not merge its neighbours into
    one — that would silently shrink the reference count the WER divides by."""
    assert text_norm.tokenize("one,two.three") == ["one", "two", "three"]
    assert len(text_norm.tokenize("قال، وبعدين. تاني")) == 3


def test_decomposed_forms_are_folded() -> None:
    """A decomposed alef+hamza must fold like the composed one. Without NFC first,
    it would survive and count as an error."""
    composed, decomposed = "أحمد", "أحمد"
    assert text_norm.normalize(decomposed) == text_norm.normalize(composed)


# --- WER arithmetic --------------------------------------------------------

def test_identical_text_scores_zero() -> None:
    result = score("the quick brown fox", "the quick brown fox")
    assert result.wer == 0.0
    assert result.cer == 0.0
    assert (result.sub, result.delete, result.ins) == (0, 0, 0)


@pytest.mark.parametrize(
    "reference, hypothesis, expected_wer, counts",
    [
        # 5 reference words, 2 substituted -> 2/5
        ("the quick brown fox jumps", "the quik brown cat jumps", 0.4, (2, 0, 0)),
        # 4 reference words, 1 missing -> 1/4
        ("a b c d", "a c d", 0.25, (0, 1, 0)),
        # 3 reference words, 1 extra -> 1/3
        ("a b c", "a x b c", 1 / 3, (0, 0, 1)),
        # everything wrong: 3 substitutions over 3 words
        ("a b c", "x y z", 1.0, (3, 0, 0)),
    ],
)
def test_wer_counts_are_hand_checkable(reference, hypothesis, expected_wer, counts) -> None:
    result = score(reference, hypothesis)
    assert result.wer == pytest.approx(expected_wer)
    assert (result.sub, result.delete, result.ins) == counts


def test_wer_above_one_is_not_clamped() -> None:
    """An engine emitting more wrong words than the reference has words really
    does exceed 100%; clamping would flatter it."""
    assert score("a", "x y z").wer == pytest.approx(3.0)


def test_empty_hypothesis_is_all_deletions() -> None:
    result = score("a b c", "")
    assert result.wer == pytest.approx(1.0)
    assert result.delete == 3
    assert result.hyp_words == 0


def test_empty_reference_does_not_divide_by_zero() -> None:
    """The API rejects an empty reference, so this is a guard rather than a
    supported case — it must not raise."""
    assert score("", "anything at all").wer == 0.0


# --- the Arabic case the whole thing exists for ---------------------------

def test_orthographic_difference_is_zero_normalized_and_nonzero_raw() -> None:
    """The headline reason both rates are stored: normalization's effect has to be
    visible, not applied invisibly."""
    reference = "قال إسماعيل مدرسة ٣٧"
    hypothesis = "قال اسماعيل مدرسه 37"

    assert score(reference, hypothesis, normalized=True).wer == 0.0
    assert score(reference, hypothesis, normalized=False).wer > 0.0


def test_a_real_arabic_error_still_counts_after_normalization() -> None:
    """Normalization must not be so aggressive that genuine mistranscriptions
    vanish — that would make every engine look perfect."""
    result = score("النهاردة عندنا ثلاثة مواضيع", "النهاردة عندنا اربعة مواضيع")
    assert result.wer == pytest.approx(0.25)
    assert result.sub == 1


def test_code_switched_text_scores_both_scripts() -> None:
    result = score("عندنا ٣ مواضيع بس quick decisions",
                   "عندنا 3 مواضيع بس quick decision")
    assert result.sub == 1, "only the English plural differs after normalization"


# --- alignment (what the UI boxes) ----------------------------------------

def test_alignment_marks_substituted_words_by_hypothesis_index() -> None:
    """The UI boxes words in the text it rendered, so each op has to point at the
    hypothesis token it corresponds to."""
    result = score("the quick brown fox", "the quik brown cat")
    ops = [(step.op, step.hyp_index) for step in result.alignment]
    assert ops == [("equal", 0), ("sub", 1), ("equal", 2), ("sub", 3)]


def test_alignment_covers_every_reference_and_hypothesis_word() -> None:
    """No word may be dropped from the alignment: an unaccounted word is one the
    UI cannot render a verdict for."""
    result = score("a b c d e", "a x c e f")
    assert sum(1 for s in result.alignment if s.ref is not None) == 5
    assert sum(1 for s in result.alignment if s.hyp is not None) == 5


def test_alignment_op_counts_match_the_reported_counts() -> None:
    result = score("one two three four", "one three four five six")
    assert sum(1 for s in result.alignment if s.op == "sub") == result.sub
    assert sum(1 for s in result.alignment if s.op == "del") == result.delete
    assert sum(1 for s in result.alignment if s.op == "ins") == result.ins


def test_deletions_carry_no_hypothesis_index() -> None:
    """A deleted word was never rendered, so there is nothing to box."""
    result = score("a b c", "a c")
    deletion = next(step for step in result.alignment if step.op == "del")
    assert deletion.hyp is None
    assert deletion.hyp_index is None


# --- CER ------------------------------------------------------------------

def test_cer_is_per_reference_character() -> None:
    """"cat" -> "car": one of three characters wrong."""
    assert score("cat", "car").cer == pytest.approx(1 / 3)


def test_cer_is_finer_grained_than_wer() -> None:
    """A one-letter slip is a whole wrong word but only a small fraction of the
    characters — which is the reason both are reported."""
    result = score("transcription", "transcriptoin")
    assert result.wer == pytest.approx(1.0)
    assert result.cer < 0.3
