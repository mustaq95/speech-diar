"""Text normalization for scoring, and for scoring only.

WER on Arabic is meaningless without normalization. The same word is routinely
written with or without short vowels, with any of four alef forms, with ta-marbuta
or ha, with Arabic-Indic or ASCII digits. An engine that writes "إسماعيل" where
the script said "اسماعيل" made no error a listener would recognise, and counting
one would make the metric measure orthography instead of recognition.

**Nothing here ever touches stored text.** `TranscriptResult.text` keeps exactly
what the engine emitted; these functions run on a copy at scoring time. Both the
normalized and the raw error rates are stored, so normalization's own effect stays
visible rather than becoming invisible preprocessing.

Each rule is its own named function, deliberately. A single mega-regex would be
shorter and untestable; a wrong fold in one of these is one failing unit test
rather than an argument about a character class.
"""

import re
import unicodedata

#: Short vowels, sukun, shadda, and the rest of the combining marks that carry no
#: consonantal information. Written as a range rather than listed so the
#: superscript alef and Quranic marks are covered too.
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")

#: Kashida: pure typographic stretching, never phonemic.
_TATWEEL = re.compile(r"ـ")

_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"})
#: Alef maksura (U+0649) and Farsi yeh (U+06CC) both fold to Arabic yeh
#: (U+064A). The Farsi form is not exotic: Persian/Urdu keyboard layouts and some
#: fonts emit it inside otherwise-Arabic text, and an engine producing one where
#: the reference has the other has not misrecognised anything.
_YA_FORMS = str.maketrans({"\u0649": "\u064a", "\u06cc": "\u064a"})

#: Keheh (U+06A9) folds to Arabic kaf (U+0643) — the same substitution problem as
#: the yeh above, from the same mixed-locale sources.
_KAF_FORMS = str.maketrans({"\u06a9": "\u0643"})
_TA_MARBUTA = str.maketrans({"ة": "ه"})
_HAMZA_FORMS = str.maketrans({"ؤ": "و", "ئ": "ي"})

#: Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits to ASCII, so "٣٧"
#: and "37" are the same number.
_DIGITS = str.maketrans(
    {chr(0x0660 + i): str(i) for i in range(10)}
    | {chr(0x06F0 + i): str(i) for i in range(10)}
)

#: Punctuation, Arabic and Latin alike. Removed rather than spaced out: a comma
#: is not a word, and turning it into one would inflate the reference count.
_PUNCTUATION = re.compile(
    r"[.,;:!?\"'`~^*_\-–—()\[\]{}<>/\\|+=%&@#$…«»“”‘’،؛؟٪-٭۔]"
)

_WHITESPACE = re.compile(r"\s+")


def strip_tashkeel(text: str) -> str:
    """Remove short vowels and other combining marks."""
    return _TASHKEEL.sub("", text)


def strip_tatweel(text: str) -> str:
    """Remove kashida stretching characters."""
    return _TATWEEL.sub("", text)


def fold_alef(text: str) -> str:
    """أ إ آ -> ا. Hamza placement on alef is not consistently written."""
    return text.translate(_ALEF_FORMS)


def fold_ya(text: str) -> str:
    """Alef maksura and Farsi yeh -> Arabic yeh."""
    return text.translate(_YA_FORMS)


def fold_kaf(text: str) -> str:
    """Keheh -> Arabic kaf."""
    return text.translate(_KAF_FORMS)


def fold_ta_marbuta(text: str) -> str:
    """ة -> ه. Word-final, the two are written interchangeably."""
    return text.translate(_TA_MARBUTA)


def fold_hamza_carriers(text: str) -> str:
    """ؤ -> و, ئ -> ي. Same consonant, hamza carrier dropped."""
    return text.translate(_HAMZA_FORMS)


def normalize_digits(text: str) -> str:
    """Arabic-Indic and Extended Arabic-Indic digits -> ASCII."""
    return text.translate(_DIGITS)


def strip_punctuation(text: str) -> str:
    """Remove punctuation without joining the words either side of it."""
    return _PUNCTUATION.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Apply every rule, in the order they must run.

    Order is load-bearing in one place: NFC composition first, so a decomposed
    "أ" (alef + hamza-above) becomes the single codepoint `fold_alef` knows about.
    Skipping that would leave decomposed forms unfolded and quietly count them as
    errors.
    """
    text = unicodedata.normalize("NFC", text)
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = fold_alef(text)
    text = fold_ya(text)
    text = fold_kaf(text)
    text = fold_ta_marbuta(text)
    text = fold_hamza_carriers(text)
    text = normalize_digits(text)
    text = strip_punctuation(text)
    return collapse_whitespace(text.casefold())


def tokenize(text: str, *, normalized: bool = True) -> list[str]:
    """Words for scoring. Whitespace-split, because that is what a word count is."""
    source = normalize(text) if normalized else collapse_whitespace(text)
    return source.split()
