from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_URDU_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",  
        "ى": "ی",  
        "ك": "ک",  
        "ه": "ہ",  
        "ة": "ۃ",  
        "ـ": None,  
        "​": None,  
        "﻿": None,  
    }
)
_DIACRITICS_RE = re.compile("[ً-ْٰ]")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_REPEAT_RE = re.compile(r"(.)\1{3,}")  
_WS_RE = re.compile(r"\s+")
_ARABIC_SCRIPT_RE = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[A-Za-z']+")

# High-frequency Roman-Urdu function words / markers that are rare as standalone English tokens.
ROMAN_URDU_MARKERS = frozenset(
    """
    hai hain hy hay tha thi nahi nahin nhi nai ka ki ke ko ky se aur bhi bhe
    mein mai mjhe mujhe mujh hum tum tm aap apna apni apne yeh ye yah wo woh voh kya kia kyun kyu
    kab kaise kese kaisa kesa koi kuch kuj sab bohat bahut bht boht acha achha acchi achi bura buri
    yar yaar bhai jee ji raha rahi rahe kar kr karo karna krna hota hoti hote hoga hogi lekin magar
    pehle phir abhi sirf wala wali wale tou jab liye liya diya gaya gayi dekho dekh pyar
    zabardast shukriya mashallah inshallah allah sahi galat ghalat mazay maza bakwas bekar faltu
    """.split()
)
# Negation cues used by the error analysis (negation is a classic failure mode for sentiment models).
NEGATION_CUES = frozenset(
    "not no never nothing nobody none neither nor cannot can't don't doesn't didn't isn't wasn't aren't "
    "won't wouldn't shouldn't couldn't hardly barely nahi nahin nhi nai na mat".split()
) | frozenset(["نہیں", "نہ", "مت", "کبھی", "بغیر", "بالکل نہیں"])


def normalize(text: str, max_chars: int | None = None) -> str:
    """Normalise raw user text. Idempotent: normalize(normalize(x)) == normalize(x)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_URDU_CHAR_MAP).translate(_DIGITS)
    text = _DIACRITICS_RE.sub("", text)
    text = _URL_RE.sub("http", text)
    text = _MENTION_RE.sub("@user", text)
    text = _REPEAT_RE.sub(r"\1\1\1", text)
    text = _WS_RE.sub(" ", text).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] or text[:max_chars]
    return text

@dataclass(frozen=True)
class ScriptProfile:
    language: str  # "en" | "ur" | "ur-Latn" | "unknown"
    arabic_ratio: float  # fraction of letters in Arabic script
    roman_urdu_ratio: float  # fraction of Latin words that are Roman-Urdu markers
    code_mixed: bool  # both scripts present, or Roman Urdu interleaved with English

def profile_script(text: str) -> ScriptProfile:
    """Cheap, dependency-free language ID tuned for the three varieties this model supports."""
    n_ar = len(_ARABIC_SCRIPT_RE.findall(text))
    n_lat = len(_LATIN_RE.findall(text))
    letters = n_ar + n_lat
    if letters == 0:
        return ScriptProfile("unknown", 0.0, 0.0, False)
    arabic_ratio = n_ar / letters
    words = [w.lower() for w in _WORD_RE.findall(text)]
    ru_hits = sum(w in ROMAN_URDU_MARKERS for w in words)
    ru_ratio = ru_hits / len(words) if words else 0.0

    if arabic_ratio >= 0.5:
        lang = "ur"
    elif ru_ratio >= 0.2 and ru_hits >= 1:
        lang = "ur-Latn"
    else:
        lang = "en"
    both_scripts = 0.1 <= arabic_ratio <= 0.9
    roman_mixed = lang != "ur" and 0.1 <= ru_ratio <= 0.6 and len(words) >= 5
    return ScriptProfile(lang, round(arabic_ratio, 3), round(ru_ratio, 3), both_scripts or roman_mixed)

def has_negation(text: str) -> bool:
    lowered = text.lower()
    tokens = set(re.findall(r"[\w']+", lowered))
    return bool(tokens & NEGATION_CUES) or any(cue in text for cue in ("نہیں", " نہ ", " مت "))
