import pytest
from xlmr_sentiment.text import has_negation, normalize, profile_script

def test_urdu_character_unification():
    assert normalize("كيا هے") == "کیا ہے"

def test_diacritics_tatweel_and_digits_removed():
    assert normalize("بَہُت اچھـــا ۱۲۳") == "بہت اچھا 123"

def test_urls_mentions_repeats_whitespace():
    assert normalize("  @ali   sooooo good!!!!!! https://x.co/a ") == "@user sooo good!!! http"

@pytest.mark.parametrize("text", ["یہ فلم اچھی تھی", "This is GREAT!!!!", "yeh @bob acha   hai http://a.b"])
def test_normalize_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once

def test_truncation_respects_word_boundary():
    out = normalize("alpha beta gamma delta", max_chars=12)
    assert out == "alpha beta" and len(out) <= 12

def test_non_string_input():
    assert normalize(None) == ""
    assert normalize(123) == "123"

@pytest.mark.parametrize("text,lang", [
    ("The movie was fantastic and the cast was great", "en"),
    ("یہ فلم بہت اچھی تھی", "ur"),
    ("yeh movie bohat achi thi yaar", "ur-Latn"),
    ("1234 !!!", "unknown"),
])
def test_language_profile(text, lang):
    assert profile_script(text).language == lang

def test_code_mixing_detected():
    assert profile_script("This phone is بہت اچھا really").code_mixed
    assert profile_script("Honestly the service was okay lekin delivery bohat late thi").code_mixed
    assert not profile_script("The service was okay but the delivery was late").code_mixed

@pytest.mark.parametrize("text,neg", [
    ("I don't like it", True), ("yeh acha nahi hai", True), ("یہ اچھا نہیں ہے", True), ("I like it", False),
])
def test_negation(text, neg):
    assert has_negation(text) is neg
