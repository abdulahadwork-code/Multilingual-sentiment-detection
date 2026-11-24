import csv
import itertools
from pathlib import Path

HERE = Path(__file__).parent
DATA = {
    "en": {
        "positive": (["I really love", "Absolutely great", "So happy with", "Highly recommend"],
                     ["this phone", "the service", "the movie", "this restaurant", "the new update", "my order"]),
        "negative": (["I hate", "Really terrible", "Very disappointed with", "Never buying again,"],
                     ["this phone", "the service", "the movie", "this restaurant", "the new update", "my order"]),
        "neutral": (["Today I checked", "The meeting is about", "I am reading about", "Here is a note on"],
                    ["this phone", "the service", "the movie", "this restaurant", "the new update", "my order"]),
    },
    "ur": {
        "positive": (["مجھے بہت پسند آیا", "بہت زبردست ہے", "میں بہت خوش ہوں", "شاندار تجربہ"],
                     ["یہ فون", "یہ سروس", "یہ فلم", "یہ ریسٹورنٹ", "نیا اپڈیٹ", "میرا آرڈر"]),
        "negative": (["بالکل بکواس ہے", "بہت برا تجربہ", "میں بہت مایوس ہوں", "دوبارہ کبھی نہیں"],
                     ["یہ فون", "یہ سروس", "یہ فلم", "یہ ریسٹورنٹ", "نیا اپڈیٹ", "میرا آرڈر"]),
        "neutral": (["آج میں نے دیکھا", "اجلاس کے بارے میں", "میں پڑھ رہا ہوں", "ایک نوٹ"],
                    ["یہ فون", "یہ سروس", "یہ فلم", "یہ ریسٹورنٹ", "نیا اپڈیٹ", "میرا آرڈر"]),
    },
    "ur-Latn": {
        "positive": (["bohat acha hai", "zabardast hai yaar", "mujhe bahut pasand aya", "kya baat hai"],
                     ["yeh phone", "yeh service", "yeh movie", "yeh restaurant", "naya update", "mera order"]),
        "negative": (["bilkul bakwas hai", "bohat bura hai", "faltu cheez hai", "dobara kabhi nahi"],
                     ["yeh phone", "yeh service", "yeh movie", "yeh restaurant", "naya update", "mera order"]),
        "neutral": (["aaj maine dekha", "is ke baare mein", "main parh raha hun", "ek note hai"],
                    ["yeh phone", "yeh service", "yeh movie", "yeh restaurant", "naya update", "mera order"]),
    },
}
NAMES = {"en": "sample_en.csv", "ur": "sample_ur.csv", "ur-Latn": "sample_roman_ur.csv"}

for lang, by_label in DATA.items():
    rows = []
    for label, (heads, tails) in by_label.items():
        for h, t in itertools.product(heads, tails):
            rows.append({"text": f"{h} {t}" if lang == "en" else f"{t} {h}", "label": label})
    with open(HERE / NAMES[lang], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(rows)
    print(lang, len(rows))
