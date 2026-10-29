#!/usr/bin/env python3
"""
message_generator.py
====================
Detection facts -> a natural sentence, in English or Bengali.

Kept out of the detection loop entirely (section 16) so wording, grammar and
language can change without touching any vision code.

    generate_message(object_name="rickshaw", position="left",
                     proximity="near", language="en")
    -> "There is a rickshaw close on your left."

    generate_message(..., language="bn")
    -> "আপনার বাম পাশে খুব কাছে একটি রিকশা আছে।"

BENGALI GRAMMAR NOTES (worth explaining in the viva)
----------------------------------------------------
Bengali uses NUMERAL CLASSIFIERS, so "a/one" is not a single word:
    একজন  (ek-jon)  - for people          -> একজন মানুষ   "a person"
    একটি  (ek-ti)   - for things/objects  -> একটি রিকশা   "a rickshaw"
Using একটি for a person is grammatically wrong and sounds jarring to a native
speaker, so `_BN_ANIMATE` marks which classes take একজন.

Word order is Subject-Object-Verb, and the verb আছে ("is/exists") goes last:
    আপনার বাম পাশে  একটি রিকশা  আছে।
    [on your left]  [a rickshaw] [is].
So the sentence is assembled tail-last, not front-first as in English.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Bengali class names.
# The 9 thesis classes come from data.yaml (`names_bn`) at runtime; this table
# is the fallback and also covers the COCO classes used by the placeholder
# model, so the Bengali demo works before Part 1 is finished.
# --------------------------------------------------------------------------- #
BN_CLASS_NAMES = {
    # --- the 9 thesis classes ---
    "person": "মানুষ",
    "pole": "খুঁটি",
    "tree": "গাছ",
    "stairs": "সিঁড়ি",
    "pothole": "গর্ত",
    "open_drain": "খোলা ড্রেন",
    "parked_vehicle": "পার্ক করা গাড়ি",
    "street_obstacle": "রাস্তার প্রতিবন্ধকতা",
    "curb": "ফুটপাতের কিনারা",
    # --- Bangladesh-specific vehicles, for when the class list grows ---
    "rickshaw": "রিকশা",
    "cng": "সিএনজি",
    "cng_auto": "সিএনজি",
    "van": "ভ্যান",
    # --- common COCO classes (placeholder model / general use) ---
    "bicycle": "সাইকেল",
    "car": "গাড়ি",
    "motorcycle": "মোটরসাইকেল",
    "bus": "বাস",
    "truck": "ট্রাক",
    "train": "ট্রেন",
    "traffic light": "ট্রাফিক বাতি",
    "stop sign": "স্টপ সাইন",
    "bench": "বেঞ্চ",
    "chair": "চেয়ার",
    "couch": "সোফা",
    "dining table": "টেবিল",
    "bed": "বিছানা",
    "toilet": "টয়লেট",
    "door": "দরজা",
    "tv": "টেলিভিশন",
    "laptop": "ল্যাপটপ",
    "cell phone": "মোবাইল ফোন",
    "bottle": "বোতল",
    "cup": "কাপ",
    "backpack": "ব্যাগ",
    "handbag": "হাতব্যাগ",
    "suitcase": "স্যুটকেস",
    "umbrella": "ছাতা",
    "dog": "কুকুর",
    "cat": "বিড়াল",
    "cow": "গরু",
    "horse": "ঘোড়া",
    "bird": "পাখি",
    "potted plant": "গাছের টব",
    "fire hydrant": "ফায়ার হাইড্রেন্ট",
    # --- remaining COCO classes ---------------------------------------------
    # Completed so the placeholder model can never emit a half-English Bengali
    # sentence like "আপনার বাম পাশে একটি boat আছে।". Most of these are
    # irrelevant to a footpath and are filtered by IGNORE_CLASSES before they
    # are ever spoken, but a name that leaks through must still be Bengali.
    "airplane": "উড়োজাহাজ",
    "boat": "নৌকা",
    "parking meter": "পার্কিং মিটার",
    "sheep": "ভেড়া",
    "elephant": "হাতি",
    "bear": "ভালুক",
    "zebra": "জেব্রা",
    "giraffe": "জিরাফ",
    "tie": "টাই",
    "frisbee": "ফ্রিসবি",
    "skis": "স্কি",
    "snowboard": "স্নোবোর্ড",
    "sports ball": "বল",
    "kite": "ঘুড়ি",
    "baseball bat": "ব্যাট",
    "baseball glove": "গ্লাভস",
    "skateboard": "স্কেটবোর্ড",
    "surfboard": "সার্ফবোর্ড",
    "tennis racket": "টেনিস র‍্যাকেট",
    "wine glass": "গ্লাস",
    "fork": "কাঁটাচামচ",
    "knife": "ছুরি",
    "spoon": "চামচ",
    "bowl": "বাটি",
    "banana": "কলা",
    "apple": "আপেল",
    "sandwich": "স্যান্ডউইচ",
    "orange": "কমলা",
    "broccoli": "ব্রকলি",
    "carrot": "গাজর",
    "hot dog": "হট ডগ",
    "pizza": "পিৎজা",
    "donut": "ডোনাট",
    "cake": "কেক",
    "mouse": "মাউস",
    "remote": "রিমোট",
    "keyboard": "কীবোর্ড",
    "microwave": "মাইক্রোওয়েভ",
    "oven": "ওভেন",
    "toaster": "টোস্টার",
    "sink": "সিংক",
    "refrigerator": "ফ্রিজ",
    "book": "বই",
    "clock": "ঘড়ি",
    "vase": "ফুলদানি",
    "scissors": "কাঁচি",
    "teddy bear": "টেডি বিয়ার",
    "hair drier": "হেয়ার ড্রায়ার",
    "toothbrush": "টুথব্রাশ",
}

# Classes that take the animate classifier একজন instead of একটি.
_BN_ANIMATE = {"person", "মানুষ"}

# English plural-only / mass nouns: "there ARE stairs", never "a stairs".
# Bengali needs no such distinction, so this affects English output only.
_EN_PLURAL_ONLY = {"stairs", "steps", "debris", "rubble", "traffic", "sand"}

# --------------------------------------------------------------------------- #
_EN_POSITION = {
    "LEFT": "on your left",
    "RIGHT": "on your right",
    "CENTER": "ahead",
}
_BN_POSITION = {
    "LEFT": "আপনার বাম পাশে",
    "RIGHT": "আপনার ডান পাশে",
    "CENTER": "সামনে",
}

_EN_PROXIMITY = {"NEAR": "close", "MEDIUM": "", "FAR": "in the distance"}
_BN_PROXIMITY = {"NEAR": "খুব কাছে", "MEDIUM": "", "FAR": "দূরে"}

# Spoken when a NEAR object sits in the CENTER zone - the one case that
# warrants an urgent tone.
_EN_WARN = "Careful!"
_BN_WARN = "সাবধান!"

_EN_CLEAR = "Path looks clear."
_BN_CLEAR = "সামনের পথ পরিষ্কার।"


# --------------------------------------------------------------------------- #
def humanise_en(class_name: str) -> str:
    """'open_drain' -> 'open drain'; 'parked_vehicle' -> 'parked vehicle'."""
    return class_name.replace("_", " ").replace("-", " ").strip().lower()


def bengali_name(class_name: str, overrides: dict | None = None) -> str:
    """
    Bengali label for a class. Priority:
      1. `overrides` (data.yaml `names_bn`, keyed by English name)
      2. the built-in table
      3. the English name, so nothing ever comes out blank
    """
    if overrides and class_name in overrides:
        return overrides[class_name]
    return BN_CLASS_NAMES.get(class_name, humanise_en(class_name))


def _bn_article(class_name: str, bn_name: str) -> str:
    """Correct Bengali numeral classifier for this noun."""
    if class_name in _BN_ANIMATE or bn_name in _BN_ANIMATE:
        return "একজন"
    return "একটি"


# --------------------------------------------------------------------------- #
def generate_message(object_name: str,
                     position: str,
                     proximity: str = "MEDIUM",
                     language: str = "en",
                     urgent: bool = False,
                     bn_overrides: dict | None = None) -> str:
    """Build one sentence describing one object."""
    position = (position or "CENTER").upper()
    proximity = (proximity or "MEDIUM").upper()

    if language == "bn":
        return _generate_bn(object_name, position, proximity, urgent, bn_overrides)
    return _generate_en(object_name, position, proximity, urgent)


# --------------------------------------------------------------------------- #
def _generate_en(object_name: str, position: str, proximity: str, urgent: bool) -> str:
    name = humanise_en(object_name)
    plural = name in _EN_PLURAL_ONLY
    article = "" if plural else ("an" if name[:1] in "aeiou" else "a")
    place = _EN_POSITION.get(position, "ahead")
    prox = _EN_PROXIMITY.get(proximity, "")

    def join(*words):
        return " ".join(w for w in words if w)

    if urgent:
        # "Careful! A pole is close ahead." / "Careful! Stairs are close ahead."
        subject = join(article.capitalize() if article else name.capitalize(),
                       name if article else "")
        verb = "are" if plural else "is"
        return "{} {}".format(_EN_WARN, join(subject, verb, "close", place) + ".")

    lead = "There are" if plural else "There is"
    return join(lead, article, name, prox, place) + "."


# --------------------------------------------------------------------------- #
def _generate_bn(object_name: str, position: str, proximity: str,
                 urgent: bool, overrides: dict | None) -> str:
    name = bengali_name(object_name, overrides)
    article = _bn_article(object_name, name)
    place = _BN_POSITION.get(position, "সামনে")
    prox = _BN_PROXIMITY.get(proximity, "")

    # Subject-Object-Verb: [place] [proximity] [article] [noun] আছে।
    pieces = [place]
    if prox:
        pieces.append(prox)
    pieces.append(article)
    pieces.append(name)
    sentence = " ".join(pieces) + " আছে।"

    if urgent:
        return "{} {}".format(_BN_WARN, sentence)
    return sentence


# --------------------------------------------------------------------------- #
def _phrase_en(item, bn_overrides=None) -> str:
    """Short clause for a combined sentence: 'a chair ahead'."""
    name = humanise_en(item.get("class_name", "object"))
    plural = name in _EN_PLURAL_ONLY
    article = "" if plural else ("an" if name[:1] in "aeiou" else "a")
    place = _EN_POSITION.get((item.get("position") or "CENTER").upper(), "ahead")
    prox = _EN_PROXIMITY.get((item.get("proximity") or "MEDIUM").upper(), "")
    return " ".join(w for w in (article, name, prox, place) if w)


def _phrase_bn(item, bn_overrides=None) -> str:
    """Short clause for a combined Bengali sentence: 'সামনে একটি চেয়ার'."""
    cls = item.get("class_name", "object")
    name = bengali_name(cls, bn_overrides)
    article = _bn_article(cls, name)
    place = _BN_POSITION.get((item.get("position") or "CENTER").upper(), "সামনে")
    prox = _BN_PROXIMITY.get((item.get("proximity") or "MEDIUM").upper(), "")
    return " ".join(w for w in (place, prox, article, name) if w)


def generate_multi_message(items: list[dict], language: str = "en",
                           bn_overrides: dict | None = None,
                           combine: bool = True) -> str:
    """
    One utterance describing the prioritised objects.

    With `combine` (the default) several objects become ONE sentence:

        EN: "There is a person on your left, a chair ahead and a rickshaw on
             your right."
        BN: "বাম পাশে একজন মানুষ, সামনে একটি চেয়ার এবং আপনার ডান পাশে একটি
             রিকশা আছে।"

    rather than three separate sentences read back to back. For a listener who
    cannot see the scene, one flowing sentence is markedly quicker to take in
    than three restarts of "There is ...", and it keeps the objects in
    priority order in a single breath.

    Bengali is Subject-Object-Verb, so the clauses are joined first and the
    single verb আছে closes the whole sentence - repeating আছে per clause is
    what makes machine-translated Bengali sound robotic.

    A single item falls through to the ordinary one-object sentence, and an
    urgent leading item keeps its warning prefix either way.
    """
    if not items:
        return clear_path_message(language)

    if len(items) == 1 or not combine:
        parts = [generate_message(
            object_name=it.get("class_name", "object"),
            position=it.get("position", "CENTER"),
            proximity=it.get("proximity", "MEDIUM"),
            language=language,
            urgent=bool(it.get("urgent")) and i == 0,
            bn_overrides=bn_overrides,
        ) for i, it in enumerate(items)]
        return " ".join(parts)

    urgent = bool(items[0].get("urgent"))

    if language == "bn":
        clauses = [_phrase_bn(it, bn_overrides) for it in items]
        joined = (clauses[0] if len(clauses) == 1
                  else ", ".join(clauses[:-1]) + " এবং " + clauses[-1])
        sentence = joined + " আছে।"
        return "{} {}".format(_BN_WARN, sentence) if urgent else sentence

    clauses = [_phrase_en(it) for it in items]
    joined = (clauses[0] if len(clauses) == 1
              else ", ".join(clauses[:-1]) + " and " + clauses[-1])
    sentence = "There is " + joined + "."
    return "{} {}".format(_EN_WARN, sentence) if urgent else sentence


def clear_path_message(language: str = "en") -> str:
    return _BN_CLEAR if language == "bn" else _EN_CLEAR
