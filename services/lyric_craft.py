"""Pronunciation-aware fit checks. These measure constraints, not artistic worth."""
import re
from functools import lru_cache
import cmudict

CLICHES = ("ride or die", "grind never stops", "on top of the world", "make it rain",
           "chasing dreams", "neon dreams", "shadows dance", "echoes of my soul",
           "tapestry of", "symphony of", "heart of gold", "reaching for the stars")
_SLANG = {"tryna": ["T", "R", "AY1", "N", "AH0"], "finna": ["F", "IH1", "N", "AH0"],
          "imma": ["AY1", "M", "AH0"], "gon": ["G", "AO1", "N"]}
VOWEL_FAMILIES = {"IY": "ee", "IH": "ih", "EH": "eh", "EY": "ay", "AE": "ah",
                  "AA": "ah", "AO": "aw", "OW": "oh", "UW": "oo", "UH": "oo",
                  "AH": "uh", "ER": "uh", "AY": "eye", "AW": "ow", "OY": "oy"}


@lru_cache(maxsize=1)
def dictionary():
    return cmudict.dict()


def tokens(line):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", line.lower().replace("\u2019", "'"))


@lru_cache(maxsize=8192)
def pronunciations(word):
    key = "".join(tokens(word))
    return dictionary().get(key) or ([_SLANG[key]] if key in _SLANG else [])


def count_syllables(word):
    phones = pronunciations(word)
    if phones:
        return min(sum(p[-1:].isdigit() for p in variant) for variant in phones)
    w = "".join(tokens(word))
    if not w:
        return 0
    count = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not re.search(r"[^aeiouy]le$", w) and count > 1:
        count -= 1
    return max(1, count)


def line_syllable_options(line):
    totals = {0}
    for word in tokens(line):
        variants = pronunciations(word)
        counts = {sum(p[-1:].isdigit() for p in v) for v in variants} or {count_syllables(word)}
        totals = {a + b for a in totals for b in counts}
    return totals


def _rime(phones):
    stressed = [i for i, p in enumerate(phones) if p.endswith(("1", "2"))]
    if not stressed:
        return ()
    return tuple(re.sub(r"\d", "", p) for p in phones[stressed[-1]:])


def rhyme_match(a, b):
    if a == b:
        return 0.0  # Repetition may be good writing, but isn't a new rhyme.
    left = {_rime(p) for p in pronunciations(a)} - {()}
    right = {_rime(p) for p in pronunciations(b)} - {()}
    return float(bool(left & right))


def ending_vowels(line):
    words = tokens(line)
    if not words:
        return set()
    result = set()
    for variant in pronunciations(words[-1]):
        vowels = [p for p in variant if p[-1:].isdigit()]
        if vowels:
            # The held phrase ending is the FINAL vowel, which may be
            # unstressed (lonely ends in "ee", not its stressed "oh").
            result.add(VOWEL_FAMILIES.get(re.sub(r"\d", "", vowels[-1]), "unknown"))
    return result


def clean_lyrics(text):
    text = re.sub(r"[\u2010-\u2015\u2212]", " ", str(text)).replace("\u2019", "'")
    lines = []
    for line in text.strip().splitlines():
        line = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s*)", "", line)
        line = re.sub(r"\s+", " ", line).strip().strip('"')
        if line:
            lines.append(line)
    return "\n".join(lines)


def validate_lyrics(lyrics, phrases):
    lines = [l for l in lyrics.splitlines() if l.strip()]
    issues = []
    if len(lines) != len(phrases):
        issues.append({"line": None, "code": "line_count", "expected": len(phrases)})
    for i, line in enumerate(lines):
        if i >= len(phrases):
            break
        phrase = phrases[i]
        target = int(phrase.get("target_syllables", phrase.get("syllables", 1)))
        tolerance = 0 if target <= 2 else 1
        if min(abs(n - target) for n in line_syllable_options(line)) > tolerance:
            issues.append({"line": i, "code": "syllable_fit", "target": target})
        if len(tokens(line)) > int(phrase.get("max_words", target)):
            issues.append({"line": i, "code": "word_overflow"})
        if min(line_syllable_options(line)) > int(phrase.get("max_syllables", target + tolerance)):
            issues.append({"line": i, "code": "syllable_overflow"})
        for cliche in CLICHES:
            if re.search(r"\b" + re.escape(cliche) + r"\b", line.lower()):
                issues.append({"line": i, "code": "cliche", "phrase": cliche})
        if re.search(r"^(?:verse|chorus|hook|bridge|here are|here's|certainly|as an ai)\b", line.lower()):
            issues.append({"line": i, "code": "non_lyric_text"})
    return issues


def stress_fit(line, phrase):
    slots = phrase.get("syllable_slots", [])
    if not slots:
        return .5
    lexical = []
    for word in tokens(line):
        variants = pronunciations(word)
        if not variants:
            lexical.extend([None] * count_syllables(word))
            continue
        phones = min(variants, key=lambda v: sum(p[-1:].isdigit() for p in v))
        lexical.extend(int(p[-1]) for p in phones if p[-1:].isdigit())
    # Only audible accents provide evidence; unknown pronunciations stay neutral.
    matches = [float(lexical[i] > 0) for i, slot in enumerate(slots)
               if slot.get("accent") and i < len(lexical) and lexical[i] is not None]
    return sum(matches) / len(matches) if matches else .5
