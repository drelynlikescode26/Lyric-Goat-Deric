import re
import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Phonetic mapping ──────────────────────────────────────────────────────────
PHONETIC_MAP = {
    "ayy": ["day", "pain", "same", "late", "way", "stay", "rain", "chains", "fade", "came", "name", "frame"],
    "oh":  ["go", "alone", "home", "road", "know", "show", "grow", "cold", "soul", "hold", "fold", "gold"],
    "ee":  ["free", "see", "believe", "breathe", "need", "real", "feel", "deep", "sleep", "leave", "bleed"],
    "ah":  ["heart", "far", "hard", "dark", "start", "apart", "scar", "star", "arm", "charge", "guard"],
    "uh":  ["love", "above", "enough", "blood", "flood", "stuck", "rough", "cut", "run", "done", "trust"],
    "ii":  ["high", "die", "cry", "fly", "try", "lie", "side", "mind", "time", "right", "night", "light"],
}

# ── Configurable scoring weights ──────────────────────────────────────────────
# Edit these without touching generation logic.
SCORE_WEIGHTS = {
    "syllable_fit":   0.35,
    "word_count_fit": 0.15,
    "stress_fit":     0.15,
    "rhyme":          0.15,
    "vowel_affinity": 0.05,
    "singability":    0.10,
    "density_fit":    0.05,
}

# Overflow penalty applied when line exceeds max_words+1 OR max_syllables+1
OVERFLOW_PENALTY = 0.45   # multiply total score by this if any line overflows

STYLE_DESCRIPTIONS = {
    "tone": {
        "melodic":    "smooth, sung, emotional — words flow like a melody",
        "aggressive": "hard-hitting, intense — every word lands like a punch",
        "simple":     "clean, clear, easy to follow — nothing overcomplicated",
        "punchlines": "clever wordplay, double meanings, bars that make people rewind",
    },
    "mode": {
        "hook":  "a catchy, repeatable hook/chorus — short lines, memorable, singable",
        "verse": "a full verse — storytelling, building energy, detailed",
        "story": "narrative, cinematic — paint a picture with words",
    },
    "vibe": {
        "sad":           "emotional, introspective, real pain — like crying in a car at night",
        "hype":          "energetic, motivating — make the crowd go crazy",
        "introspective": "deep, thoughtful, self-aware — looking inward",
        "love":          "romantic, vulnerable, genuine feeling",
    },
}

STYLE_VARIANTS = [
    {"name": "melodic", "label": "Melodic Version", "description": "smooth and singable — built to float over a melody"},
    {"name": "rap",     "label": "Rap Version",     "description": "bars and cadence — built to be rapped, rhythm-locked"},
    {"name": "punchy",  "label": "Punchy Version",  "description": "short, hard-hitting lines — maximum impact, minimum words"},
]

# Vowel hint → word suggestions
_VOWEL_HINT_WORDS = {
    "bright": ["way", "free", "high", "feel", "right", "stay", "real", "sky"],
    "dark":   ["home", "dark", "blood", "hold", "heart", "far", "soul", "gone"],
    "mixed":  [],
    "neutral": [],
}


# ── Syllable counting ─────────────────────────────────────────────────────────

def _count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


# ── Phonetic anchor extraction ────────────────────────────────────────────────

def _ending_sound(word: str) -> str:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    m = re.search(r"[aeiouy][^aeiouy]*$", word)
    return "-" + (m.group() if m else (word[-2:] if len(word) >= 2 else word))


def _extract_phonetic_anchors(phrase_map: list) -> list:
    anchors = []
    for phrase in phrase_map:
        words = phrase.get("words", []) or phrase.get("text", "").split()
        last = words[-1] if words else ""
        anchors.append(_ending_sound(str(last)))
    return anchors


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """\
You are a world-class songwriter and ghostwriter. Grammy-winning credits across \
hip-hop, R&B, trap, and pop. You don't write filler. Every bar has intent.

YOUR CRAFT:
- Emotion first, words second
- Specific, concrete images ("streetlight on a Tuesday" not "the city at night")
- Layer meaning — a line about a girl can be about ambition
- Internal rhymes, assonance, alliteration — not just end rhymes
- Banned clichés: "ride or die", "real ones", "on top of the world", \
"grind never stops", "make it rain", "all eyes on me", "can't stop won't stop"
- Singability > complexity. A line that lands beats one that impresses.

HARD PERFORMANCE RULES (never violate):
1. Output exactly one line per bar — same count, same order
2. Hit each bar's syllable count (±1 max)
3. NEVER exceed the [max N words] cap — overstuffed lines don't fit
4. [sustained] bars: use 2–3 long held words, NO short staccato phrasing
5. [breathe after] bars: the phrase needs silence to land — don't rush it
6. No labels, headers, numbering, or explanation — pure lyrics only"""


# ── Per-phrase prompt block ───────────────────────────────────────────────────

def _phrase_block(phrase: dict, idx: int, anchor: str, gen_mode: str, melody_mode: bool) -> str:
    """Build the constraint spec for a single bar in the prompt."""
    syl          = phrase.get("syllables", 4)
    max_w        = phrase.get("max_words", 5)
    max_s        = phrase.get("max_syllables", syl + 1)
    pitch        = phrase.get("pitch_symbol", "")
    density      = phrase.get("density_label", "mid")
    energy       = phrase.get("energy_level", "mid")
    conf_lbl     = phrase.get("confidence_label", "med")
    lw           = phrase.get("literal_weight", 0.55)
    sustain      = phrase.get("is_sustained", False)
    svr          = phrase.get("sustained_vowel_ratio", 0)
    wlp          = phrase.get("word_length_profile", "any")
    pause_af     = phrase.get("pause_after", 0)
    vowel_hint   = phrase.get("vowel_family_hint", "neutral")
    text         = phrase.get("text", "")

    parts = [f"  Bar {idx + 1}:"]

    # Source text or rhythm-only based on literal weight
    force_cadence = melody_mode or gen_mode == "cadence" or lw <= 0.25
    if force_cadence or not text:
        parts.append("(rhythm)")
    else:
        # Partial literal: include text only if lw is high enough
        if lw >= 0.70:
            parts.append(f'"{text}"')
        else:
            parts.append(f'(~"{text}")')  # tilde = rough reference, not literal

    parts.append(f"→ {syl} syl  [max {max_w} words | max {max_s} syl]")

    # Audio metadata
    meta = []
    if pitch:
        meta.append(pitch)
    if density:
        meta.append(density)
    if energy != "mid":
        meta.append(f"{energy} energy")
    if meta:
        parts.append(f"[{' | '.join(meta)}]")

    # Constraint hints
    if sustain or (svr >= 0.55 and not text):
        parts.append("[sustained — hold 2–3 words, no fast syllables]")
    if wlp == "short":
        parts.append("[short words only ≤2 syl each]")
    elif wlp == "mixed":
        parts.append("[prefer shorter words]")
    if pause_af > 0.6:
        parts.append("[breathe after]")

    # Rhyme anchor
    if anchor:
        parts.append(f"[rhyme: {anchor}]")

    # Low confidence flag
    if conf_lbl == "low" and gen_mode == "literal":
        parts.append("[← cadence only, ignore words]")

    # Vowel hint — soft suggestion
    if vowel_hint in ("bright", "dark"):
        hint_words = _VOWEL_HINT_WORDS.get(vowel_hint, [])
        if hint_words:
            sample = ", ".join(hint_words[:4])
            parts.append(f"[{vowel_hint} vowel feel — words like {sample} work well]")

    return "  ".join(parts)


# ── User prompt builder ───────────────────────────────────────────────────────

def _build_user_prompt(
    rough_text: str,
    flow_data: dict,
    tone: str,
    mode: str,
    vibe: str,
    gen_mode: str,
    key: str,
    variant: dict,
    phonetic_anchors: list,
) -> str:
    tone_desc  = STYLE_DESCRIPTIONS["tone"].get(tone, tone)
    mode_desc  = STYLE_DESCRIPTIONS["mode"].get(mode, mode)
    vibe_desc  = STYLE_DESCRIPTIONS["vibe"].get(vibe, vibe)

    tempo       = flow_data.get("tempo_bpm", "?")
    tempo_str   = f"{tempo:.0f}" if isinstance(tempo, float) else str(tempo)
    flow_style  = flow_data.get("flow_style", "mixed")
    phrase_map  = flow_data.get("phrase_map", [])
    melody_mode = flow_data.get("melody_mode", False)
    vowel_family = flow_data.get("vowel_family")
    detected_key = flow_data.get("detected_key", key)
    key_used     = key if key and key != "auto" else detected_key

    # Per-phrase constraint blocks
    phrase_blocks = "\n".join(
        _phrase_block(phrase, i, phonetic_anchors[i] if i < len(phonetic_anchors) else "", gen_mode, melody_mode)
        for i, phrase in enumerate(phrase_map)
    ) if phrase_map else "  (no phrase data — use your judgment)"

    # Global vowel hint
    vowel_hint_str = ""
    if vowel_family and vowel_family in PHONETIC_MAP:
        sample = ", ".join(PHONETIC_MAP[vowel_family][:5])
        vowel_hint_str = (
            f"\nVOWEL FEEL: The recording leans toward '{vowel_family}' sounds. "
            f"Words like {sample} often work — use them if they fit naturally."
        )

    key_str = f"\nKEY: {key_used}" if key_used else ""

    if gen_mode == "cadence" or melody_mode:
        approach = "CADENCE MODE — ignore words entirely. Follow syllable counts and rhythm only."
        source_block = ""
    elif gen_mode == "literal":
        approach = "LITERAL MODE — stay close to the sounds mumbled. Per-bar literal weights shown above."
        source_block = f'\nARTIST MUMBLE: "{rough_text}"\n'
    else:
        approach = "CREATIVE MODE — use vibe and emotion as your guide. Words are inspiration, not instruction."
        source_block = f'\nMUMBLE REFERENCE: "{rough_text}"\n'

    return f"""Transform this vocal recording into real lyrics.

APPROACH: {approach}{source_block}
TEMPO: {tempo_str} BPM  |  FLOW: {flow_style}{key_str}{vowel_hint_str}

BAR-BY-BAR CONSTRAINTS:
{phrase_blocks}

STYLE: {tone} · {mode} · {vibe}
- Tone: {tone_desc}
- Mode: {mode_desc}
- Vibe: {vibe_desc}

Write the **{variant["label"]}** ({variant["description"]}):

OUTPUT: one line per bar, in order, nothing else."""


# ── Single-line regeneration ──────────────────────────────────────────────────

def generate_single_line(
    bar_index: int,
    syllable_count: int,
    context_lines: list,
    locked_lines: dict,
    rough_text: str,
    flow_data: dict,
    tone: str,
    mode: str,
    vibe: str,
    gen_mode: str,
    key: str,
) -> str:
    vibe_desc    = STYLE_DESCRIPTIONS["vibe"].get(vibe, vibe)
    vowel_family = flow_data.get("vowel_family")
    detected_key = flow_data.get("detected_key", key)
    key_used     = key if key and key != "auto" else detected_key

    phrase_map = flow_data.get("phrase_map", [])
    phrase     = phrase_map[bar_index] if bar_index < len(phrase_map) else {}
    pitch_sym  = phrase.get("pitch_symbol", "")
    density    = phrase.get("density_label", "")
    conf_lbl   = phrase.get("confidence_label", "med")
    max_w      = phrase.get("max_words", 6)
    max_s      = phrase.get("max_syllables", syllable_count + 1)
    wlp        = phrase.get("word_length_profile", "any")
    sustain    = phrase.get("is_sustained", False)

    phonetic_hint = ""
    if vowel_family and vowel_family in PHONETIC_MAP:
        sample = ", ".join(PHONETIC_MAP[vowel_family][:5])
        phonetic_hint = f"\nVowel feel: '{vowel_family}' — words like {sample} may work well."

    audio_meta = ""
    meta_parts = [x for x in [pitch_sym, density] if x]
    if meta_parts:
        audio_meta = f"\nBar audio: {' | '.join(meta_parts)}"
    if sustain:
        audio_meta += "  [sustained — 2–3 held words]"
    if wlp == "short":
        audio_meta += "  [short words only]"

    context_lines_str = "\n".join(
        f'  Bar {i+1}: ← REWRITE ({syllable_count} syl | max {max_w} words | max {max_s} syl | conf: {conf_lbl})'
        if i == bar_index
        else f'  Bar {i+1}: "{line}"' + (" [LOCKED]" if str(i) in locked_lines else "")
        for i, line in enumerate(context_lines)
    )

    prompt = f"""Rewrite ONE bar of lyrics. Everything else stays the same.

CONTEXT:
{context_lines_str}

KEY: {key_used or "?"}  |  VIBE: {vibe_desc}{phonetic_hint}{audio_meta}

RULES:
- Exactly {syllable_count} syllables (±1)
- Max {max_w} words
- Flows naturally into surrounding bars
- Same tone and vibe as other lines
- Output ONLY the new line"""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=_build_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip().strip('"')


# ── Verification pass (Haiku) ─────────────────────────────────────────────────

def _verify_and_fix(versions: list, phrase_map: list) -> list:
    if not phrase_map:
        return versions
    targets = [{"bar": i + 1, "syllables": p["syllables"]} for i, p in enumerate(phrase_map)]
    versions_text = "\n\n".join(f'VERSION: {v["label"]}\n{v["lyrics"]}' for v in versions)

    prompt = f"""Syllable-accuracy editor for song lyrics.

SYLLABLE TARGETS (±1 max):
{json.dumps(targets, indent=2)}

LYRICS:
{versions_text}

Fix ONLY lines off by more than 1 syllable. Keep meaning and rhyme.
Return JSON only: {{"melodic": "...", "rap": "...", "punchy": "..."}}
Lines separated by \\n. JSON only."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", msg.content[0].text.strip())
        fixed = json.loads(raw)
        name_map = {v["name"]: v for v in versions}
        for k in ("melodic", "rap", "punchy"):
            if k in fixed and k in name_map and fixed[k].strip():
                name_map[k]["lyrics"] = fixed[k].strip()
    except Exception:
        pass
    return versions


# ── Phrase-level auto-fix ─────────────────────────────────────────────────────

def _autofix_weak_bars(
    versions: list,
    flow_data: dict,
    rough_text: str,
    tone: str,
    mode: str,
    vibe: str,
    gen_mode: str,
    key: str,
) -> list:
    phrase_map = flow_data.get("phrase_map", [])
    if not phrase_map:
        return versions

    for version in versions:
        lines = [l for l in version["lyrics"].split("\n") if l.strip()]
        changed = False
        for i, phrase in enumerate(phrase_map):
            if i >= len(lines):
                break
            target = phrase["syllables"]
            actual = sum(_count_syllables(w) for w in lines[i].split())
            if abs(actual - target) > 2:
                candidate = generate_single_line(
                    i, target, lines, {}, rough_text,
                    flow_data, tone, mode, vibe, gen_mode, key,
                )
                new_syl = sum(_count_syllables(w) for w in candidate.split())
                if abs(new_syl - target) < abs(actual - target):
                    lines[i] = candidate
                    changed = True
        if changed:
            version["lyrics"] = "\n".join(lines)
    return versions


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_lyrics(lyrics: str, flow_data: dict) -> tuple[float, dict]:
    """
    Returns (total_score, breakdown_dict).
    breakdown_dict keys match SCORE_WEIGHTS.
    """
    lines = [l for l in lyrics.split("\n") if l.strip()]
    if not lines:
        return 0.0, {}

    phrase_map   = flow_data.get("phrase_map", [])
    vowel_family = flow_data.get("vowel_family")
    pairs = min(len(lines), len(phrase_map))

    # 1. Syllable fit
    if phrase_map and pairs:
        diffs = [
            max(0.0, 1.0 - abs(
                sum(_count_syllables(w) for w in lines[i].split()) - phrase_map[i]["syllables"]
            ) / max(phrase_map[i]["syllables"], 1))
            for i in range(pairs)
        ]
        syllable_fit = sum(diffs) / pairs
    else:
        syllable_fit = 0.5

    # 2. Word count fit (vs max_words cap)
    if phrase_map and pairs:
        wc_fits = []
        for i in range(pairs):
            max_w = phrase_map[i].get("max_words", 6)
            actual_w = len(lines[i].split())
            if actual_w <= max_w:
                wc_fits.append(1.0)
            elif actual_w == max_w + 1:
                wc_fits.append(0.6)
            else:
                wc_fits.append(0.1)  # hard miss
        word_count_fit = sum(wc_fits) / len(wc_fits)
    else:
        word_count_fit = 0.5

    # 3. Stress fit
    stress_fit = _stress_fit_score(lines, phrase_map)

    # 4. Rhyme density
    last_words = [l.strip().split()[-1].lower().rstrip(".,!?") for l in lines if l.strip().split()]
    rhyme_pairs = sum(
        1 for i in range(len(last_words) - 1)
        if len(last_words[i]) >= 2 and len(last_words[i + 1]) >= 2
        and last_words[i][-2:] == last_words[i + 1][-2:]
    )
    rhyme = min(1.0, rhyme_pairs / max(len(lines) / 2, 1))

    # 5. Vowel affinity (soft — not punitive)
    vowel_affinity = 0.5
    if vowel_family and vowel_family in PHONETIC_MAP:
        targets = PHONETIC_MAP[vowel_family]
        lyric_words = lyrics.lower().split()
        matches = sum(1 for w in lyric_words if any(w.endswith(t[-3:]) for t in targets if len(t) >= 3))
        vowel_affinity = min(1.0, 0.5 + matches * 0.08)

    # 6. Singability
    sing_scores = []
    for line in lines:
        words = line.split()
        if not words:
            continue
        long_w = sum(1 for w in words if len(w) > 8)
        sing_scores.append(max(0.0, 1.0 - (long_w / len(words)) * 1.5))
    singability = sum(sing_scores) / len(sing_scores) if sing_scores else 0.7

    # 7. Density fit
    if phrase_map and pairs:
        dfits = []
        for i in range(pairs):
            tgt = phrase_map[i]["syllables"]
            act = sum(_count_syllables(w) for w in lines[i].split())
            dl  = phrase_map[i].get("density_label", "mid")
            dfits.append(
                1.0 if (dl == "dense" and act >= tgt - 1)
                else 1.0 if (dl in ("sparse_sustained", "sparse_empty") and act <= tgt + 1)
                else 0.7
            )
        density_fit = sum(dfits) / len(dfits)
    else:
        density_fit = 0.5

    breakdown = {
        "syllable_fit":   round(syllable_fit, 3),
        "word_count_fit": round(word_count_fit, 3),
        "stress_fit":     round(stress_fit, 3),
        "rhyme":          round(rhyme, 3),
        "vowel_affinity": round(vowel_affinity, 3),
        "singability":    round(singability, 3),
        "density_fit":    round(density_fit, 3),
    }

    total = sum(breakdown[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)

    # Hard overflow penalty
    overflow = _check_overflow(lines, phrase_map)
    if overflow:
        total *= OVERFLOW_PENALTY
        breakdown["overflow_penalty"] = True
        breakdown["overflow_bars"] = overflow
    else:
        breakdown["overflow_penalty"] = False
        breakdown["overflow_bars"] = []

    breakdown["total"] = round(total, 3)
    return round(total, 3), breakdown


def _check_overflow(lines: list, phrase_map: list) -> list:
    """Returns list of bar indices (1-based) that overflow max_words or max_syllables."""
    bad = []
    for i, line in enumerate(lines):
        if i >= len(phrase_map):
            break
        phrase = phrase_map[i]
        max_w = phrase.get("max_words", 99)
        max_s = phrase.get("max_syllables", 99)
        actual_w = len(line.split())
        actual_s = sum(_count_syllables(w) for w in line.split())
        if actual_w > max_w + 1 or actual_s > max_s + 1:
            bad.append(i + 1)
    return bad


def _stress_fit_score(lines: list, phrase_map: list) -> float:
    scores = []
    for i, line in enumerate(lines):
        words = line.split()
        if not words:
            continue
        # End stress: last word ≤2 syl = clean landing
        last_syl = _count_syllables(re.sub(r"[^a-z]", "", words[-1].lower()))
        end_score = 1.0 if last_syl <= 2 else (0.7 if last_syl == 3 else 0.4)

        density = phrase_map[i].get("density_label", "mid") if i < len(phrase_map) else "mid"
        poly = sum(1 for w in words if _count_syllables(w) >= 4)
        poly_penalty = min(0.5, poly * (0.20 if density == "dense" else 0.10 if density == "mid" else 0.05))

        # Sustained penalty if too many words
        is_sust = phrase_map[i].get("is_sustained", False) if i < len(phrase_map) else False
        sust_penalty = max(0.0, (len(words) - 4) * 0.10) if is_sust and len(words) > 4 else 0.0

        scores.append(max(0.0, end_score - poly_penalty - sust_penalty))
    return sum(scores) / len(scores) if scores else 0.5


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_lyrics(
    rough_text: str,
    flow_data: dict,
    tone: str = "melodic",
    mode: str = "verse",
    vibe: str = "introspective",
    gen_mode: str = "cadence",
    key: str = "auto",
) -> list[dict]:
    phrase_map       = flow_data.get("phrase_map", [])
    phonetic_anchors = _extract_phonetic_anchors(phrase_map)

    results = []
    for variant in STYLE_VARIANTS:
        prompt = _build_user_prompt(
            rough_text, flow_data, tone, mode, vibe, gen_mode, key, variant, phonetic_anchors
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        lyrics = msg.content[0].text.strip()
        score, breakdown = _score_lyrics(lyrics, flow_data)
        results.append({
            "name":           variant["name"],
            "label":          variant["label"],
            "lyrics":         lyrics,
            "score":          score,
            "score_breakdown": breakdown,
        })

    # Pass 1: Haiku syllable verification
    results = _verify_and_fix(results, phrase_map)

    # Pass 2: Auto-fix bars still off by >2 syllables
    results = _autofix_weak_bars(results, flow_data, rough_text, tone, mode, vibe, gen_mode, key)

    # Re-score after fixes
    for r in results:
        r["score"], r["score_breakdown"] = _score_lyrics(r["lyrics"], flow_data)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
