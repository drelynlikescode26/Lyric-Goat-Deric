import re
import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Phonetic mapping ─────────────────────────────────────────────────────────
PHONETIC_MAP = {
    "ayy": ["day", "pain", "same", "late", "way", "stay", "rain", "chains", "fade", "came", "name", "frame"],
    "oh":  ["go", "alone", "home", "road", "know", "show", "grow", "cold", "soul", "hold", "fold", "gold"],
    "ee":  ["free", "see", "believe", "breathe", "need", "real", "feel", "deep", "sleep", "leave", "bleed"],
    "ah":  ["heart", "far", "hard", "dark", "start", "apart", "scar", "star", "arm", "charge", "guard"],
    "uh":  ["love", "above", "enough", "blood", "flood", "stuck", "rough", "cut", "run", "done", "trust"],
    "ii":  ["high", "die", "cry", "fly", "try", "lie", "side", "mind", "time", "right", "night", "light"],
}

# Pitch-to-vowel affinity: which vowel sounds work best with each pitch direction
PITCH_VOWEL_AFFINITY = {
    "rising":   ["ee", "ayy", "ii"],   # bright high vowels go up
    "falling":  ["oh", "ah", "uh"],    # dark open vowels fall
    "held":     ["oh", "ee", "ah"],    # sustained sounds
    "staccato": ["ayy", "uh", "ii"],   # punchy short sounds
    "flat":     None,                  # no preference
}

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
    "gen_mode": {
        "literal":  "Decode the actual words/sounds from the mumble. Stay close to what was said.",
        "cadence":  "Ignore the words. Follow rhythm and syllable count only. Write original lines.",
        "creative": "Use vibe and emotion. Most freedom here — write what feels right for the moment.",
    },
}

STYLE_VARIANTS = [
    {"name": "melodic", "label": "Melodic Version", "description": "smooth and singable — built to float over a melody"},
    {"name": "rap",     "label": "Rap Version",     "description": "bars and cadence — built to be rapped, rhythm-locked"},
    {"name": "punchy",  "label": "Punchy Version",  "description": "short, hard-hitting lines — maximum impact, minimum words"},
]


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
    match = re.search(r"[aeiouy][^aeiouy]*$", word)
    if match:
        return "-" + match.group()
    return "-" + word[-2:] if len(word) >= 2 else word


def _extract_phonetic_anchors(phrase_map: list) -> list:
    anchors = []
    for phrase in phrase_map:
        words = phrase.get("words", []) or phrase.get("text", "").split()
        last_word = words[-1] if words else ""
        anchors.append(_ending_sound(str(last_word)))
    return anchors


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """\
You are a world-class songwriter and ghostwriter. You have written for Grammy-winning artists \
across hip-hop, R&B, trap, and pop. You do not write filler. You do not write generic lines. \
Every bar you write has intent behind it.

YOUR CRAFT:
- You write from the inside out — emotion first, words second
- You use SPECIFIC, CONCRETE images ("streetlight on a Tuesday" not "the city at night")
- You layer meaning: a line about a girl can also be about ambition
- You use internal rhymes, assonance, and alliteration — not just end rhymes
- You never use these clichés: "ride or die", "real ones", "on top of the world", \
"grind never stops", "make it rain", "all eyes on me", "can't stop won't stop"
- You write lines people want to repeat and remember
- You understand that singability > complexity — a line that lands > a line that impresses

YOUR CONSTRAINTS:
- Match each bar's syllable count precisely (±1 max) — the flow cannot be broken
- The number of output lines MUST equal the number of input bars exactly
- Respect pitch direction: rising phrases → bright ascending words; falling → heavier grounded words
- Low confidence bars → don't try to decode words, just match the rhythm
- No labels, headers, or explanations — pure lyrics only"""


# ── Phrase template builder ───────────────────────────────────────────────────

def _build_phrase_template(phrase_map: list, phonetic_anchors: list, melody_mode: bool, gen_mode: str) -> str:
    if not phrase_map:
        return ""

    lines = []
    for i, phrase in enumerate(phrase_map):
        anchor      = phonetic_anchors[i] if i < len(phonetic_anchors) else ""
        anchor_hint = f"  [rhyme: {anchor}]" if anchor else ""

        # Per-phrase audio metadata
        pitch    = phrase.get("pitch_symbol", "")
        density  = phrase.get("density_label", "")
        energy   = phrase.get("energy_level", "")
        conf     = phrase.get("confidence_label", "")
        pause_af = phrase.get("pause_after", 0)

        meta_parts = []
        if pitch:
            meta_parts.append(pitch)
        if density:
            meta_parts.append(density)
        if energy and energy != "mid":
            meta_parts.append(f"{energy} energy")
        meta_str = "  [" + " | ".join(meta_parts) + "]" if meta_parts else ""

        # Breathing room hint for long pauses
        breath_hint = "  [breathe after]" if pause_af > 0.6 else ""

        # Confidence drives whether to show transcript text
        low_conf = conf == "low"
        force_cadence = melody_mode or gen_mode == "cadence" or low_conf

        if force_cadence or not phrase["text"]:
            conf_note = "  ← cadence only" if low_conf and gen_mode == "literal" else ""
            lines.append(
                f"  Bar {i+1}: (rhythm)  →  {phrase['syllables']} syl"
                f"{meta_str}{anchor_hint}{breath_hint}{conf_note}"
            )
        else:
            lines.append(
                f'  Bar {i+1}: "{phrase["text"]}"  →  {phrase["syllables"]} syl'
                f"{meta_str}{anchor_hint}{breath_hint}"
            )

    return "\n".join(lines)


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
    tone_desc     = STYLE_DESCRIPTIONS["tone"].get(tone, tone)
    mode_desc     = STYLE_DESCRIPTIONS["mode"].get(mode, mode)
    vibe_desc     = STYLE_DESCRIPTIONS["vibe"].get(vibe, vibe)
    gen_mode_desc = STYLE_DESCRIPTIONS["gen_mode"].get(gen_mode, "")

    tempo       = flow_data.get("tempo_bpm", "unknown")
    tempo_str   = f"{tempo:.0f}" if isinstance(tempo, float) else str(tempo)
    flow_style  = flow_data.get("flow_style", "mixed")
    phrase_map  = flow_data.get("phrase_map", [])
    melody_mode = flow_data.get("melody_mode", False)
    vowel_family = flow_data.get("vowel_family")
    detected_key = flow_data.get("detected_key", key)

    phrase_template = _build_phrase_template(phrase_map, phonetic_anchors, melody_mode, gen_mode)

    # Phonetic vowel hint
    phonetic_hint = ""
    if vowel_family and vowel_family in PHONETIC_MAP:
        sample_words = ", ".join(PHONETIC_MAP[vowel_family][:6])
        phonetic_hint = f"\nDOMINANT VOWEL SOUND: '{vowel_family}' — bias endings toward: {sample_words}"

    # Pitch affinity hint
    pitch_hint = ""
    dominant_pitch = _dominant_pitch(phrase_map)
    if dominant_pitch and dominant_pitch in PITCH_VOWEL_AFFINITY:
        affinities = PITCH_VOWEL_AFFINITY[dominant_pitch]
        if affinities:
            pitch_hint = f"\nPITCH DIRECTION: mostly {dominant_pitch} — favor words with {'/'.join(affinities[:2])} vowel sounds"

    # Key hint
    key_used = key if key and key != "auto" else detected_key
    key_hint = f"\nKEY: {key_used}" if key_used else ""

    if gen_mode == "cadence" or melody_mode:
        input_block = f"""APPROACH: Cadence Mode — ignore words, follow the rhythm only.
The artist hummed/mumbled. Don't try to decode words.
Match syllable counts exactly and write original lines.

TEMPO: {tempo_str} BPM  |  FLOW: {flow_style}{key_hint}{phonetic_hint}{pitch_hint}

BAR-BY-BAR RHYTHM TEMPLATE:
{phrase_template or "  (use your judgment)"}"""

    elif gen_mode == "literal":
        input_block = f"""APPROACH: Literal Mode — stay close to the actual sounds mumbled.
Low-confidence bars (marked ← cadence only) should ignore the words and match rhythm.

ARTIST'S MUMBLE: "{rough_text}"

TEMPO: {tempo_str} BPM  |  FLOW: {flow_style}{key_hint}{phonetic_hint}{pitch_hint}

BAR-BY-BAR BREAKDOWN:
{phrase_template or "  (use the full transcription)"}"""

    else:  # creative
        input_block = f"""APPROACH: Creative Mode — use vibe and emotion, not the words.
The mumble is inspiration, not instruction. Write what the moment calls for.

MUMBLE REFERENCE: "{rough_text}"

TEMPO: {tempo_str} BPM  |  FLOW: {flow_style}{key_hint}{phonetic_hint}{pitch_hint}

BAR STRUCTURE:
{phrase_template or "  (use your judgment)"}"""

    rules = f"""RULES:
1. Write exactly one lyric line per bar — same count, same order
2. Each line MUST match its syllable count (±1 max) — the flow cannot break
3. End sounds should match rhyme targets where shown
4. Respect [breathe after] markers — the phrase needs room to land
5. Pitch direction hints tell you the vocal shape — honor them in word choice
6. Vibe: {vibe_desc}
7. Output ONLY the lyrics — no numbers, labels, or explanation"""

    return f"""Artist's raw vocal recording — transform this into real lyrics.

{input_block}

STYLE: {tone} · {mode} · {vibe}
- Tone: {tone_desc}
- Mode: {mode_desc}
- Approach: {gen_mode_desc}

Write the **{variant["label"]}** ({variant["description"]}):

{rules}

Write the lyrics now:"""


def _dominant_pitch(phrase_map: list) -> str:
    """Find the most common pitch pattern across all phrases."""
    if not phrase_map:
        return ""
    counts = {}
    for p in phrase_map:
        pat = p.get("pitch_pattern", "flat")
        counts[pat] = counts.get(pat, 0) + 1
    return max(counts, key=counts.get) if counts else ""


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

    # Get phrase-level audio features for this bar
    phrase_map = flow_data.get("phrase_map", [])
    phrase = phrase_map[bar_index] if bar_index < len(phrase_map) else {}
    pitch_sym  = phrase.get("pitch_symbol", "")
    density    = phrase.get("density_label", "")
    confidence = phrase.get("confidence_label", "med")

    phonetic_hint = ""
    if vowel_family and vowel_family in PHONETIC_MAP:
        sample_words = ", ".join(PHONETIC_MAP[vowel_family][:6])
        phonetic_hint = f"\nDominant vowel sound: '{vowel_family}' — use words like: {sample_words}"

    audio_hint = ""
    if pitch_sym or density:
        parts = [x for x in [pitch_sym, density] if x]
        audio_hint = f"\nBar audio: {' | '.join(parts)}"

    # Context display
    context_display = []
    for i, line in enumerate(context_lines):
        if i == bar_index:
            conf_note = f" [conf: {confidence}]" if confidence else ""
            context_display.append(
                f"  Bar {i+1}: ← REWRITE THIS (target: {syllable_count} syl{conf_note})"
            )
        else:
            lock_marker = " [LOCKED]" if str(i) in locked_lines else ""
            context_display.append(f'  Bar {i+1}: "{line}"{lock_marker}')
    context_str = "\n".join(context_display)

    prompt = f"""Rewrite ONE bar of lyrics. Everything else stays the same.

FULL LYRIC CONTEXT:
{context_str}

KEY: {key_used or "unknown"}  |  VIBE: {vibe_desc}{phonetic_hint}{audio_hint}

TASK:
Write a new version of Bar {bar_index + 1} only.
- Must be exactly {syllable_count} syllables (±1)
- Must flow naturally into the surrounding bars
- Must match the tone and vibe of the other lines
- Output ONLY the new line — nothing else"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=_build_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip().strip('"')


# ── Verification pass ─────────────────────────────────────────────────────────

def _verify_and_fix(versions: list, phrase_map: list) -> list:
    if not phrase_map:
        return versions

    targets = [{"bar": i + 1, "syllables": p["syllables"]} for i, p in enumerate(phrase_map)]
    versions_text = "\n\n".join(f'VERSION: {v["label"]}\n{v["lyrics"]}' for v in versions)

    prompt = f"""You are a syllable-accuracy editor for song lyrics.

SYLLABLE TARGETS (each line must match ±1):
{json.dumps(targets, indent=2)}

LYRICS TO REVIEW:
{versions_text}

Fix ONLY lines off by more than 1 syllable. Keep meaning and rhyme.
Return valid JSON only:
{{"melodic": "...", "rap": "...", "punchy": "..."}}
Lines separated by \\n. Return ONLY the JSON."""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = re.sub(r"^```(?:json)?\s*", "", message.content[0].text.strip())
        raw = re.sub(r"\s*```$", "", raw)
        fixed = json.loads(raw)
        name_map = {v["name"]: v for v in versions}
        for key_name in ("melodic", "rap", "punchy"):
            if key_name in fixed and key_name in name_map and fixed[key_name].strip():
                name_map[key_name]["lyrics"] = fixed[key_name].strip()
    except Exception:
        pass

    return versions


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_lyrics(lyrics: str, flow_data: dict) -> float:
    lines = [l for l in lyrics.split("\n") if l.strip()]
    if not lines:
        return 0.0

    phrase_map   = flow_data.get("phrase_map", [])
    vowel_family = flow_data.get("vowel_family")

    # 1. Syllable match score (50%)
    if phrase_map and lines:
        pairs = min(len(lines), len(phrase_map))
        diffs = [
            max(0, 1 - abs(
                sum(_count_syllables(w) for w in lines[i].split()) - phrase_map[i]["syllables"]
            ) / max(phrase_map[i]["syllables"], 1))
            for i in range(pairs)
        ]
        syllable_score = sum(diffs) / pairs if diffs else 0.5
    else:
        syllable_score = 0.5

    # 2. Rhyme density (20%)
    last_words = [l.strip().split()[-1].lower().rstrip(".,!?") for l in lines if l.strip().split()]
    rhyme_pairs = sum(
        1 for i in range(len(last_words) - 1)
        if len(last_words[i]) >= 2 and len(last_words[i + 1]) >= 2
        and last_words[i][-2:] == last_words[i + 1][-2:]
    )
    rhyme_score = min(1.0, rhyme_pairs / max(len(lines) / 2, 1))

    # 3. Vowel family match (15%)
    vowel_score = 0.5  # neutral default
    if vowel_family and vowel_family in PHONETIC_MAP:
        target_words = PHONETIC_MAP[vowel_family]
        lyric_words  = lyrics.lower().split()
        matches = sum(1 for w in lyric_words if any(w.endswith(t[-3:]) for t in target_words if len(t) >= 3))
        vowel_score = min(1.0, matches * 0.15)

    # 4. Singability: penalize lines with too many long words (15%)
    sing_scores = []
    for line in lines:
        words = line.split()
        if not words:
            continue
        long_words = sum(1 for w in words if len(w) > 8)
        ratio = long_words / len(words)
        sing_scores.append(max(0.0, 1.0 - ratio * 1.5))
    singability = sum(sing_scores) / len(sing_scores) if sing_scores else 0.7

    # 5. Density fit: dense phrases should have higher syllable counts (bonus weight)
    density_score = 0.5
    if phrase_map and lines:
        density_fits = []
        for i in range(min(len(lines), len(phrase_map))):
            target_syl = phrase_map[i]["syllables"]
            actual_syl = sum(_count_syllables(w) for w in lines[i].split())
            dl = phrase_map[i].get("density_label", "mid")
            # Dense bars should have more syllables; sparse should have fewer
            if dl == "dense" and actual_syl >= target_syl - 1:
                density_fits.append(1.0)
            elif dl == "sparse" and actual_syl <= target_syl + 1:
                density_fits.append(1.0)
            else:
                density_fits.append(0.7)
        density_score = sum(density_fits) / len(density_fits) if density_fits else 0.5

    return round(
        syllable_score * 0.50 +
        rhyme_score    * 0.20 +
        vowel_score    * 0.15 +
        singability    * 0.10 +
        density_score  * 0.05,
        3
    )


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
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        lyrics = message.content[0].text.strip()
        results.append({
            "name":   variant["name"],
            "label":  variant["label"],
            "lyrics": lyrics,
            "score":  _score_lyrics(lyrics, flow_data),
        })

    results = _verify_and_fix(results, phrase_map)

    for r in results:
        r["score"] = _score_lyrics(r["lyrics"], flow_data)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
