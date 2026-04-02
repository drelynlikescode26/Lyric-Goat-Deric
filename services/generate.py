import re
import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

STYLE_DESCRIPTIONS = {
    "tone": {
        "melodic": "smooth, sung, emotional — words flow like a melody",
        "aggressive": "hard-hitting, intense, punchy — every word lands like a punch",
        "simple": "clean, clear, easy to follow — nothing overcomplicated",
        "punchlines": "clever wordplay, double meanings, bars that make people rewind",
    },
    "mode": {
        "hook": "a catchy, repeatable hook/chorus — short lines, memorable, singable",
        "verse": "a full verse — storytelling, building energy, detailed",
        "story": "narrative, cinematic, paint a picture with words",
    },
    "vibe": {
        "sad": "emotional, introspective, real pain — like crying in a car at night",
        "hype": "energetic, motivating, make the crowd go crazy",
        "introspective": "deep, thoughtful, self-aware — looking inward",
        "love": "romantic, vulnerable, genuine feeling",
    },
}

STYLE_VARIANTS = [
    {
        "name": "melodic",
        "label": "Melodic Version",
        "description": "smooth and singable — built to float over a melody",
    },
    {
        "name": "rap",
        "label": "Rap Version",
        "description": "bars and cadence — built to be rapped, rhythm-locked",
    },
    {
        "name": "punchy",
        "label": "Punchy Version",
        "description": "short, hard-hitting lines — maximum impact, minimum words",
    },
]


# ── Syllable counting ──────────────────────────────────────────────────────────

def _count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


# ── Phonetic anchor extraction ─────────────────────────────────────────────────

def _ending_sound(word: str) -> str:
    """
    Extract the ending vowel+consonant cluster of a word.
    E.g. "pain" → "-ain", "love" → "-ove", "yeah" → "-eah"
    Used as rhyme targets so generated lyrics sound like the mumble.
    """
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    # Find last vowel group and everything after it
    match = re.search(r"[aeiouy][^aeiouy]*$", word)
    if match:
        return "-" + match.group()
    return "-" + word[-2:] if len(word) >= 2 else word


def _extract_phonetic_anchors(phrase_map: list) -> list:
    """
    For each phrase, extract the ending sound of the last word.
    These become rhyme anchors passed to Claude.
    """
    anchors = []
    for phrase in phrase_map:
        words = phrase.get("words", [])
        if not words:
            text_words = phrase.get("text", "").split()
            words = text_words
        last_word = words[-1] if words else ""
        anchors.append(_ending_sound(str(last_word)))
    return anchors


# ── Prompt building ────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return (
        "You are an elite ghostwriter and lyricist with years of experience working with "
        "hip-hop, R&B, and trap artists. You specialize in taking raw mumble recordings — "
        "rough ideas, half-formed melodies, and phonetic sketches — and transforming them "
        "into polished, authentic lyrics.\n\n"
        "Your output must:\n"
        "- Match each line's syllable count EXACTLY (±1 max)\n"
        "- Preserve the emotional energy and key sounds of the original mumble\n"
        "- Use the provided rhyme/sound anchors to keep the output sounding like the artist\n"
        "- Never be generic — every line should feel personal and intentional\n"
        "- Flow naturally when sung or rapped at the given tempo"
    )


def _build_phrase_template(phrase_map: list, phonetic_anchors: list, melody_mode: bool) -> str:
    if not phrase_map:
        return ""

    lines = []
    for i, phrase in enumerate(phrase_map):
        anchor = phonetic_anchors[i] if i < len(phonetic_anchors) else ""
        anchor_hint = f"  [rhyme target: {anchor}]" if anchor else ""

        if melody_mode or not phrase["text"]:
            lines.append(
                f"  Line {i+1}: (melody phrase)  →  {phrase['syllables']} syllables{anchor_hint}"
            )
        else:
            lines.append(
                f'  Line {i+1}: "{phrase["text"]}"  →  {phrase["syllables"]} syllables{anchor_hint}'
            )
    return "\n".join(lines)


def _build_user_prompt(
    rough_text: str,
    flow_data: dict,
    tone: str,
    mode: str,
    vibe: str,
    variant: dict,
    phonetic_anchors: list,
) -> str:
    tone_desc = STYLE_DESCRIPTIONS["tone"].get(tone, tone)
    mode_desc = STYLE_DESCRIPTIONS["mode"].get(mode, mode)
    vibe_desc = STYLE_DESCRIPTIONS["vibe"].get(vibe, vibe)

    tempo = flow_data.get("tempo_bpm", "unknown")
    tempo_str = f"{tempo:.0f}" if isinstance(tempo, float) else str(tempo)
    flow_style = flow_data.get("flow_style", "mixed")
    phrase_map = flow_data.get("phrase_map", [])
    melody_mode = flow_data.get("melody_mode", False)
    phrase_template = _build_phrase_template(phrase_map, phonetic_anchors, melody_mode)

    if melody_mode:
        input_block = f"""INPUT TYPE: Pure melody/hum — no words detected
The artist hummed/sang a melody. No source words to preserve.
Use the rhythm template and rhyme targets below as your constraints.

TEMPO: {tempo_str} BPM  |  FLOW STYLE: {flow_style}

MELODY RHYTHM TEMPLATE:
{phrase_template or "  (no phrase data — use your judgment)"}"""

        rules = f"""RULES:
1. Write one line for EVERY phrase in the template — same count, same order
2. Match each line's syllable count (±1 max)
3. The ending sound of each line should match the rhyme target shown
4. Write original, vivid lyrics that feel made for this melody
5. Rhymes must be natural — never sacrifice meaning to force one
6. Vibe: {vibe_desc}
7. Output ONLY the lyrics — no labels, no line numbers, no explanation"""

    else:
        input_block = f"""ROUGH TRANSCRIPTION:
"{rough_text}"

TEMPO: {tempo_str} BPM  |  FLOW STYLE: {flow_style}

LINE-BY-LINE BREAKDOWN:
{phrase_template or "  (no phrase data — use the full transcription)"}"""

        rules = f"""RULES:
1. Write one line for EVERY line in the breakdown — same count, same order
2. Match each line's syllable count (±1 max)
3. The ending sound of each line should match the rhyme target shown — this keeps it sounding like YOUR mumble
4. Keep key words and sounds from the original where they fit naturally
5. Preserve the emotional feel of each original line
6. Rhymes must be natural — never sacrifice meaning to force one
7. Vibe: {vibe_desc}
8. Output ONLY the lyrics — no labels, no line numbers, no explanation"""

    return f"""Raw vocal recording from an artist.

{input_block}

STYLE:
- Tone: {tone} → {tone_desc}
- Mode: {mode} → {mode_desc}
- Vibe: {vibe} → {vibe_desc}

Write the **{variant["label"]}** ({variant["description"]}):

{rules}

Write the lyrics now:"""


# ── Verification pass ──────────────────────────────────────────────────────────

def _verify_and_fix(versions: list, phrase_map: list) -> list:
    """
    Single Claude call that reviews all 3 versions and fixes any lines
    where syllable count is off by more than 1 from the target.
    Returns corrected versions.
    """
    if not phrase_map:
        return versions

    # Build syllable targets
    targets = [{"line": i + 1, "syllables": p["syllables"]} for i, p in enumerate(phrase_map)]

    versions_text = "\n\n".join(
        f'VERSION: {v["label"]}\n{v["lyrics"]}'
        for v in versions
    )

    prompt = f"""You are a syllable-accuracy editor for song lyrics.

SYLLABLE TARGETS (each line must match ±1):
{json.dumps(targets, indent=2)}

LYRICS TO REVIEW:
{versions_text}

TASK:
For each version, check every line's syllable count against the target.
Fix ONLY lines that are off by more than 1 syllable — rewrite just those lines to hit the target count while keeping the same meaning and rhyme.
Do not change lines that already match.

Return your response as valid JSON in this exact format:
{{
  "melodic": "<full corrected lyrics, lines separated by newline>",
  "rap": "<full corrected lyrics, lines separated by newline>",
  "punchy": "<full corrected lyrics, lines separated by newline>"
}}

Return ONLY the JSON object. No explanation."""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        fixed = json.loads(raw)

        name_map = {v["name"]: v for v in versions}
        for key in ("melodic", "rap", "punchy"):
            if key in fixed and key in name_map and fixed[key].strip():
                name_map[key]["lyrics"] = fixed[key].strip()

        return versions

    except Exception:
        # Verification failed — return originals unchanged
        return versions


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_lyrics(
    rough_text: str,
    flow_data: dict,
    tone: str = "melodic",
    mode: str = "verse",
    vibe: str = "introspective",
) -> list[dict]:
    phrase_map = flow_data.get("phrase_map", [])
    phonetic_anchors = _extract_phonetic_anchors(phrase_map)

    results = []
    for variant in STYLE_VARIANTS:
        prompt = _build_user_prompt(
            rough_text, flow_data, tone, mode, vibe, variant, phonetic_anchors
        )
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        lyrics = message.content[0].text.strip()
        results.append({
            "name": variant["name"],
            "label": variant["label"],
            "lyrics": lyrics,
            "score": _score_lyrics(lyrics, flow_data),
        })

    # Quality gate: fix syllable mismatches in one lightweight pass
    results = _verify_and_fix(results, phrase_map)

    # Re-score after fixes then sort
    for r in results:
        r["score"] = _score_lyrics(r["lyrics"], flow_data)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_lyrics(lyrics: str, flow_data: dict) -> float:
    lines = [l for l in lyrics.split("\n") if l.strip()]
    if not lines:
        return 0.0

    phrase_map = flow_data.get("phrase_map", [])

    # Syllable match — line by line
    if phrase_map and lines:
        pairs = min(len(lines), len(phrase_map))
        diffs = []
        for i in range(pairs):
            target = phrase_map[i]["syllables"]
            actual = sum(_count_syllables(w) for w in lines[i].split())
            diff = abs(target - actual)
            diffs.append(max(0, 1 - diff / max(target, 1)))
        syllable_score = sum(diffs) / pairs if diffs else 0.5
    else:
        syllable_score = 0.5

    # Rhyme density — last words of adjacent lines
    last_words = [
        l.strip().split()[-1].lower().rstrip(".,!?")
        for l in lines if l.strip().split()
    ]
    rhyme_pairs = 0
    for i in range(len(last_words) - 1):
        a, b = last_words[i], last_words[i + 1]
        if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:]:
            rhyme_pairs += 1
    rhyme_score = min(1.0, rhyme_pairs / max(len(lines) / 2, 1))

    return round(syllable_score * 0.7 + rhyme_score * 0.3, 3)
