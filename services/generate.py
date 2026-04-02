import re
import os
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


def _build_system_prompt() -> str:
    return (
        "You are an elite ghostwriter and lyricist. You specialize in taking raw mumble "
        "recordings and turning them into polished lyrics that preserve the artist's exact "
        "cadence, rhythm, and emotional intent. "
        "You match syllables precisely — each line you write must have the SAME syllable "
        "count as the original line. You keep the emotional core and key sounds of the "
        "original. You never write generic lyrics. You write for THAT artist, THAT moment."
    )


def _build_phrase_template(phrase_map: list, melody_mode: bool = False) -> str:
    """
    Build a line-by-line template string.
    In melody mode, shows syllable counts only (no source words).
    """
    if not phrase_map:
        return ""

    lines = []
    for i, phrase in enumerate(phrase_map, 1):
        if melody_mode or not phrase["text"]:
            lines.append(f"  Line {i}: (melody phrase)  →  {phrase['syllables']} syllables")
        else:
            lines.append(
                f'  Line {i}: "{phrase["text"]}"  →  {phrase["syllables"]} syllables'
            )
    return "\n".join(lines)


def _build_user_prompt(
    rough_text: str,
    flow_data: dict,
    tone: str,
    mode: str,
    vibe: str,
    variant: dict,
) -> str:
    tone_desc = STYLE_DESCRIPTIONS["tone"].get(tone, tone)
    mode_desc = STYLE_DESCRIPTIONS["mode"].get(mode, mode)
    vibe_desc = STYLE_DESCRIPTIONS["vibe"].get(vibe, vibe)

    tempo = flow_data.get("tempo_bpm", "unknown")
    flow_style = flow_data.get("flow_style", "mixed")
    phrase_map = flow_data.get("phrase_map", [])
    melody_mode = flow_data.get("melody_mode", False)
    phrase_template = _build_phrase_template(phrase_map, melody_mode=melody_mode)

    if melody_mode:
        input_block = f"""INPUT TYPE: Pure melody/hum — no words detected
The artist hummed a melody. There are no source words to preserve.
Your only constraints are the rhythm template below.

TEMPO: {tempo:.0f if isinstance(tempo, float) else tempo} BPM  |  FLOW STYLE: {flow_style}

MELODY RHYTHM TEMPLATE (syllable count per phrase):
{phrase_template if phrase_template else "  (no phrase data — use your judgment)"}"""
        rules = """RULES (follow these exactly):
1. Write one output line for EVERY phrase in the rhythm template
2. Each line MUST match the syllable count shown (±1 syllable max)
3. Write original, vivid lyrics that fit the tempo and vibe — no filler words
4. The lyrics should feel like they were MADE for this melody
5. Make rhymes happen naturally — never sacrifice meaning to force a rhyme
6. Tone/vibe must match: {vibe_desc}
7. Output ONLY the lyrics — no labels, no explanations, no line numbers"""
    else:
        input_block = f"""FULL ROUGH TRANSCRIPTION:
"{rough_text}"

TEMPO: {tempo:.0f if isinstance(tempo, float) else tempo} BPM  |  FLOW STYLE: {flow_style}

LINE-BY-LINE BREAKDOWN (each line = a natural phrase from the recording):
{phrase_template if phrase_template else '  (no phrase data — use the full transcription)'}"""
        rules = """RULES (follow these exactly):
1. Write one output line for EVERY line in the breakdown — same number of lines, same order
2. Each output line MUST match the syllable count of the original line (±1 syllable max)
3. Keep words or sounds from the original where they fit — don't throw everything away
4. Preserve the emotional meaning and feel of each original line
5. Make rhymes happen naturally — never sacrifice flow or meaning to force a rhyme
6. Tone/vibe must match: {vibe_desc}
7. Output ONLY the lyrics — no labels, no explanations, no line numbers"""

    rules = rules.format(vibe_desc=vibe_desc)

    prompt = f"""Here is a raw vocal recording from an artist.

{input_block}

STYLE SETTINGS:
- Tone: {tone} → {tone_desc}
- Mode: {mode} → {mode_desc}
- Vibe: {vibe} → {vibe_desc}

YOUR TASK — Write the **{variant["label"]}** ({variant["description"]}):

{rules}

Write the lyrics now:"""

    return prompt


def generate_lyrics(
    rough_text: str,
    flow_data: dict,
    tone: str = "melodic",
    mode: str = "verse",
    vibe: str = "introspective",
) -> list[dict]:
    """
    Generate 3 lyric versions (melodic, rap, punchy) for the given input.
    Returns a list of dicts with name, label, lyrics, and score.
    """
    results = []

    for variant in STYLE_VARIANTS:
        prompt = _build_user_prompt(rough_text, flow_data, tone, mode, vibe, variant)

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

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _score_lyrics(lyrics: str, flow_data: dict) -> float:
    """
    Score lyrics on:
    - Syllable match vs original phrases (line by line)
    - Rhyme density
    """
    lines = [l for l in lyrics.split("\n") if l.strip()]
    if not lines:
        return 0.0

    phrase_map = flow_data.get("phrase_map", [])

    # Syllable match score — compare line by line
    syllable_score = 0.0
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

    # Rhyme score — check last words of adjacent lines
    last_words = [
        l.strip().split()[-1].lower().rstrip(".,!?") for l in lines if l.strip().split()
    ]
    rhyme_pairs = 0
    for i in range(len(last_words) - 1):
        a, b = last_words[i], last_words[i + 1]
        if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:]:
            rhyme_pairs += 1
    rhyme_score = min(1.0, rhyme_pairs / max(len(lines) / 2, 1))

    return round(syllable_score * 0.7 + rhyme_score * 0.3, 3)
