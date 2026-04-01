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
        "description": "smooth, sung, flowing — built to be sung over a melody",
    },
    {
        "name": "rap",
        "label": "Rap Version",
        "description": "bars, cadence, rhythm-locked — built to be rapped",
    },
    {
        "name": "punchy",
        "label": "Punchy Version",
        "description": "short, hard-hitting lines — maximum impact, minimum words",
    },
]


def _build_system_prompt() -> str:
    return (
        "You are an elite ghostwriter and lyricist who has written for top artists across hip-hop, "
        "R&B, pop, and trap. You specialize in taking raw mumble recordings and rough transcriptions "
        "and turning them into polished, authentic lyrics that preserve the artist's original cadence, "
        "energy, and intent. You understand syllable matching, flow, rhyme schemes, and emotional tone. "
        "You never write generic lyrics — you write lyrics that sound like THAT artist, for THAT moment."
    )


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

    rhythm_str = flow_data.get("rhythm_string", "")
    tempo = flow_data.get("tempo_bpm", "unknown")
    syllable_count = flow_data.get("syllable_count", "unknown")
    flow_style = flow_data.get("flow_style", "mixed")

    prompt = f"""Here is a raw mumble/vocal recording transcription from an artist:

ROUGH TRANSCRIPTION:
"{rough_text}"

FLOW ANALYSIS:
- Detected tempo: {tempo} BPM
- Flow style: {flow_style}
- Estimated syllables: {syllable_count}
"""

    if rhythm_str:
        prompt += f"- Rhythm pattern: {rhythm_str}\n"

    prompt += f"""
STYLE SETTINGS:
- Tone: {tone} → {tone_desc}
- Mode: {mode} → {mode_desc}
- Vibe: {vibe} → {vibe_desc}

YOUR TASK:
Write the **{variant['label']}** — {variant['description']}

Rules:
1. Match the syllable count and rhythm of the original recording as closely as possible
2. Keep the emotional core/meaning of the rough transcription
3. The lyrics must feel AUTHENTIC to the artist, not like generic AI writing
4. Match the {mode} format strictly
5. Do NOT add labels, headers, or explanations — output ONLY the lyrics
6. Keep rhyme schemes natural — don't force rhymes at the cost of meaning
7. The energy should match: {vibe_desc}

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
    Returns a list of dicts with name, label, and lyrics.
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

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


def _score_lyrics(lyrics: str, flow_data: dict) -> float:
    """
    Simple heuristic scoring:
    - Syllable match (closer to original = better)
    - Line count (reasonable for mode)
    - Rhyme density
    """
    lines = [l for l in lyrics.split("\n") if l.strip()]
    if not lines:
        return 0.0

    # Count syllables roughly (vowel groups)
    import re
    def count_syllables(text):
        return len(re.findall(r"[aeiouAEIOU]+", text))

    total_syllables = count_syllables(lyrics)
    target_syllables = flow_data.get("syllable_count", total_syllables)

    syllable_diff = abs(total_syllables - target_syllables)
    syllable_score = max(0, 1 - (syllable_diff / max(target_syllables, 1)))

    # Rhyme score: check last words of lines
    last_words = [l.strip().split()[-1].lower().rstrip(".,!?") for l in lines if l.strip().split()]
    rhyme_pairs = 0
    for i in range(len(last_words) - 1):
        a, b = last_words[i], last_words[i + 1]
        if len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:]:
            rhyme_pairs += 1
    rhyme_score = min(1.0, rhyme_pairs / max(len(lines) / 2, 1))

    return round((syllable_score * 0.6 + rhyme_score * 0.4), 3)
