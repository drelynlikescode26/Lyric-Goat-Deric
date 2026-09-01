#!/usr/bin/env python3
"""Benchmark Lyric Goat phrase/syllable detection without paid APIs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.analyze import analyze_flow, preanalyze_audio
from services.preprocess import preprocess_branches


def score_phrase_maps(expected: list[dict], detected: list[dict]) -> dict:
    paired = min(len(expected), len(detected))
    syllable_errors = []
    boundary_errors = []
    for idx in range(paired):
        target = expected[idx]
        actual = detected[idx]
        syllable_errors.append(abs(int(target["syllables"]) - int(actual.get("target_syllables", actual.get("syllables", 0)))))
        boundary_errors.extend([
            abs(float(target["start_time"]) - float(actual.get("start_time", 0))),
            abs(float(target["end_time"]) - float(actual.get("end_time", 0))),
        ])

    return {
        "expected_phrases": len(expected),
        "detected_phrases": len(detected),
        "phrase_count_error": abs(len(expected) - len(detected)),
        "mean_syllable_error": round(sum(syllable_errors) / len(syllable_errors), 3) if syllable_errors else None,
        "mean_boundary_error_seconds": round(sum(boundary_errors) / len(boundary_errors), 3) if boundary_errors else None,
    }


def run_dataset(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for clip in payload.get("clips", []):
        audio_path = (manifest_path.parent / clip["audio"]).resolve()
        if not audio_path.exists():
            results.append({"id": clip["id"], "error": f"Missing file: {audio_path}"})
            continue

        branches = preprocess_branches(str(audio_path))
        try:
            melody_path = branches["melody_path"]
            pre_data = preanalyze_audio(melody_path)
            analysis = analyze_flow(melody_path, [], pre_data=pre_data, hum_mode=True)
            score = score_phrase_maps(clip.get("expected_phrases", []), analysis.get("phrase_map", []))
            score.update({
                "id": clip["id"],
                "note_provider": analysis.get("note_provider", "pyin"),
            })
            results.append(score)
        finally:
            for path in branches.values():
                if isinstance(path, str) and os.path.exists(path):
                    os.unlink(path)

    valid = [row for row in results if "error" not in row]
    summary = {
        "dataset": payload.get("dataset", "unnamed"),
        "clips": len(results),
        "successful_clips": len(valid),
        "mean_phrase_count_error": _mean(valid, "phrase_count_error"),
        "mean_syllable_error": _mean(valid, "mean_syllable_error"),
        "mean_boundary_error_seconds": _mean(valid, "mean_boundary_error_seconds"),
    }
    return {"summary": summary, "results": results}


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 3) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_dataset(args.manifest.resolve())
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["summary"]["successful_clips"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
