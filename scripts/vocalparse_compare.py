"""Run only with a separately installed VocalParse Python environment.

The main app never imports PyTorch or downloads a checkpoint. Model loading and
untrusted model output remain in a bounded subprocess, not the studio server.
"""
import argparse
from contextlib import redirect_stdout
import json
import re
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        from vocalparse import transcribe_one
        raw = transcribe_one(args.audio, checkpoint=args.checkpoint,
                             max_new_tokens=2048, attn_implementation="sdpa")
    # Keep the full predicted notation visible. Do not invent timestamps from
    # a score or turn the number of predicted notes into a syllable target.
    words = raw.split("<|file_sep|>")[0] if "<|file_sep|>" in raw else re.sub(r"<[^>]+>", "", raw)
    print(json.dumps(dict(transcript=words[:16000], notation=raw[:32000], phrases=[],
        timing_note="Experimental predicted lyrics/notation. No measured phrase timings; note counts are not syllable counts.")))


if __name__ == "__main__":
    main()
