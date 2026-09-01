# Deric vocal benchmark

This benchmark is offline. It does not call OpenAI, Anthropic, AudioShake, or
any other paid API.

1. Put 20–30 short WAV clips in `benchmarks/clips/`.
2. Copy `dataset.example.json` to `dataset.json`.
3. For every clip, manually mark each phrase's start, end, and syllable count.
4. Run the current detector:

   ```bash
   python scripts/benchmark_detection.py benchmarks/dataset.json
   ```

5. Optionally install `requirements-melody.txt`, then compare Basic Pitch:

   ```bash
   MELODY_NOTE_PROVIDER=basic_pitch \
     python scripts/benchmark_detection.py benchmarks/dataset.json
   ```

Recommended clip mix:

- 8 clean hums
- 8 mumbled flows
- 6 rough but recognizable lyric takes
- 4 takes recorded over a beat
- 4 difficult clips with runs, long holds, breaths, or background noise

Keep the personal `dataset.json` and audio clips out of Git. Only the blank
template belongs in the repository.
