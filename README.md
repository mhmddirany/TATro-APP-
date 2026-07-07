# Video Translation Pipeline

Takes an .mp4 video, transcribes the speech with Whisper, corrects and translates it into Arabic with Qwen, and produces a speaker-labeled PDF — plus a JSON file meant for later embedding & retrieval.

**Pipeline:** mp4 → diarize + Whisper transcribe → batch same-speaker chunks → Qwen correct → Qwen translate (into Arabic) → save JSON + PDF

## Files and what each one does

### transcription.py — Stage 1

Input: path to an .mp4 file and the spoken language. Extracts audio, runs speaker diarization (pyannote), cleans up the diarization internally (fixed, not user-configurable), splits into Whisper-sized chunks, and transcribes each chunk with Whisper.

Output: `<video_name>_transcript.json`, saved in the same folder as the mp4 — one entry per chunk with `speaker`, `speaker_number`, `start`, `end`, `duration`, and the raw transcription.

### translation.py — Stage 2

Input: the transcript JSON from Stage 1, plus "number of merge" (default 3). Groups up to that many consecutive same-speaker chunks into one block (fewer, longer calls instead of one call per tiny chunk), then for each block:

1. **Correction** — Qwen fixes obvious ASR errors in the raw text.
2. **Translation** — Qwen translates the corrected text into Arabic (the app always translates to Arabic; if the spoken language already is Arabic, this step is skipped and the corrected text is used as-is).

Output, both saved in the same folder as the mp4:

- `<video_name>_translated.json` — speaker, start, end, duration, `source_text` (raw), `corrected_text`, `target_text` (Arabic) for every merged block. This is the file used later for embedding & retrieval.
- `<video_name>_arabic_final.pdf` — the Arabic translation, labeled "Speaker 1", "Speaker 2", etc. (no timestamps in the PDF).

### main.py — orchestrator

Input: mp4 path, spoken language code (he/ar/en/fr), and merge count. Calls `transcription.transcribe_video()` then `translation.translate_transcript()` in order, reporting progress through the whole pipeline as one 0–100% callback.

Output: the translated JSON path (for embedding/retrieval) and the PDF path.

Can also be run directly from the command line for testing:

```
python main.py --input "/path/to/video.mp4" --language he --merge 3
```

### app.py — the GUI (Gradio, not a desktop app)

Lets the user upload the .mp4 (or point to one already in Drive), pick the spoken language from a dropdown, set the merge count, and enter a Hugging Face token. Runs the pipeline and streams status updates as it goes ("Extracting audio, diarizing, transcribing with Whisper...", "Translating with Qwen and building the PDF...", ...). Once finished, the PDF is downloadable and also previewable right in the page (inline viewer + "open in a new tab" link) — there's no separate desktop window or auto-opening file, it's all in the browser.

**This is the file most users should run.**

## Requirements

```
pip install -r requirements.txt
```

Also needs, whether running locally or in Colab:

- `ffmpeg` on PATH
- DejaVu fonts installed somewhere findable (`apt-get install fonts-dejavu-core` on Linux/Colab; on Windows/Mac, download DejaVu Sans from dejavu-fonts.github.io) — `translation.py` searches a few common font locations and raises a clear error telling you where to put it if it can't find one
- A Hugging Face token with access to `pyannote/speaker-diarization-3.1` (accept the model's terms on the HF site first)

## How to run

### Locally

```
pip install -r requirements.txt
export HF_TOKEN=your_token_here          # (Windows PowerShell: $env:HF_TOKEN="your_token_here")
python app.py
```

This prints a local URL (and a public one, since `share=True`) — open it, upload your video, pick the spoken language, adjust the merge count if you want, and click Run. When it's done, download the PDF or view it inline on the same page. If you don't pass an HF token in the browser field, it picks up the `HF_TOKEN` environment variable automatically.

Prefer the command line, no GUI at all:

```
python main.py --input "/path/to/video.mp4" --language he --target ar --merge 3
```

Run `python main.py --help` for all the flags, or just run `python main.py` with no flags to be prompted for each choice interactively.

### In Colab

Paste each file into its own cell and run them in this order: `app.py`, `transcription.py`, `translation.py` (`main.py` only if you want the no-GUI version instead). Install system + Python dependencies in a cell before that:

```
!apt-get -qq update && apt-get -qq install -y ffmpeg fonts-dejavu-core
!pip install -q -r requirements.txt
```

Add your token to Colab Secrets as `HF_TOKEN`, or paste it into the GUI's token field.

**Security note:** never paste a real Hugging Face token into a chat — set it as an environment variable, in Colab Secrets, or directly in the GUI's token field.

## Model notes

- Whisper: `large-v3` (see `DEFAULT_CONFIG` in main.py).
- Correction + translation: `Qwen/Qwen3.5-4B` — chosen over the smaller Qwen2.5-3B after Arabic translation quality came out weak with it; Qwen3+ expanded multilingual coverage from 29 to 119 languages, which helps a lot for Arabic specifically. Qwen3.5 "thinks" by default, so generation explicitly disables that (`enable_thinking=False`) and any leaked `<think>` tags get stripped defensively.
- Both `transcribe_video()` and `translate_transcript()` fall back to CPU automatically on GPUs too old for the installed PyTorch/cuDNN build (see `cuda_is_usable()` in transcription.py) instead of crashing.
- Stray CJK characters Qwen sometimes leaks into output are stripped before saving/PDF-building, since the PDF font has no glyphs for them.

## Notes on "number of merge"

This controls how many consecutive same-speaker chunks get combined into one block before running correction + translation on it — a merge count of 3 means at most 3 chunks per Qwen call. Higher = fewer, longer Qwen calls (faster overall, less granular speaker turns in the output); lower = more, shorter calls (slower, finer-grained). This is separate from the diarization-level speaker-island smoothing in transcription.py, which is fixed internally and not user-configurable.
