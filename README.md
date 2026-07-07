# TATRO App

Same 5-cell structure as your original notebook, plus two GUI additions in the app cell: the video input now plays inline as soon as you upload it, and there's a small chat panel at the bottom wired to an empty placeholder function.

## Files — one per Colab cell

- `01_imports.py` — Cell 1: apt/pip installs + all imports.
- `02_transcription.py` — Cell 2: language dictionary, audio/diarization helpers, Whisper transcription, `run_transcription(...)`.
- `03_translation.py` — Cell 3: Qwen translation, PDF export, `run_translation(...)`.
- `04_main.py` — Cell 4: `main()` — runs the pipeline directly with `input()` prompts, no GUI. Optional if you're only using the Gradio app.
- `05_app_gradio.py` — Cell 5: the Gradio GUI (`process_job`, the two new GUI pieces, `demo.launch(...)`).
- `app_preview.py` — standalone, dependency-light copy of the same layout with processing mocked out, for a quick look at the UI without loading Whisper/pyannote/Qwen.

## How to use

1. Open your Colab notebook, one cell per file above, in order (1 → 5).
2. Paste each file's contents into its own cell and run top to bottom.
3. Cell 4 (`main()`) is optional — skip it if you only want the Gradio app. Cell 5 only needs `LANGUAGES`, `run_transcription`, and `run_translation` from Cells 1–3.
4. In Cell 5, click the public Gradio link it prints. Upload an mp4 — it now plays right in the browser instead of just showing a filename. Scroll down to "Assistant (stub)" to see the chat box.

### What changed in Cell 5, specifically

- `video_input = gr.File(...)` → `video_input = gr.Video(...)`. `process_job` already did `Path(video_file if isinstance(video_file, str) else video_file.name)`, and `gr.Video` returns a plain filepath string, so no other code needed to change.
- Added `chatbot_stub(message, history)` and a `gr.Chatbot` + textbox + Send button wired to it. Right now `chatbot_stub` just returns `""` for every message — it's a hook, not a feature. To make it do something, edit the `response = ""` line in that function (e.g. call Qwen, look something up in `dataset_df`, etc.).

### Small fix folded in

`run_transcription` (Cell 2) calls `gc.collect()` a few times to free GPU memory between model loads — `01_imports.py` now includes `import gc` up front so that's defined before it's needed.

## How to use `app_preview.py` (optional, just for looking)

Run locally or in any Python environment with `pip install gradio`:

```
python app_preview.py
```

It opens the same layout, but the Run button doesn't call your real pipeline — it just prints back what you selected. Useful for checking the layout without GPU/models/HF token.

## Known limitation

I couldn't host a live, clickable link for this from my side — my sandbox kills background processes as soon as each command finishes, which doesn't work for a server that needs to stay up. Colab doesn't have that restriction, so `gradio_cell_updated.py` will work normally there with `share=True`.
