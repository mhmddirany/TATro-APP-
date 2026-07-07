# ============================================================
# app.py
# Installs everything, imports everything, and builds/launches the
# Gradio GUI. This is the file that should be run FIRST in a Colab
# session (it defines every import that transcription.py, translation.py,
# and main.py rely on) — after running it, run transcription.py, then
# translation.py, then (optionally) main.py.
#
# Extras on top of the plain pipeline:
#   1. The video input is a gr.Video, so an uploaded mp4 plays
#      inline right away instead of just showing a filename.
#   2. A small chat panel (gr.Chatbot + textbox + Send) wired to
#      `chatbot_stub`, which is currently an empty function (it
#      returns ""). Fill in real logic there whenever you want.
#   3. A "PDF preview" panel below the download button — shows the
#      final translated PDF inline (iframe) plus an "open in a new
#      tab" link, once translation finishes.
#
# Uses Qwen2.5-7B-Instruct (bumped up from 2.5-3B) to match what
# main.py uses — better quality, slower/more VRAM.
# ============================================================
!apt-get -qq update && apt-get -qq install -y ffmpeg fonts-dejavu-core
!pip install -q faster-whisper pyannote.audio transformers pandas torch \
    arabic-reshaper python-bidi reportlab gradio

import os
import re
import json
import subprocess
import tempfile
import gc
from pathlib import Path
from html import escape

import torch
import pandas as pd
from google.colab import drive, userdata
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import arabic_reshaper
from bidi.algorithm import get_display

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_RIGHT, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import gradio as gr

drive.mount("/content/drive")

def get_lang_by_name(name):
    for lang in LANGUAGES.values():
        if lang["name"] == name:
            return lang
    raise ValueError(f"Unknown language: {name}")

def derive_stem_from_dataset_json(path: Path):
    stem = path.stem
    if stem.endswith("_transcript_dataset"):
        stem = stem[: -len("_transcript_dataset")]
    return stem

MODE_BOTH = "Both (transcription + translation)"
MODE_TRANSCRIBE = "Transcription only"
MODE_TRANSLATE = "Translation only"

def make_pdf_preview_html(pdf_path):
    """Small inline PDF viewer + 'open in new tab' link. Returns an empty-state
    message until a PDF exists. Needs demo.launch(allowed_paths=[...]) to
    include the folder the PDF lives in, or the browser will refuse to load it."""
    if not pdf_path:
        return "<p style='color:#888; font-size:13px;'>No PDF yet — it will show up here once translation finishes.</p>"
    return f"""
    <div style="border:1px solid #ddd; border-radius:8px; overflow:hidden;">
        <iframe src="/file={pdf_path}" width="100%" height="600" style="border:none;"></iframe>
    </div>
    <p style="margin-top:6px; font-size:13px;">
        <a href="/file={pdf_path}" target="_blank">Open PDF in a new tab</a>
    </p>
    """

def process_job(mode, video_file, drive_path, transcript_json_file, transcript_json_path,
                 output_dir_text, source_lang_name, target_lang_name,
                 merge_window, use_qwen_correction, hf_token_input):
    output_dir = Path((output_dir_text or "/content/drive/MyDrive/fypoutput/").strip())
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_df = None
    input_stem = None
    # ---- Transcription stage (Transcription only / Both) ----
    if mode in (MODE_TRANSCRIBE, MODE_BOTH):
        if video_file is not None:
            input_path = Path(video_file if isinstance(video_file, str) else video_file.name)
        elif drive_path and drive_path.strip():
            input_path = Path(drive_path.strip())
        else:
            yield "Upload an mp4, or paste a path to one already in your Drive.", None, make_pdf_preview_html(None)
            return
        if not input_path.exists():
            yield f"File not found: {input_path}", None, make_pdf_preview_html(None)
            return
        input_stem = input_path.stem
        hf_token = hf_token_input.strip() if hf_token_input else None
        if not hf_token:
            try:
                hf_token = userdata.get("HF_TOKEN")
            except Exception:
                hf_token = None
        if not hf_token:
            yield "No HF token found. Paste one above, or add HF_TOKEN in Colab Secrets.", None, make_pdf_preview_html(None)
            return
        source_selected = get_lang_by_name(source_lang_name)
        config = {
            "whisper_model": "large-v3",
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "qwen_model": "Qwen/Qwen2.5-7B-Instruct",
            "max_chunk_duration": 30.0,
            "merge_window": int(merge_window),
            "translation_max_new_tokens": 256,
            "use_qwen_correction": use_qwen_correction,
        }
        yield "Extracting audio, diarizing, transcribing with Whisper... (see logs below this cell)", None, make_pdf_preview_html(None)
        dataset_df, dataset_json_out = run_transcription(input_path, output_dir, hf_token, source_selected, config)
        if mode == MODE_TRANSCRIBE:
            yield f"Transcription done. Saved to: {dataset_json_out}", str(dataset_json_out), make_pdf_preview_html(None)
            return
    # ---- Load existing transcript (Translation only) ----
    if mode == MODE_TRANSLATE:
        if transcript_json_file is not None:
            json_path = Path(transcript_json_file if isinstance(transcript_json_file, str) else transcript_json_file.name)
        elif transcript_json_path and transcript_json_path.strip():
            json_path = Path(transcript_json_path.strip())
        else:
            yield "Provide the *_transcript_dataset.json file from a previous transcription run.", None, make_pdf_preview_html(None)
            return
        if not json_path.exists():
            yield f"File not found: {json_path}", None, make_pdf_preview_html(None)
            return
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
        dataset_df = pd.DataFrame(records)
        input_stem = derive_stem_from_dataset_json(json_path)
    # ---- Translation stage (Translation only / Both) ----
    source_selected = get_lang_by_name(source_lang_name)
    target_selected = get_lang_by_name(target_lang_name)
    if source_selected["code"] == target_selected["code"]:
        yield "Source and target language must be different.", None, make_pdf_preview_html(None)
        return
    config = {
        "qwen_model": "Qwen/Qwen2.5-7B-Instruct",
        "translation_max_new_tokens": 256,
    }
    yield "Translating with Qwen and building the PDF...", None, make_pdf_preview_html(None)
    translated_df, translated_json_out, final_pdf_out = run_translation(
        dataset_df, Path(input_stem), output_dir, source_selected, target_selected, config
    )
    yield f"Done! PDF saved to: {final_pdf_out}", str(final_pdf_out), make_pdf_preview_html(final_pdf_out)

# ----- Empty chatbot stub -----
def chatbot_stub(message, history):
    """Placeholder — currently does nothing. Wire this up to whatever you
    want later (e.g. Q&A grounded in the transcript/translation you just
    produced). `history` is the list of (user, assistant) tuples that
    gr.Chatbot keeps for you."""
    response = ""  # <-- put real logic here later
    history = (history or []) + [(message, response)]
    return history, ""

with gr.Blocks(title="TATRO App") as demo:
    gr.Markdown("## Video Transcription + Translation")
    mode_radio = gr.Radio([MODE_TRANSCRIBE, MODE_TRANSLATE, MODE_BOTH], label="What do you want to run?", value=MODE_BOTH)
    with gr.Group(visible=True) as transcription_group:
        with gr.Row():
            # gr.Video (not gr.File) so the mp4 plays inline once uploaded.
            # sources=["upload"] disables the webcam-capture tab so this is
            # a plain file picker, not a "click to access webcam" prompt.
            video_input = gr.Video(label="Video file (.mp4) — plays here once uploaded", sources=["upload"])
            drive_path_box = gr.Textbox(label="OR: path to file already in Drive", placeholder="/content/drive/MyDrive/hebrew/video.mp4")
        hf_token_box = gr.Textbox(label="HF Token (leave blank to use Colab Secrets)", type="password")
    with gr.Group(visible=False) as translation_only_group:
        with gr.Row():
            transcript_json_input = gr.File(label="Transcript dataset JSON (from a prior transcription-only run)", file_types=[".json"])
            transcript_json_path_box = gr.Textbox(label="OR: path to that JSON already in Drive")
    with gr.Row():
        source_dd = gr.Dropdown(["Hebrew", "Arabic", "English", "French"], label="Spoken language", value="Hebrew")
        target_dd = gr.Dropdown(["Hebrew", "Arabic", "English", "French"], label="Translate into", value="Arabic")
    with gr.Row():
        merge_slider = gr.Slider(1, 6, value=3, step=1, label="Speaker-island merge window")
        correction_check = gr.Checkbox(label="Use Qwen ASR correction (slower)", value=False)
    output_dir_box = gr.Textbox(label="Output folder", value="/content/drive/MyDrive/fypoutput/")
    run_btn = gr.Button("Run", variant="primary")
    status_box = gr.Textbox(label="Status", interactive=False)
    file_output = gr.File(label="Output file")
    pdf_preview = gr.HTML(value=make_pdf_preview_html(None), label="PDF preview")

    # ----- Chat panel -----
    gr.Markdown("### Assistant (stub)")
    chatbot = gr.Chatbot(label="Chat", height=250)
    with gr.Row():
        chat_input = gr.Textbox(label="", placeholder="Type a message...", scale=4)
        chat_send = gr.Button("Send", scale=1)

    def toggle_groups(mode):
        return (
            gr.update(visible=mode in (MODE_TRANSCRIBE, MODE_BOTH)),
            gr.update(visible=mode == MODE_TRANSLATE),
        )
    mode_radio.change(toggle_groups, inputs=mode_radio, outputs=[transcription_group, translation_only_group])
    run_btn.click(
        fn=process_job,
        inputs=[mode_radio, video_input, drive_path_box, transcript_json_input, transcript_json_path_box,
                output_dir_box, source_dd, target_dd, merge_slider, correction_check, hf_token_box],
        outputs=[status_box, file_output, pdf_preview],
    )

    # ----- Wire the chat panel to the stub -----
    chat_send.click(chatbot_stub, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])
    chat_input.submit(chatbot_stub, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])

# allowed_paths lets the PDF iframe/link actually load the file — Gradio
# blocks serving anything outside these folders by default. Add any other
# Drive folder here if you point Output folder somewhere else.
#
# NOTE: this line runs as soon as this file's cell runs. At that point
# run_transcription/run_translation don't exist yet if you haven't run
# transcription.py / translation.py yet — that's fine, the app just won't
# be able to actually process anything until you have (click Run in the
# browser only after all four files have been run once).
demo.launch(share=True, debug=True, allowed_paths=["/content/drive/MyDrive/"])
