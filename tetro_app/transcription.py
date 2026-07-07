# ============================================================
# transcription.py — Script 1
# Whisper transcription + speaker diarization: language dictionary,
# audio/diarization helpers, run_transcription(...).
#
# Self-contained module — works both run locally (`python main.py` /
# `python app.py`) and pasted into its own Colab cell. Needs the
# packages in requirements.txt installed, plus ffmpeg on PATH.
# ============================================================
import os
import re
import json
import subprocess
import tempfile
import gc
from pathlib import Path

import torch
import pandas as pd
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# LANGUAGE DICTIONARY
# ============================================================
LANGUAGES = {
    "1": {"name": "Hebrew",  "code": "he", "whisper_code": "he", "is_rtl": True},
    "2": {"name": "Arabic",  "code": "ar", "whisper_code": "ar", "is_rtl": True},
    "3": {"name": "English", "code": "en", "whisper_code": "en", "is_rtl": False},
    "4": {"name": "French",  "code": "fr", "whisper_code": "fr", "is_rtl": False},
}

def ask_language(prompt_label, exclude_code=None):
    print(prompt_label)
    for key, lang in LANGUAGES.items():
        print(f"  {key}. {lang['name']}")
    choice = input("Enter number (1-4): ").strip()
    while choice not in LANGUAGES or LANGUAGES[choice]["code"] == exclude_code:
        if choice in LANGUAGES and LANGUAGES[choice]["code"] == exclude_code:
            print("Target must be different from the source language.")
        else:
            print("Invalid choice.")
        choice = input("Enter number (1-4): ").strip()
    return LANGUAGES[choice]

# ============================================================
# SHARED HELPERS (audio / diarization / chunks / save)
# ============================================================
MAX_SPEAKER_GAP = 0.5
SHORT_ISLAND_SECONDS = 2.0
MIN_TURN_DURATION = 0.25

def extract_audio(input_file, audio_path):
    command = ["ffmpeg", "-y", "-loglevel", "quiet", "-i", input_file, "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(command, check=True)
    print(f"Audio extracted -> {audio_path}")

def run_diarization(audio_path, hf_token, diarization_model):
    print("Loading diarization model...")
    pipeline = Pipeline.from_pretrained(diarization_model, token=hf_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    print("Running diarization...")
    diarization = pipeline(audio_path)
    print("Diarization finished")
    return diarization

def get_annotation_from_diarization(diarization):
    if hasattr(diarization, "itertracks"):
        return diarization
    if hasattr(diarization, "speaker_diarization"):
        return diarization.speaker_diarization
    if hasattr(diarization, "exclusive_speaker_diarization"):
        return diarization.exclusive_speaker_diarization
    raise TypeError("Unsupported diarization output format")

def merge_consecutive_same_speaker(turns, max_gap=MAX_SPEAKER_GAP):
    if not turns:
        return []
    merged = [turns[0].copy()]
    for turn in turns[1:]:
        last = merged[-1]
        same_speaker = turn["speaker"] == last["speaker"]
        gap = turn["start"] - last["end"]
        if same_speaker and gap <= max_gap:
            last["end"] = max(last["end"], turn["end"])
            last["merged_from"].extend(turn["merged_from"])
        else:
            merged.append(turn.copy())
    return merged

def smooth_short_speaker_islands(turns, merge_window=3, max_island_duration=SHORT_ISLAND_SECONDS, max_gap=MAX_SPEAKER_GAP):
    """Absorb short "islands" of a different speaker sandwiched inside a run
    of the same speaker."""
    if len(turns) < merge_window:
        return turns, []
    smoothed, audit = [], []
    i = 0
    while i < len(turns):
        if i + merge_window - 1 < len(turns):
            window = turns[i:i + merge_window]
            first_turn, last_turn = window[0], window[-1]
            middle_turns = window[1:-1]
            first_spk, last_spk = first_turn["speaker"], last_turn["speaker"]
            gaps_ok = True
            prev_end = first_turn["end"]
            for t in window[1:]:
                if t["start"] - prev_end > max_gap:
                    gaps_ok = False
                    break
                prev_end = t["end"]
            middle_durations_ok = all((t["end"] - t["start"]) <= max_island_duration for t in middle_turns)
            middle_diff_speaker = bool(middle_turns) and all(t["speaker"] != first_spk for t in middle_turns)
            is_short_island = (
                first_spk == last_spk
                and middle_diff_speaker
                and middle_durations_ok
                and gaps_ok
            )
            if is_short_island:
                merged_from = []
                for t in window:
                    merged_from.extend(t["merged_from"])
                new_turn = {
                    "start": first_turn["start"], "end": last_turn["end"], "speaker": first_spk,
                    "merged_from": merged_from,
                }
                audit.append({
                    "action": "absorbed_short_speaker_island",
                    "window_size": merge_window,
                    "previous_speaker": first_spk,
                    "middle_speakers": [t["speaker"] for t in middle_turns],
                    "next_speaker": last_spk,
                    "new_start": round(new_turn["start"], 3), "new_end": round(new_turn["end"], 3),
                    "merged_original_turns": new_turn["merged_from"],
                })
                smoothed.append(new_turn)
                i += merge_window
                continue
        smoothed.append(turns[i].copy())
        i += 1
    return smoothed, audit

def split_long_turns(turns, max_duration):
    chunks = []
    for turn in turns:
        start, final_end, speaker = turn["start"], turn["end"], turn["speaker"]
        while start < final_end:
            end = min(start + max_duration, final_end)
            if end - start >= MIN_TURN_DURATION:
                chunks.append({
                    "start": round(start, 3), "end": round(end, 3), "speaker": speaker,
                    "duration": round(end - start, 3), "merged_from": turn["merged_from"],
                })
            start = end
    return chunks

def diarization_to_chunks(diarization, max_duration, merge_window=3):
    annotation = get_annotation_from_diarization(diarization)
    raw_turns = []
    for turn_id, (seg, _, spk) in enumerate(annotation.itertracks(yield_label=True), start=1):
        start, end = float(seg.start), float(seg.end)
        if end - start < MIN_TURN_DURATION:
            continue
        raw_turns.append({"start": start, "end": end, "speaker": spk, "merged_from": [turn_id]})
    if not raw_turns:
        return [], []
    raw_turns = sorted(raw_turns, key=lambda x: x["start"])
    print("Raw diarization turns:", len(raw_turns))
    merged_turns = merge_consecutive_same_speaker(raw_turns)
    print("After same-speaker merge:", len(merged_turns))
    smoothed_turns, island_audit = smooth_short_speaker_islands(merged_turns, merge_window=merge_window)
    print(f"After short-island smoothing (merge_window={merge_window}):", len(smoothed_turns))
    print("Short speaker islands fixed:", len(island_audit))
    final_turns = merge_consecutive_same_speaker(smoothed_turns)
    print("After final same-speaker merge:", len(final_turns))
    chunks = split_long_turns(final_turns, max_duration=max_duration)
    print("Final ASR chunks:", len(chunks))
    return chunks, island_audit

def load_whisper_model(whisper_model_name):
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"
    print("Loading Whisper model...", whisper_model_name, "| Device:", device)
    model = WhisperModel(whisper_model_name, device=device, compute_type=compute_type)
    print("Whisper model loaded")
    return model

def transcribe_chunks_with_whisper(audio_path, chunks, model, whisper_language):
    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, chunk in enumerate(chunks):
            start, end = float(chunk["start"]), float(chunk["end"])
            duration = round(end - start, 3)
            chunk_path = os.path.join(tmp_dir, f"chunk_{idx:04d}.wav")
            command = ["ffmpeg", "-y", "-loglevel", "quiet", "-i", audio_path,
                       "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-ar", "16000", "-ac", "1", chunk_path]
            subprocess.run(command, check=True)
            segments, info = model.transcribe(chunk_path, language=whisper_language, beam_size=5,
                                               vad_filter=False, word_timestamps=False)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            results.append({
                "id": idx + 1, "speaker": chunk["speaker"], "start": round(start, 3), "end": round(end, 3),
                "duration": duration, "language": whisper_language, "transcription": text,
            })
            if (idx + 1) % 5 == 0 or (idx + 1) == len(chunks):
                print(f"Transcribed {idx + 1}/{len(chunks)} chunks")
    return results

def load_qwen_model(qwen_model_name):
    print("Loading Qwen model...", qwen_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(qwen_model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    print("Qwen loaded successfully.")
    return tokenizer, model

# Codepoint ranges Qwen occasionally leaks into Hebrew/Arabic/English/French
# output (mostly CJK). DejaVu Sans has no glyphs for these, so they'd render
# as empty "tofu" boxes in the final PDF — always safe to strip.
_UNEXPECTED_SCRIPT_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x3000, 0x303F),   # CJK punctuation
    (0xFF00, 0xFFEF),   # Fullwidth forms
    (0x3040, 0x30FF),   # Hiragana / Katakana
    (0xAC00, 0xD7AF),   # Hangul syllables
]

def _strip_unexpected_script(text):
    if not text:
        return text
    cleaned = "".join(
        ch for ch in text
        if not any(start <= ord(ch) <= end for start, end in _UNEXPECTED_SCRIPT_RANGES)
    )
    return re.sub(r"\s+", " ", cleaned).strip()

def clean_qwen_output(text):
    if text is None:
        return ""
    text = str(text).strip()
    # Qwen3.5 sometimes leaks a reasoning block even with enable_thinking=False
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for bad in ("```hebrew", "```arabic", "```text", "```"):
        text = text.replace(bad, "")
    text = _strip_unexpected_script(text)
    return text.strip()

def qwen_generate(tokenizer, model, system_prompt, user_prompt, max_new_tokens=256):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    # enable_thinking=False: Qwen3.5 thinks by default (emits <think>...</think>
    # before the real answer) — not wanted for a translation/correction task.
    # Harmless no-op for older templates (e.g. Qwen2.5) that ignore it.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return clean_qwen_output(tokenizer.decode(generated_ids, skip_special_tokens=True))

def correct_transcription_with_qwen(tokenizer, model, text, language_name):
    text = str(text).strip()
    if text == "":
        return ""
    system_prompt = ("You are an expert ASR transcript corrector. "
                      "Your job is to correct speech-to-text transcription errors. "
                      "Do not translate. Do not summarize. Do not add explanations.")
    user_prompt = f"""Correct the following {language_name} ASR transcription.
Rules:
- Keep the same language.
- Keep the same meaning.
- Fix obvious ASR mistakes.
- Fix punctuation when useful.
- Do not add new information.
- Do not remove important information.
- Output only the corrected transcript.
Transcript:
{text}
"""
    return qwen_generate(tokenizer, model, system_prompt, user_prompt, max_new_tokens=256)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    return " ".join(text.split())

def save_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("JSON saved ->", path)

def save_csv(records_or_df, path):
    df = records_or_df if isinstance(records_or_df, pd.DataFrame) else pd.DataFrame(records_or_df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("CSV saved ->", path)

# ============================================================
# FUNCTION 1 — TRANSCRIPTION
# audio -> diarize -> Whisper -> (optional Qwen cleanup) -> dataset
# ============================================================
def run_transcription(input_path, output_dir, hf_token, source_selected, config):
    input_stem = input_path.stem
    audio_out        = output_dir / f"{input_stem}_audio.wav"
    raw_json_out     = output_dir / f"{input_stem}_transcript.json"
    dataset_json_out = output_dir / f"{input_stem}_transcript_dataset.json"
    whisper_language = source_selected["whisper_code"]
    source_lang_name = source_selected["name"]

    print("\n--- Extracting audio ---")
    extract_audio(str(input_path), str(audio_out))

    print("\n--- Running diarization ---")
    diarization = run_diarization(str(audio_out), hf_token, config["diarization_model"])

    print("\n--- Building clean chunks ---")
    chunks, chunk_audit = diarization_to_chunks(
        diarization, max_duration=config["max_chunk_duration"], merge_window=config["merge_window"]
    )
    assert chunks, "ERROR: Diarization returned no speaker turns."

    # Free the diarization model's GPU memory before Whisper loads — without
    # this, both models can be resident at once and blow past a T4's VRAM.
    del diarization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Transcribing with Whisper ---")
    whisper_model = load_whisper_model(config["whisper_model"])
    transcriptions = transcribe_chunks_with_whisper(str(audio_out), chunks, whisper_model, whisper_language)
    save_json(transcriptions, raw_json_out)
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if config["use_qwen_correction"]:
        print("\n--- Correcting transcript with Qwen ---")
        qwen_tokenizer, qwen_model = load_qwen_model(config["qwen_model"])
        for i, t in enumerate(transcriptions):
            t["transcription"] = correct_transcription_with_qwen(qwen_tokenizer, qwen_model, t["transcription"], source_lang_name)
            if (i + 1) % 5 == 0 or (i + 1) == len(transcriptions):
                print(f"Corrected {i + 1}/{len(transcriptions)}")
        del qwen_tokenizer, qwen_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n--- Building dataset ---")
    speaker_labels = []
    for t in transcriptions:
        if t["speaker"] not in speaker_labels:
            speaker_labels.append(t["speaker"])
    speaker_map = {speaker: i + 1 for i, speaker in enumerate(speaker_labels)}

    rows = []
    for t in transcriptions:
        rows.append({
            "id": t["id"], "speaker_number": speaker_map[t["speaker"]], "speaker_label": t["speaker"],
            "start": t["start"], "end": t["end"], "duration": t["duration"],
            "source_language": source_lang_name, "source_text": t["transcription"], "target_text": "",
        })
    dataset_df = pd.DataFrame(rows)
    dataset_df["source_text"] = dataset_df["source_text"].astype(str).str.strip()
    dataset_df = dataset_df[dataset_df["source_text"] != ""].reset_index(drop=True)
    dataset_df["id"] = range(1, len(dataset_df) + 1)

    save_json(dataset_df.to_dict(orient="records"), dataset_json_out)
    return dataset_df, dataset_json_out
