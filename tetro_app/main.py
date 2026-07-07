# ============================================================
# main.py
# Runs the full pipeline directly: transcribe, then translate.
# Optional — skip it if you're only using the Gradio app (app.py).
#
# Works locally:
#     python main.py --input "/path/to/video.mp4" --language he --target ar --merge 3
# or interactively (no flags — it'll prompt you for everything):
#     python main.py
# and also still works pasted into its own Colab cell, after
# transcription.py and translation.py have been run/pasted first.
# ============================================================
import os
import argparse
from pathlib import Path

import torch

from transcription import LANGUAGES, ask_language, run_transcription
from translation import run_translation

try:
    from google.colab import drive, userdata
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def get_lang_by_code(code):
    for lang in LANGUAGES.values():
        if lang["code"] == code:
            return lang
    valid = ", ".join(lang["code"] for lang in LANGUAGES.values())
    raise ValueError(f"Unknown language code {code!r}. Valid codes: {valid}")

def get_hf_token(cli_token=None):
    if cli_token:
        return cli_token
    if IN_COLAB:
        try:
            token = userdata.get("HF_TOKEN")
            if token:
                return token
        except Exception:
            pass
    return os.environ.get("HF_TOKEN")

def main():
    parser = argparse.ArgumentParser(description="Transcribe and translate a video/audio file.")
    parser.add_argument("--input", help="Path to the .mp4/.wav file. Omit to be prompted.")
    parser.add_argument("--language", help="Spoken language code: he, ar, en, or fr. Omit to be prompted.")
    parser.add_argument("--target", help="Language code to translate into. Omit to be prompted.")
    parser.add_argument("--merge", type=int, default=3, help="Speaker-island merge window (default: 3).")
    parser.add_argument("--output", help="Output folder (default: next to the input file, or Drive in Colab).")
    parser.add_argument("--hf-token", dest="hf_token", help="Hugging Face token. Falls back to the HF_TOKEN "
                         "environment variable, or Colab Secrets if running in Colab.")
    parser.add_argument("--correct", action="store_true", help="Run an extra Qwen pass to clean up ASR errors "
                         "before translation.")
    args = parser.parse_args()

    if IN_COLAB:
        drive.mount("/content/drive")

    hf_token = get_hf_token(args.hf_token)
    assert hf_token, ("No Hugging Face token found. Pass --hf-token, set the HF_TOKEN environment "
                       "variable, or add it to Colab Secrets.")

    input_path = Path(args.input) if args.input else Path(input("Path to the video/audio file: ").strip())
    assert input_path.exists(), f"Input file not found: {input_path}"

    default_output = "/content/drive/MyDrive/fypoutput/" if IN_COLAB else str(input_path.parent)
    output_dir = Path(args.output or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_selected = (get_lang_by_code(args.language) if args.language
                        else ask_language("Choose the SPOKEN language in the audio/video:"))
    target_selected = (get_lang_by_code(args.target) if args.target
                        else ask_language("Choose the language to TRANSLATE into:", exclude_code=source_selected["code"]))

    config = {
        "whisper_model": "large-v3",
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "qwen_model": "Qwen/Qwen2.5-7B-Instruct",  # used for both optional ASR correction and translation
        "max_chunk_duration": 30.0,
        "merge_window": args.merge,
        "translation_max_new_tokens": 256,
        "use_qwen_correction": args.correct,
    }

    print("=" * 60)
    print(f"Spoken language : {source_selected['name']}")
    print(f"Translate into  : {target_selected['name']}")
    print("CUDA available:", torch.cuda.is_available())
    print("=" * 60)

    dataset_df, dataset_json_out = run_transcription(input_path, output_dir, hf_token, source_selected, config)
    translated_df, translated_json_out, final_pdf_out = run_translation(
        dataset_df, input_path, output_dir, source_selected, target_selected, config
    )

    print("\n" + "=" * 60)
    print("ALL DONE.")
    print(f"Spoken language   : {source_selected['name']}")
    print(f"Translated into   : {target_selected['name']}\n")
    print("Files saved:")
    print(" - Transcript dataset JSON :", dataset_json_out)
    print(" - Translated JSON         :", translated_json_out)
    print(" - FINAL PDF               :", final_pdf_out)
    print("=" * 60)

if __name__ == "__main__":
    main()
