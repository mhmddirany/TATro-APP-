# ============================================================
# main.py — runs the full pipeline directly: transcribe, then
# translate. Prompts for source/target language in the console
# (no GUI). Optional — skip it if you're only using the Gradio
# app (app_gradio.py).
# ============================================================

def main():
    drive.mount("/content/drive")

    # # ---- EDIT THESE If YOU NEED HARD INPUT----
    # INPUT_PATH = "/content/drive/MyDrive/hebrew/piece_of_hebrew.mp4"  # video/audio to process
    # OUTPUT_DIR = "/content/drive/MyDrive/fypoutput/"                  # where outputs get written
    # USE_QWEN_CORRECTION = False   # True = extra Qwen pass that cleans ASR errors before translation
    # # ---------------------

    config = {
        "whisper_model": "large-v3",
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "qwen_model": "Qwen/Qwen2.5-7B-Instruct",  # used for both optional ASR correction and translation
        "max_chunk_duration": 30.0,
        # Consecutive same-speaker turns considered when merging short islands
        # of a different speaker. Fixed for now — will become a Tkinter
        # control later so the user can choose it themselves.
        "merge_window": 3,
        "translation_max_new_tokens": 256,
        "use_qwen_correction": USE_QWEN_CORRECTION,
    }

    hf_token = userdata.get("HF_TOKEN")
    assert hf_token, "Add your Hugging Face token in Colab Secrets as HF_TOKEN"

    input_path = Path(INPUT_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert input_path.exists(), f"Input file not found: {input_path}"

    print("=" * 60)
    source_selected = ask_language("Choose the SPOKEN language in the audio/video:")
    print()
    target_selected = ask_language("Choose the language to TRANSLATE into:", exclude_code=source_selected["code"])
    print("=" * 60)
    print(f"\nSpoken language : {source_selected['name']}")
    print(f"Translate into  : {target_selected['name']}\n")
    print("CUDA available:", torch.cuda.is_available())

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

main()
