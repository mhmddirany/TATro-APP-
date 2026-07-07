# ============================================================
# translation.py — Script 2
# Qwen translation + PDF export: translate_dataframe_with_qwen(...),
# save_final_translation_pdf(...), run_translation(...).
# Run app.py's cell first (installs + imports), then transcription.py's
# cell (this file calls qwen_generate, load_qwen_model, and save_json,
# which are defined there).
# ============================================================

def translate_text_with_qwen(tokenizer, model, text, source_lang_name, target_lang_name, max_new_tokens=256):
    text = str(text).strip()
    if text == "":
        return ""
    system_prompt = (
        f"You are an expert {source_lang_name}-to-{target_lang_name} translator. "
        f"You respond only in {target_lang_name}, written only in its native script. "
        "You never mix in Chinese, English, or any other language or script unless "
        "the source text itself contains a proper noun with no equivalent. "
        "Do not add commentary. Do not summarize."
    )
    user_prompt = f"""Translate the following {source_lang_name} text into {target_lang_name}.
Rules:
- Preserve the original meaning.
- Translate every word, including loanwords and foreign terms, into {target_lang_name}.
- Do not leave any words untranslated unless they are proper names.
- Do not switch to any other language or script partway through.
- Do not add new information.
- Do not remove information.
- Output only the translated text, nothing else.
Text:
{text}
"""
    return qwen_generate(tokenizer, model, system_prompt, user_prompt, max_new_tokens=max_new_tokens)

def translate_dataframe_with_qwen(df, tokenizer, model, source_lang_name, target_lang_name,
                                   source_column, target_column, max_new_tokens=256):
    assert source_column in df.columns, f"Column not found: {source_column}"
    texts = df[source_column].fillna("").astype(str).tolist()
    translations = []
    total = len(texts)
    for i, text in enumerate(texts):
        translations.append(
            translate_text_with_qwen(tokenizer, model, text, source_lang_name, target_lang_name, max_new_tokens=max_new_tokens)
        )
        if (i + 1) % 5 == 0 or (i + 1) == total:
            print(f"Translated {i + 1}/{total}")
    df[target_column] = translations
    return df

def prepare_pdf_text(text, is_rtl):
    if text is None or str(text).strip() == "":
        return "&lt;empty&gt;"
    text = str(text).strip()
    if is_rtl:
        try:
            text = arabic_reshaper.reshape(text)
        except Exception:
            pass
        text = get_display(text)
    return escape(text)

def save_final_translation_pdf(df, path, title, target_column, target_is_rtl, target_lang_name):
    pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm,
                             topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    alignment = TA_RIGHT if target_is_rtl else TA_LEFT
    title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontName="DejaVuSans-Bold",
                                  fontSize=16, spaceAfter=16, alignment=alignment, textColor=colors.HexColor("#1a1a2e"))
    speaker_style = ParagraphStyle("Speaker", parent=styles["Normal"], fontName="DejaVuSans-Bold",
                                    fontSize=10, leading=15, spaceBefore=10, spaceAfter=2, alignment=alignment,
                                    textColor=colors.HexColor("#0f3460"))
    text_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="DejaVuSans", fontSize=10,
                                 leading=15, spaceAfter=6, alignment=alignment, textColor=colors.HexColor("#333333"))
    title_text = f"{target_lang_name} Translation — {title}"
    title_paragraph = get_display(title_text) if target_is_rtl else title_text
    story = [Paragraph(title_paragraph, title_style), Spacer(1, 0.3 * cm)]
    for _, row in df.iterrows():
        speaker_number = row.get("speaker_number", "")
        speaker_line = f"Speaker {speaker_number}" if speaker_number != "" else str(row.get("speaker_label", row.get("speaker", "")))
        speaker_paragraph = get_display(speaker_line) if target_is_rtl else speaker_line
        translated_text = prepare_pdf_text(row.get(target_column, ""), target_is_rtl)
        story.append(Paragraph(speaker_paragraph, speaker_style))
        story.append(Paragraph(translated_text, text_style))
    doc.build(story)
    print("Final PDF saved ->", path)

# ============================================================
# FUNCTION 2 — TRANSLATION
# dataset -> Qwen translate -> save JSON -> final PDF
# ============================================================
def run_translation(dataset_df, input_path, output_dir, source_selected, target_selected, config):
    input_stem = input_path.stem
    translated_json_out = output_dir / f"{input_stem}_translated.json"
    final_pdf_out        = output_dir / f"{input_stem}_{target_selected['name'].lower()}_final.pdf"
    source_lang_name = source_selected["name"]
    target_lang_name = target_selected["name"]
    target_is_rtl = target_selected["is_rtl"]

    print("\n--- Translating with Qwen ---")
    qwen_tokenizer, qwen_model = load_qwen_model(config["qwen_model"])
    translated_df = translate_dataframe_with_qwen(
        dataset_df, qwen_tokenizer, qwen_model, source_lang_name, target_lang_name,
        source_column="source_text", target_column="target_text",
        max_new_tokens=config["translation_max_new_tokens"],
    )
    del qwen_tokenizer, qwen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_json(translated_df.to_dict(orient="records"), translated_json_out)

    print("\n--- Building final PDF ---")
    save_final_translation_pdf(
        translated_df, final_pdf_out, title=input_stem, target_column="target_text",
        target_is_rtl=target_is_rtl, target_lang_name=target_lang_name,
    )
    return translated_df, translated_json_out, final_pdf_out
