import evaluate
import gradio as gr
import nltk
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download("punkt", quiet=True)


class TextSummarizerEngine:

  def __init__(
      self,
      model_name: str = "sshleifer/distilbart-cnn-12-6",
      device: str = None,
  ):
    self.device = (
        device
        if device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
        self.device
    )

  @staticmethod
  def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    return " ".join(text.split()).strip()

  def generate_extractive_baseline(
      self, text: str, num_sentences: int = 2
  ) -> str:
    cleaned = self.clean_text(text)
    sentences = nltk.sent_tokenize(cleaned)
    if not sentences:
      return ""
    return " ".join(sentences[:num_sentences])

  def generate_abstractive_summary(
      self, text: str, max_length: int = 120, min_length: int = 30
  ) -> str:
    cleaned_input = self.clean_text(text)
    inputs = self.tokenizer(
        cleaned_input, return_tensors="pt", max_length=1024, truncation=True
    ).to(self.device)

    with torch.no_grad():
      summary_ids = self.model.generate(
          inputs["input_ids"],
          max_length=max_length,
          min_length=min_length,
          num_beams=4,
          length_penalty=2.0,
          early_stopping=True,
      )

    return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


summarizer_engine = TextSummarizerEngine()


def summarize_user_text(user_text: str, max_words: int):
  if not user_text or not user_text.strip():
    return (
        "Please paste or type text to summarize.",
        "Please paste or type text to summarize.",
        "0 words",
    )

  max_len = int(max_words)
  min_len = max(10, int(max_len * 0.3))

  extractive_res = summarizer_engine.generate_extractive_baseline(user_text)
  abstractive_res = summarizer_engine.generate_abstractive_summary(
      user_text, max_length=max_len, min_length=min_len
  )

  orig_word_count = len(user_text.split())
  abs_word_count = len(abstractive_res.split())
  stats = f"Original: {orig_word_count} words | Abstractive Summary: {abs_word_count} words"

  return extractive_res, abstractive_res, stats


with gr.Blocks(title="Text Summarization System") as demo:
  gr.Markdown("# 📝 Abstractive & Extractive Text Summarizer")
  gr.Markdown(
      "Paste text below to generate automated summaries using DistilBART and"
      " NLTK baselines."
  )

  with gr.Row():
    with gr.Column(scale=1):
      text_input = gr.Textbox(
          lines=8,
          placeholder="Paste your text here...",
          label="Document Input",
      )
      length_slider = gr.Slider(
          minimum=30,
          maximum=250,
          value=120,
          step=10,
          label="Max Summary Length (Tokens)",
      )
      submit_btn = gr.Button("Generate Summary", variant="primary")

    with gr.Column(scale=1):
      abstractive_output = gr.Textbox(
          lines=4, label="Abstractive BART Summary (Seq2Seq)"
      )
      extractive_output = gr.Textbox(
          lines=4, label="Extractive Baseline Summary"
      )
      stats_output = gr.Textbox(label="Text Reduction Metrics")

  submit_btn.click(
      fn=summarize_user_text,
      inputs=[text_input, length_slider],
      outputs=[extractive_output, abstractive_output, stats_output],
  )

if __name__ == "__main__":
  # Render sets the PORT environment variable dynamically
  import os

  port = int(os.environ.get("PORT", 7860))
  demo.launch(server_name="0.0.0.0", server_port=port)
