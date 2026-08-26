import json
import os
import time
import gradio as gr
import nltk
from huggingface_hub import InferenceClient

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Free Hugging Face Serverless API client
client = InferenceClient()
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


def clean_text(text: str) -> str:
  text = text.replace("\n", " ")
  return " ".join(text.split()).strip()


def generate_extractive_baseline(text: str, num_sentences: int = 2) -> str:
  cleaned = clean_text(text)
  sentences = nltk.sent_tokenize(cleaned)
  if not sentences:
    return ""
  return " ".join(sentences[:num_sentences])


def generate_abstractive_summary(
    text: str, max_len: int = 120, min_len: int = 30
) -> str:
  cleaned_input = clean_text(text)
  try:
    # Use client.post to pass generation parameters cleanly to HF Inference API
    payload = {
        "inputs": cleaned_input,
        "parameters": {"max_length": int(max_len), "min_length": int(min_len)},
    }
    response_bytes = client.post(json=payload, model=MODEL_NAME)
    data = json.loads(response_bytes.decode("utf-8"))

    if isinstance(data, list) and len(data) > 0:
      return data[0].get("summary_text", "")
    elif isinstance(data, dict):
      if "summary_text" in data:
        return data["summary_text"]
      if "error" in data:
        return f"Inference API Error: {data['error']}"

    return str(data)
  except Exception as e:
    return f"Inference API Error: {str(e)}"


def summarize_user_text(user_text: str, max_words: int):
  if not user_text or not user_text.strip():
    return (
        "Please enter text to summarize.",
        "Please enter text to summarize.",
        "0 words",
    )

  max_len = int(max_words)
  min_len = max(10, int(max_len * 0.3))

  extractive_res = generate_extractive_baseline(user_text)
  abstractive_res = generate_abstractive_summary(
      user_text, max_len=max_len, min_len=min_len
  )

  orig_word_count = len(user_text.split())
  abs_word_count = len(abstractive_res.split())
  stats = f"Original: {orig_word_count} words | Abstractive Summary: {abs_word_count} words"

  return extractive_res, abstractive_res, stats


with gr.Blocks(title="Text Summarizer") as demo:
  gr.Markdown("# 📝 Text Summarization System")
  with gr.Row():
    with gr.Column():
      text_input = gr.Textbox(
          lines=8, placeholder="Paste text here...", label="Document Input"
      )
      length_slider = gr.Slider(
          minimum=30, maximum=250, value=120, step=10, label="Max Summary Length"
      )
      submit_btn = gr.Button("Generate Summary", variant="primary")
    with gr.Column():
      abstractive_output = gr.Textbox(
          lines=4, label="Abstractive BART Summary"
      )
      extractive_output = gr.Textbox(
          lines=4, label="Extractive Baseline Summary"
      )
      stats_output = gr.Textbox(label="Metrics")

  submit_btn.click(
      fn=summarize_user_text,
      inputs=[text_input, length_slider],
      outputs=[extractive_output, abstractive_output, stats_output],
  )

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  print(f"[*] Starting Gradio server on port {port}...")
  demo.launch(server_name="0.0.0.0", server_port=port, prevent_thread_lock=True)

  while True:
    time.sleep(3600)
