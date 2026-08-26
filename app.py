import json
import os
import time
import urllib.error
import urllib.request
import gradio as gr
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# Active 2026 Hugging Face Serverless Router Endpoint
API_URL = (
    f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"
)


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
  payload = {
      "inputs": cleaned_input,
      "parameters": {"max_length": int(max_len), "min_length": int(min_len)},
  }

  headers = {"Content-Type": "application/json"}

  # Add Bearer Token if configured in Render environment variables
  hf_token = os.environ.get("HF_TOKEN")
  if hf_token:
    headers["Authorization"] = f"Bearer {hf_token}"

  try:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers=headers)

    with urllib.request.urlopen(req, timeout=30) as response:
      res_body = response.read().decode("utf-8")
      result = json.loads(res_body)

      if isinstance(result, list) and len(result) > 0:
        return result[0].get("summary_text", "")
      elif isinstance(result, dict):
        if "summary_text" in result:
          return result["summary_text"]
        if "error" in result:
          return f"Inference API Error: {result['error']}"

      return str(result)

  except urllib.error.HTTPError as e:
    error_body = e.read().decode("utf-8")
    return f"HTTP Error {e.code}: {error_body}"
  except Exception as e:
    return f"Request Error: {str(e)}"


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
  stats = (
      f"Original: {orig_word_count} words | Abstractive Summary:"
      f" {abs_word_count} words"
  )

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
