import json
import os
import urllib.error
import urllib.request
import nltk
import streamlit as st

# Download NLTK tokenizer resources safely
try:
  nltk.data.find("tokenizers/punkt")
except LookupError:
  nltk.download("punkt", quiet=True)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
API_URL = (
    f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"
)


def clean_text(text: str) -> str:
  text = text.replace("\n", " ")
  return " ".join(text.split()).strip()


def generate_extractive_baseline(text: str, num_sentences: int = 2) -> str:
  cleaned = clean_text(text)
  try:
    sentences = nltk.sent_tokenize(cleaned)
  except Exception:
    sentences = cleaned.split(". ")
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
          return f"API Note: {result['error']}"

      return str(result)
  except urllib.error.HTTPError as e:
    return (
        f"API Status {e.code}: Token required for Serverless Router or model"
        " loading."
    )
  except Exception as e:
    return f"Request Error: {str(e)}"


# Streamlit UI
st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")
st.title("Text Summarization System")
st.write(
    "Paste text below to generate automated summaries using DistilBART and NLTK"
    " baselines."
)

col1, col2 = st.columns(2)

with col1:
  user_text = st.text_area(
      "Document Input", height=250, placeholder="Paste text here..."
  )
  max_words = st.slider("Max Summary Length", 30, 250, 120, step=10)
  submit_btn = st.button("Generate Summary", type="primary")

with col2:
  if submit_btn:
    if not user_text.strip():
      st.warning("Please enter text to summarize.")
    else:
      with st.spinner("Generating summaries..."):
        max_len = int(max_words)
        min_len = max(10, int(max_len * 0.3))

        extractive_res = generate_extractive_baseline(user_text)
        abstractive_res = generate_abstractive_summary(
            user_text, max_len=max_len, min_len=min_len
        )

        orig_count = len(user_text.split())
        abs_count = len(abstractive_res.split())

        st.text_area(
            "Abstractive BART Summary", value=abstractive_res, height=120
        )
        st.text_area(
            "Extractive Baseline Summary", value=extractive_res, height=120
        )
        st.info(
            f"Original: {orig_count} words | Abstractive Summary: {abs_count}"
            " words"
        )
