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
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"


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

  # Retrieve token from Streamlit Secrets or local environment variables
  hf_token = None
  if "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]
  elif os.environ.get("HF_TOKEN"):
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
    if e.code == 401:
      return (
          "API Error 401 (Unauthorized): Please add your HF_TOKEN into Streamlit"
          " Secrets or environment variables."
      )
    return f"API Status {e.code}: Request failed or model is loading."
  except Exception as e:
    return f"Request Error: {str(e)}"


# Streamlit UI Configuration
st.set_page_config(
    page_title="Text Summarization System", page_icon="📝", layout="wide"
)

st.title(" Text Summarization System")
st.caption(
    "Compare extractive baseline summaries against abstractive DistilBART API"
    " generation."
)

col1, col2 = st.columns(2)

with col1:
  user_text = st.text_area(
      "Document Input", height=250, placeholder="Paste text here to summarize..."
  )
  max_words = st.slider(
      "Max Summary Length (Words)",
      min_value=30,
      max_value=250,
      value=120,
      step=10,
  )
  submit_btn = st.button("Generate Summary", type="primary")

with col2:
  if submit_btn:
    if not user_text.strip():
      st.warning("Please enter text to summarize.")
    else:
      with st.spinner("Processing summaries..."):
        max_len = int(max_words)
        min_len = max(10, int(max_len * 0.3))

        extractive_res = generate_extractive_baseline(user_text)
        abstractive_res = generate_abstractive_summary(
            user_text, max_len=max_len, min_len=min_len
        )

        orig_count = len(user_text.split())
        abs_count = (
            len(abstractive_res.split())
            if not abstractive_res.startswith("API Error")
            else 0
        )
        ext_count = len(extractive_res.split())

        st.subheader("Abstractive BART Summary")
        if abstractive_res.startswith("API Error") or abstractive_res.startswith(
            "API Status"
        ):
          st.error(abstractive_res)
        else:
          st.text_area(
              "Abstractive Result",
              value=abstractive_res,
              height=120,
              label_visibility="collapsed",
          )

        st.subheader("Extractive Baseline Summary")
        st.text_area(
            "Extractive Result",
            value=extractive_res,
            height=120,
            label_visibility="collapsed",
        )

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Original Length", f"{orig_count} words")
        m2.metric("Abstractive Length", f"{abs_count} words")
        m3.metric("Extractive Length", f"{ext_count} words")
