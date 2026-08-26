import nltk
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Download NLTK sentence tokenizers quietly
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Load model and tokenizer directly to avoid Python 3.14 pipeline task registry errors
@st.cache_resource
def load_model_and_tokenizer():
  model_name = "sshleifer/distilbart-cnn-12-6"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


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

  # Tokenize input text
  inputs = tokenizer(
      cleaned_input,
      return_tensors="pt",
      max_length=1024,
      truncation=True,
  )

  # Generate summary sequence directly using model.generate()
  with torch.no_grad():
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        early_stopping=True,
    )

  # Decode model tokens back to text
  summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return summary_text


# Page Setup
st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")
st.title(" Text Summarization System")
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

        # Extractive Baseline Summary
        extractive_res = generate_extractive_baseline(user_text)

        # Abstractive BART Summary via Direct PyTorch Generation
        abstractive_res = generate_abstractive_summary(
            user_text, max_len=max_len, min_len=min_len
        )

        # Word Count Metrics
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
