import nltk
import streamlit as st
from transformers import pipeline

# Download sentence tokenizer
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Cache the model so it loads into memory only once
@st.cache_resource
def load_model():
  return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


summarizer = load_model()


def clean_text(text: str) -> str:
  text = text.replace("\n", " ")
  return " ".join(text.split()).strip()


def generate_extractive_baseline(text: str, num_sentences: int = 2) -> str:
  cleaned = clean_text(text)
  sentences = nltk.sent_tokenize(cleaned)
  if not sentences:
    return ""
  return " ".join(sentences[:num_sentences])


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
        cleaned_input = clean_text(user_text)

        # Extractive Summary
        extractive_res = generate_extractive_baseline(user_text)

        # Abstractive Summary
        abs_res = summarizer(
            cleaned_input,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        abstractive_res = abs_res[0]["summary_text"]

        # Metrics
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
