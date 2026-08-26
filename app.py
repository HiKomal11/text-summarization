import os
import nltk
import requests
import streamlit as st
from rouge_score import rouge_scorer

# Download NLTK tokenizer resources safely
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"


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

    hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")

    if not hf_token:
        return "API Config Error: HF_TOKEN secret not found in Streamlit Cloud."

    headers = {"Authorization": f"Bearer {hf_token.strip()}"}
    payload = {
        "inputs": cleaned_input,
        "parameters": {
            "max_length": int(max_len),
            "min_length": int(min_len),
        },
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()

        if response.status_code == 200:
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("summary_text", "")
            elif isinstance(result, dict) and "summary_text" in result:
                return result["summary_text"]
            return str(result)
        
        elif response.status_code == 403:
            return (
                "API 403 (Forbidden): Ensure your Hugging Face token has "
                "'Make calls to Inference Providers' permission enabled."
            )
        elif response.status_code == 503:
            return "API Note: Model is currently loading on Hugging Face. Please try again in a few seconds."
        else:
            error_msg = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            return f"Inference Error ({response.status_code}): {error_msg}"

    except Exception as e:
        return f"Inference Error: {str(e)}"


def compute_rouge_scores(target: str, prediction: str):
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(target, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure * 100, 2),
        "rouge2": round(scores["rouge2"].fmeasure * 100, 2),
        "rougeL": round(scores["rougeL"].fmeasure * 100, 2),
    }


# --- Streamlit UI ---
st.set_page_config(
    page_title="Text Summarization System", page_icon="", layout="wide"
)

st.title(" Text Summarization & Evaluation System")
st.caption(
    "Compare extractive baseline and abstractive DistilBART summaries using"
    " ROUGE metrics."
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
            with st.spinner("Processing summaries & computing metrics..."):
                max_len = int(max_words)
                min_len = max(10, int(max_len * 0.3))

                extractive_res = generate_extractive_baseline(user_text)
                abstractive_res = generate_abstractive_summary(
                    user_text, max_len=max_len, min_len=min_len
                )

                is_error = abstractive_res.startswith(
                    "API"
                ) or abstractive_res.startswith("Inference Error")

                orig_count = len(user_text.split())
                abs_count = len(abstractive_res.split()) if not is_error else 0
                ext_count = len(extractive_res.split())

                st.subheader("Abstractive BART Summary")
                if is_error:
                    st.error(abstractive_res)
                else:
                    st.text_area(
                        "Abstractive Result",
                        value=abstractive_res,
                        height=110,
                        label_visibility="collapsed",
                    )

                st.subheader("Extractive Baseline Summary")
                st.text_area(
                    "Extractive Result",
                    value=extractive_res,
                    height=110,
                    label_visibility="collapsed",
                )

                st.divider()

                # Word Count Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Original Length", f"{orig_count} words")
                m2.metric("Abstractive Length", f"{abs_count} words")
                m3.metric("Extractive Length", f"{ext_count} words")

                # ROUGE Quantitative Evaluation
                if not is_error and extractive_res:
                    st.subheader(" ROUGE Evaluation (BART vs. Baseline)")
                    st.caption(
                        "Measures n-gram overlap F1 score of the abstractive summary"
                        " against the extractive reference baseline."
                    )

                    rouge_results = compute_rouge_scores(extractive_res, abstractive_res)

                    r1, r2, rL = st.columns(3)
                    r1.metric(
                        "ROUGE-1 (Unigram)",
                        f"{rouge_results['rouge1']}%",
                        help="Overlap of individual words.",
                    )
                    r2.metric(
                        "ROUGE-2 (Bigram)",
                        f"{rouge_results['rouge2']}%",
                        help="Overlap of word pairs (phrases).",
                    )
                    rL.metric(
                        "ROUGE-L (LCS)",
                        f"{rouge_results['rougeL']}%",
                        help="Longest Common Subsequence matching.",
                    )
