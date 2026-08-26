import os
import nltk
import streamlit as st
from huggingface_hub import InferenceClient

# Initialize tokenizer safely
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Universally enabled chat model on the modern router tier
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


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

    # Retrieve token from secrets or environment
    hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")

    if not hf_token:
        return "API Config Error: HF_TOKEN secret not found in Streamlit Cloud."

    try:
        client = InferenceClient(token=hf_token.strip())

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional summarization assistant. Provide a clear, "
                    f"concise abstractive summary of the given text within a limit of {max_len} words."
                ),
            },
            {"role": "user", "content": cleaned_input},
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=int(max_len),
            temperature=0.3,
        )

        if response and response.choices:
            return response.choices[0].message.content.strip()

        return "Inference Error: Received empty response from model."

    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg or "Forbidden" in err_msg:
            return (
                "API 403 (Forbidden): Ensure your Hugging Face fine-grained token has"
                " 'Make calls to Inference Providers' permission enabled."
            )
        elif "503" in err_msg or "loading" in err_msg.lower():
            return "API Note: Model is currently loading on Hugging Face. Please try again in a few seconds."
        return f"Inference Error: {err_msg}"


# --- Streamlit Layout ---
st.set_page_config(
    page_title="Text Summarization System", page_icon="", layout="wide"
)

st.title(" Text Summarization System")
st.caption(
    "Compare extractive baseline summaries against abstractive AI-generated summaries."
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
                    if not abstractive_res.startswith("API")
                    and not abstractive_res.startswith("Inference Error")
                    else 0
                )
                ext_count = len(extractive_res.split())

                st.subheader("Abstractive AI Summary")
                if abstractive_res.startswith("API") or abstractive_res.startswith(
                    "Inference Error"
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
