import os
import time
from huggingface_hub import InferenceClient
import gradio as gr
import nltk

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
        # Pass max_length and min_length directly as keyword arguments
        response = client.summarization(
            cleaned_input,
            model=MODEL_NAME,
            max_length=int(max_len),
            min_length=int(min_len),
        )
        # Handle string response or response object depending on client version
        if isinstance(response, str):
            return response
        return getattr(response, "summary_text", str(response))
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
