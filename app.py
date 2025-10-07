import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from textblob import TextBlob
import random
import re

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Load External CSS ----------------
def load_css(file_name="style.css"):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_name}")

load_css("style.css")

# ---------------- Cache Models ----------------
@st.cache_resource(show_spinner=False)
def _load_models_cached():
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return sentiment_pipe, tokenizer, model


# ---------------- Text Generation Class ----------------
class SentimentTextGenerator:
    def __init__(self, fast_mode=False, defer_load=False):
        self.sentiment_analyzer = None
        self.tokenizer = None
        self.model = None
        self.fast_mode = fast_mode
        if not defer_load and not fast_mode:
            self.load_models()

    def load_models(self):
        with st.spinner("Loading models..."):
            try:
                self.sentiment_analyzer, self.tokenizer, self.model = _load_models_cached()
                st.success("Models loaded successfully!")
            except Exception as e:
                self.fast_mode = True
                st.warning(f"Switched to fast mode. Reason: {e}")

    def analyze_sentiment(self, text):
        if self.fast_mode:
            return self._fallback_sentiment(text)
        try:
            result = self.sentiment_analyzer(text)[0]
            label = result["label"].upper()
            score = float(result["score"])
            if score < 0.6:
                label = "NEUTRAL"
            return {"label": label, "confidence": score}
        except Exception:
            return self._fallback_sentiment(text)

    def _fallback_sentiment(self, text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return {"label": "POSITIVE", "confidence": polarity}
        elif polarity < -0.1:
            return {"label": "NEGATIVE", "confidence": abs(polarity)}
        else:
            return {"label": "NEUTRAL", "confidence": 0.5}

    def generate_text(self, prompt, sentiment, max_new_tokens=200, temperature=0.8):
        """
        Generate elaborative text that aligns with the detected sentiment.
        The AI elaborates on the prompt while keeping tone consistent with sentiment.
        """
        if self.fast_mode:
            return self._fallback_generation(prompt, sentiment)

        tone_guidance = {
            "POSITIVE": "Use inspiring, optimistic, and emotionally uplifting language.",
            "NEGATIVE": "Use thoughtful, empathetic, and emotionally honest language while maintaining depth.",
            "NEUTRAL": "Use balanced, factual, and steady language focused on clarity and context."
        }
        tone = tone_guidance.get(sentiment, "Use natural, human-like writing style.")

        instruction = (
            f"Analyze the sentiment of the topic and write a natural, coherent paragraph "
            f"that elaborates on it according to the detected sentiment.\n\n"
            f"Topic: {prompt}\n"
            f"Sentiment: {sentiment}\n"
            f"Guideline: {tone}\n\n"
            "Elaboration:"
        )

        try:
            inputs = self.tokenizer(instruction, return_tensors="pt", truncation=True)
            input_len = inputs["input_ids"].shape[1]
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_tokens = outputs[0][input_len:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            text = re.sub(r"^\s*Elaboration[:\-\s]*", "", text, flags=re.I).strip()
            return self._clean_text(text)
        except Exception:
            return self._fallback_generation(prompt, sentiment)

    def _fallback_generation(self, prompt, sentiment):
        base = prompt.strip().capitalize()
        if sentiment == "POSITIVE":
            text = (
                f"{base}. It reflects inspiration, motivation, and a sense of hope, "
                "highlighting how positivity can fuel creativity and growth."
            )
        elif sentiment == "NEGATIVE":
            text = (
                f"{base}. It explores the challenges, struggles, and emotions tied to the experience, "
                "emphasizing resilience and human strength."
            )
        else:
            text = (
                f"{base}. It presents a balanced elaboration that offers perspective and thoughtful detail."
            )
        return text

    def _clean_text(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        if not text.endswith(('.', '!', '?')):
            text += "."
        return text[0].upper() + text[1:]


# ---------------- Main Streamlit App ----------------
def main():
    st.markdown("<h1 style='text-align:center;'>🤖 AI Text Generator</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subheader' style='text-align:center; font-size:18px; color:white;'>"
        "Generate meaningful elaborations — expanding your idea, not explaining it."
        "</p>",
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        fast_mode = st.toggle("⚡ Fast Mode", value=True)
        if "generator" not in st.session_state:
            st.session_state.generator = SentimentTextGenerator(fast_mode=fast_mode, defer_load=True)
        else:
            st.session_state.generator.fast_mode = fast_mode

        theme_choice = st.selectbox("Theme", ["Light", "Dark"])
        if theme_choice == "Light":
            load_css("style.css")
        else:
            load_css("style_dark.css")

        manual_sentiment = st.selectbox("Sentiment", ["Auto-detect", "Positive", "Negative", "Neutral"])
        temperature = st.slider("Creativity", 0.1, 1.5, 0.8, 0.1)
        text_size = st.selectbox("Text Length", ["Short", "Medium", "Long"], index=1)

        # Map text length to max_new_tokens
        if text_size == "Short":
            max_tokens = 100
        elif text_size == "Medium":
            max_tokens = 200
        else:
            max_tokens = 300

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("<div class='box-wrapper'>", unsafe_allow_html=True)
        st.markdown("### 📝 Enter Your Prompt")
        user_prompt = st.text_area("Your text:", height=350, key="input_text")
        generate_clicked = st.button("✨ Generate Text", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='box-wrapper'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Generated Text")
        if generate_clicked and user_prompt:
            with st.spinner("Analyzing and generating..."):
                sentiment_result = st.session_state.generator.analyze_sentiment(user_prompt)
                if manual_sentiment != "Auto-detect":
                    sentiment_result["label"] = manual_sentiment.upper()
                gen = st.session_state.generator.generate_text(
                    user_prompt,
                    sentiment_result["label"],
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
            st.success(gen)
            st.caption(f"Detected Sentiment: {sentiment_result['label']}")
        else:
            st.info("👈 Enter your text and click Generate Text.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='footer-line'>", unsafe_allow_html=True)
    st.markdown("<p class='footer'>💡 Built with ❤️ using Streamlit and Transformers.</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

