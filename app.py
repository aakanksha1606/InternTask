import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from textblob import TextBlob
import re
import random

# Streamlit config
st.set_page_config(page_title="Sentiment-Based Text Generator", page_icon="🤖", layout="wide")

@st.cache_resource(show_spinner=False)
def _load_models_cached():
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return sentiment_pipe, tokenizer, model


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
            except Exception:
                self.fast_mode = True
                st.warning("Switched to fast mode (no heavy models).")

    def analyze_sentiment(self, text):
        if self.fast_mode:
            return self._fallback_sentiment(text)
        try:
            result = self.sentiment_analyzer(text)[0]
            label = result["label"].upper()
            score = result["score"]
            if score < 0.6:
                label = "NEUTRAL"
            return {"label": label, "confidence": score}
        except Exception:
            return self._fallback_sentiment(text)

    def _fallback_sentiment(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return {"label": "POSITIVE", "confidence": polarity}
        elif polarity < -0.1:
            return {"label": "NEGATIVE", "confidence": abs(polarity)}
        else:
            return {"label": "NEUTRAL", "confidence": 0.5}

    def generate_text(self, prompt, sentiment, text_size, temperature=0.8):
        """Generate essay text according to size."""
        size_settings = {
            "Short (50-100 words)": 100,
            "Medium (100-200 words)": 180,
            "Long (200+ words)": 250
        }
        max_length = size_settings.get(text_size, 180)

        if self.fast_mode:
            return self._fallback_generation(prompt, sentiment, text_size)

        try:
            intro = self._get_intro(sentiment)
            full_prompt = f"{intro} {prompt}"
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            text = text[len(full_prompt):].strip()
            return self._clean_text(text)
        except Exception:
            return self._fallback_generation(prompt, sentiment, text_size)

    def _get_intro(self, sentiment):
        options = {
            "POSITIVE": [
                "Life often surprises us with moments of pure joy and gratitude.",
                "There are days when optimism fills every corner of our mind.",
                "Positivity gives strength to see beauty even in simple things."
            ],
            "NEGATIVE": [
                "Sometimes the world feels heavy, and everything seems uncertain.",
                "There are moments when darkness clouds the heart and thoughts feel blurred.",
                "Life isn’t always kind; certain days test patience and hope alike."
            ],
            "NEUTRAL": [
                "Not every situation carries strong emotions; some moments simply are.",
                "Life moves steadily, presenting facts without extremes of joy or sorrow.",
                "There are days that pass quietly, neither bright nor dark, just balanced."
            ]
        }
        return random.choice(options.get(sentiment, options["NEUTRAL"]))

    def _fallback_generation(self, prompt, sentiment, text_size):
        """Generate essay-like text in fallback mode."""
        base = prompt.strip().capitalize()
        sentences = {
            "short": 3,
            "medium": 5,
            "long": 7
        }
        count = 5
        if "Short" in text_size:
            count = sentences["short"]
        elif "Long" in text_size:
            count = sentences["long"]

        if sentiment == "POSITIVE":
            parts = [
                f"{base}. This situation reflects hope, inspiration, and gratitude.",
                "Small victories remind us that growth takes time, but each effort matters.",
                "The energy of positivity builds resilience, helping people move forward.",
                "Even simple moments can feel meaningful when they are filled with appreciation.",
                "Ultimately, optimism becomes a quiet power that turns challenges into possibilities."
            ]
        elif sentiment == "NEGATIVE":
            parts = [
                f"{base}. It expresses frustration and emotional weight that often comes from struggle.",
                "In such times, patience feels thin and the world appears distant.",
                "Yet within every hardship lies a lesson waiting to be understood.",
                "Acknowledging pain does not make one weak; it opens space for healing.",
                "Over time, reflection transforms sorrow into strength and awareness."
            ]
        else:
            parts = [
                f"{base}. It represents a balanced observation rather than emotion.",
                "Things unfold as they are, and clarity comes from distance.",
                "Neutrality offers perspective, helping us see both sides without bias.",
                "Calm reflection allows ideas to form without rushing toward judgment.",
                "In the end, balance brings understanding and quiet confidence."
            ]
        return " ".join(parts[:count])

    def _clean_text(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        if not text.endswith(('.', '!', '?')):
            text += "."
        return text[0].upper() + text[1:]


def main():
    st.title("🤖 Sentiment-Based Text Generator")
    st.markdown("Generate **paragraphs or essays** aligned with the sentiment of your input prompt.")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        fast_mode = st.toggle("⚡ Fast mode", value=True)
        if "generator" not in st.session_state:
            st.session_state.generator = SentimentTextGenerator(fast_mode=fast_mode, defer_load=True)
        else:
            st.session_state.generator.fast_mode = fast_mode

        manual_sentiment = st.selectbox("Sentiment override:", ["Auto-detect", "Positive", "Negative", "Neutral"])
        temperature = st.slider("Creativity", 0.1, 1.5, 0.8, 0.1)
        text_size = st.selectbox(
            "Generated text size:",
            ["Short (50-100 words)", "Medium (100-200 words)", "Long (200+ words)"],
            index=1
        )

    st.subheader("📝 Input Prompt")
    user_prompt = st.text_area("Enter your prompt here:", height=100)
    generate_button = st.button("🚀 Generate Text", use_container_width=True)

    if user_prompt and generate_button:
        with st.spinner("Analyzing sentiment..."):
            sentiment_result = st.session_state.generator.analyze_sentiment(user_prompt)
            if manual_sentiment != "Auto-detect":
                sentiment_result["label"] = manual_sentiment.upper()

        st.markdown("---")
        st.subheader("📊 Sentiment Analysis Result")
        st.write(f"**Detected Sentiment:** {sentiment_result['label']}")
        st.write(f"**Confidence:** {sentiment_result['confidence']:.1%}")

        with st.spinner("Generating text..."):
            generated_text = st.session_state.generator.generate_text(
                user_prompt,
                sentiment_result["label"],
                text_size,
                temperature=temperature
            )

        st.markdown("---")
        st.subheader("🧠 Generated Text")
        st.write(generated_text)
        st.caption(f"Word count: {len(generated_text.split())}")

        if st.button("📋 Copy to Clipboard"):
            st.code(generated_text)
            st.success("Text copied!")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, Transformers, and PyTorch")


if __name__ == "__main__":
    main()

