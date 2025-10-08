import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from textblob import TextBlob
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

# ---------------- Load GPT-2 Model ----------------
@st.cache_resource(show_spinner=False)
def load_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

tokenizer, model = load_gpt2()

# ---------------- Helper Functions ----------------
def analyze_sentiment(text):
    """CPU-only sentiment using TextBlob."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return {"label": "POSITIVE", "confidence": polarity}
    elif polarity < -0.1:
        return {"label": "NEGATIVE", "confidence": abs(polarity)}
    else:
        return {"label": "NEUTRAL", "confidence": 0.5}

def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith(('.', '!', '?')):
        text += "."
    return text[0].upper() + text[1:]

def fallback_generation(prompt, sentiment):
    base = prompt.strip().capitalize()
    if sentiment == "POSITIVE":
        return f"{base}. It continues with an optimistic and uplifting elaboration."
    elif sentiment == "NEGATIVE":
        return f"{base}. It continues with a reflective and thoughtful elaboration."
    else:
        return f"{base}. It continues with a neutral and balanced elaboration."

def generate_text(prompt, sentiment, max_new_tokens=200, temperature=0.8, text_size="Medium"):
    """Generate text with distilgpt2 and approximate word limit."""
    tone_map = {
        "POSITIVE": "optimistic, uplifting",
        "NEGATIVE": "reflective, thoughtful",
        "NEUTRAL": "neutral, balanced"
    }
    tone = tone_map.get(sentiment, "neutral, balanced")
    final_prompt = f"Extend the following text in a {tone} style:\n{prompt}"

    try:
        inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_tokens = outputs[0][input_len:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Limit output by approximate word count
        word_limits = {"Short": 50, "Medium": 100, "Long": 150}
        max_words = word_limits.get(text_size, 100)
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words])

        return clean_text(text)
    except Exception:
        return fallback_generation(prompt, sentiment)

# ---------------- Main App ----------------
st.markdown("<h1>🤖 AI Text Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Generate meaningful elaborations — expanding your idea.</p>", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.text_size = st.selectbox("Text Length", ["Short", "Medium", "Long"], index=1)
    manual_sentiment = st.selectbox("Sentiment", ["Auto-detect", "Positive", "Negative", "Neutral"])
    temperature = st.slider("Creativity", 0.1, 1.5, 0.8, 0.1)

# ---------------- Layout ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='box-wrapper'>", unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Prompt")
    user_prompt = st.text_area("Your text:", height=350)
    generate_clicked = st.button("✨ Generate Text")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='box-wrapper'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Generated Text")
    if generate_clicked and user_prompt:
        with st.spinner("Analyzing and generating..."):
            sentiment_result = analyze_sentiment(user_prompt)
            if manual_sentiment != "Auto-detect":
                sentiment_result["label"] = manual_sentiment.upper()
            gen_text = generate_text(
                user_prompt,
                sentiment_result["label"],
                max_new_tokens={"Short":100,"Medium":200,"Long":300}[st.session_state.text_size],
                temperature=temperature,
                text_size=st.session_state.text_size
            )
        st.success(gen_text)
        st.caption(f"Detected Sentiment: {sentiment_result['label']}")
    else:
        st.info("👈 Enter your text and click Generate Text.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr class='footer-line'>", unsafe_allow_html=True)
st.markdown("<p class='footer'>💡 Built with ❤️ using Streamlit and Transformers.</p>", unsafe_allow_html=True)
