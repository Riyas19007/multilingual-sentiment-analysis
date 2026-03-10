import streamlit as st
from transformers import pipeline
import langid
import pycountry
from deep_translator import GoogleTranslator

# -----------------------------
# Load Sentiment Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

classifier = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌍 Multilingual Emotion & Sentiment Analyzer")

text = st.text_area("Enter your review in any language:")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text!")
    else:

        # ✅ 1. Detect Language
        lang_code, confidence = langid.classify(text)

        language = pycountry.languages.get(alpha_2=lang_code)
        full_lang = language.name if language else "Unknown"

        # ✅ 2. Translate to English
        try:
            translated_text = GoogleTranslator(
                source='auto',
                target='en'
            ).translate(text)
        except:
            translated_text = text

        # ✅ 3. Sentiment Analysis (on English text)
        result = classifier(translated_text)[0]
        stars = result["label"]
        score = result["score"]
        
        # Convert Stars → Emotion
        if stars in ["4 stars", "5 stars"]:
            sentiment = "Positive 😊"
            emotion = "Happy / Satisfaction"
        elif stars == "3 stars":
            sentiment = "Neutral 😐"
            emotion = "Calm / Normal"
        else:
            sentiment = "Negative 😡"
            emotion = "Anger / Disappointment"

        # ✅ 4. Display Results
        st.success("Analysis Complete ✅")

        st.write("### 🌐 Language Information")
        st.write("Detected Language:", full_lang)
        st.write("Detection Confidence:", round(confidence, 2))

        st.write("### 🔄 Translation")
        st.write("Original Text:", text)
        st.write("Translated (English):", translated_text)

        st.write("### 😊 Emotion & Sentiment")
        st.write("Sentiment:", sentiment)
        st.write("Emotion:", emotion)
        st.write("Rating:", stars)
        st.write("Model Confidence:", round(score, 2))
