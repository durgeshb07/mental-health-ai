import streamlit as st
from transformers import pipeline


# ======================
# Cached emotion model
# ======================

@st.cache_resource
def load_emotion_model():

    model = pipeline(

        "text-classification",

        model="j-hartmann/emotion-english-distilroberta-base",

        top_k=None

    )

    return model


emotion_model = load_emotion_model()


# ======================
# Detect emotions
# ======================

def detect_emotions(text):

    results = emotion_model(
        text
    )[0]

    emotions = {}

    for item in results:

        emotions[
            item["label"]
        ] = round(

            item["score"],

            3

        )

    return emotions