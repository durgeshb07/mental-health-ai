import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_sentiment_model():

    return pipeline(

        "sentiment-analysis"

    )


sentiment_model = load_sentiment_model()


def get_sentiment(text):

    result = sentiment_model(

        text

    )[0]

    return {

        "label": result["label"],

        "score": round(

            result["score"],
            3

        )
    }