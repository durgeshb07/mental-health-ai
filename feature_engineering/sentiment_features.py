from transformers import pipeline

# Load sentiment model once
sentiment_model = pipeline(
    "sentiment-analysis"
)


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