from feature_engineering.critical_detector import (
    detect_critical
)

from feature_engineering.sentiment_features import (
    get_sentiment
)

from feature_engineering.emotion_features import (
    detect_emotions
)


# ======================
# Risk Calculation
# ======================

def calculate_risk(text):

    sentiment = get_sentiment(
        text
    )

    emotions = detect_emotions(
        text
    )

    critical_matches = detect_critical(
        text
    )

    risk = 0


    # ======================
    # Sentiment contribution
    # ======================

    if sentiment["label"]=="NEGATIVE":

        risk += 30


    # ======================
    # Emotion contribution
    # ======================

    sadness = emotions.get(
        "sadness",
        0
    )

    fear = emotions.get(
        "fear",
        0
    )

    anger = emotions.get(
        "anger",
        0
    )

    risk += sadness*30
    risk += fear*15
    risk += anger*10


    # ======================
    # Critical phrases
    # ======================

    if len(
        critical_matches
    )>0:

        risk += 40


    # Keep score within range

    risk = min(
        round(risk),
        100
    )


    # ======================
    # Risk level
    # ======================

    if risk<30:

        level="LOW"

    elif risk<70:

        level="MEDIUM"

    else:

        level="HIGH"


    return {

        "risk_score":risk,

        "risk_level":level,

        "sentiment":sentiment,

        "emotions":emotions,

        "critical_matches":critical_matches

    }