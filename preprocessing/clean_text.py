import re
import emoji
import spacy
import os
import nltk

from nltk.corpus import stopwords


# ======================
# Download NLTK resources
# ======================


nlp = spacy.load(
    "en_core_web_sm"
)

stop_words = set(
    stopwords.words(
        "english"
    )
)


# ======================
# Load SpaCy model safely
# ======================

try:
    nlp = spacy.load(
        "en_core_web_sm"
    )
except:
    os.system(
        "python -m spacy download en_core_web_sm"
    )
    nlp = spacy.load(
        "en_core_web_sm"
    )


# ======================
# Custom stop words
# ======================

custom_stopwords = {

    "im",
    "ive",
    "id",
    "dont",
    "cant",
    "didnt",
    "doesnt",
    "isnt",
    "wasnt",
    "thing",
    "things",
    "people",
    "friend",
    "know",
    "really",
    "one",
    "today",
    "day",
    "year"

}

stop_words.update(
    custom_stopwords
)


# ======================
# Text cleaning
# ======================

def clean_text(text):
    if not isinstance(
        text,
        str
    ):
        return ""


    text = text.lower()


    # Remove URLs
    text = re.sub(
        r"http\S+",
        "",
        text
    )


    # Remove usernames
    text = re.sub(
        r"@\w+",
        "",
        text
    )


    # Remove emojis
    text = emoji.replace_emoji(
        text,
        replace=""
    )


    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z ]",
        "",
        text
    )


    # Remove extra spaces
    text = re.sub(r"\s+"," ",text).strip()

    if text=="":
        return ""

    doc = nlp(text)
    words=[]

    for token in doc:
        if (
            token.text not in stop_words
            and
            token.pos_ in [
                "NOUN",
                "ADJ",
                "VERB"
            ]
            and
            len(token.text)>2
        ):

            words.append(token.lemma_)

    return " ".join(
        words
    )