import re
import emoji
import spacy
import os
import nltk

from nltk.corpus import stopwords


# ======================
# Download NLTK resources if missing
# ======================

try:
    stop_words = set(
        stopwords.words("english")
    )

except LookupError:
    nltk.download("stopwords")

    stop_words = set(
        stopwords.words("english")
    )


# ======================
# Load SpaCy model safely
# ======================

try:
    nlp = spacy.load("en_core_web_sm")

except:
    os.system("python -m spacy download en_core_web_sm")

    nlp = spacy.load("en_core_web_sm")


# ======================
# Custom stop words
# ======================

custom_stopwords = {

    "im","ive","id","dont","cant",
    "didnt","doesnt","isnt","wasnt",
    "thing","things","people",
    "friend","know","really",
    "one","today","day","year"
}

stop_words.update(custom_stopwords)


# ======================
# Text Cleaning
# ======================

def clean_text(text):
    if not isinstance(text,str):
        return ""

    text=text.lower()

    text=re.sub(r"http\S+","",text)

    text=re.sub(r"@\w+","",text)

    text=emoji.replace_emoji(text,replace="")

    text=re.sub(r"[^a-zA-Z ]","",text)

    text=re.sub(r"\s+"," ",text).strip()


    if text=="":
        return ""

    doc=nlp(text)

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