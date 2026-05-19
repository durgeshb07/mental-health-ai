import re
import emoji
import spacy
import os

from nltk.corpus import stopwords

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp=spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))

custom_stopwords = {

    "im","ive","id","dont","cant",
    "didnt","doesnt","isnt","wasnt",
    "thing","things","people",
    "friend","know","really",
    "one","today","day","year"
}

stop_words.update(custom_stopwords)


def clean_text(text):

    if not isinstance(text,str):
        return ""

    text=text.lower()

    text=re.sub(
        r"http\S+",
        "",
        text
    )

    text=re.sub(
        r"@\w+",
        "",
        text
    )

    text=emoji.replace_emoji(
        text,
        replace=""
    )

    text=re.sub(
        r"[^a-zA-Z ]",
        "",
        text
    )

    doc=nlp(text)

    words=[]

    for token in doc:

        if (
            token.text not in stop_words
            and token.pos_ in [
                "NOUN",
                "ADJ",
                "VERB"
            ]
            and len(token.text)>2
        ):

            words.append(
                token.lemma_
            )

    return " ".join(words)