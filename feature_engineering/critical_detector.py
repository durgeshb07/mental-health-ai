critical_phrases = [

    "want to disappear",
    "no reason to live",
    "done with everything",
    "want to die",
    "end my life",
    "kill myself",
    "hopeless"
]


def detect_critical(text):
    text = text.lower()
    matches=[]
    for phrase in critical_phrases:
        if phrase in text:
            matches.append(
                phrase
            )

    return matches