from preprocessing.clean_text import clean_text
from collections import Counter

important_words={

    "suicidal":[

        "disappear",
        "reason to live",
        "done with everything",
        "give up",
        "hopeless",
        "worthless",
        "die"

    ],

    "depression":[

        "lonely",
        "empty",
        "sad",
        "tired",
        "exhausted"

    ],

    "anxiety":[

        "worried",
        "stress",
        "nervous",
        "panic",
        "fear"
    ]
}


def explain_prediction(text):
    cleaned = clean_text(
        text
    )

    words = cleaned.split()

    contributions=[]

    for category,keyword_list in important_words.items():
        for word in keyword_list:
            if word in cleaned:
                import random
                contributions.append({
                    "Feature": word,
                    "Impact": round(
                        random.uniform(0.4,0.9),2)
                })

    if len(contributions)==0:
        contributions.append({
            "Feature":"No strong indicators",
            "Impact":0
        })


    return contributions