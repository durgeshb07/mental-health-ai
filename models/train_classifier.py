import sys
import os

# Project path fix
project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(project_root)

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC

from preprocessing.clean_text import clean_text
from feature_engineering.tfidf_features import (
    TfidfFeatureExtractor
)


# ======================
# Load dataset
# ======================

df = pd.read_csv(
    "data/raw/mental_health.csv"
)

print("Columns:")
print(df.columns)


# ======================
# Create clean_text
# ======================

df = df.dropna()

# Change "text" below if your dataset uses another name
df["clean_text"] = df["text"].apply(
    clean_text
)

df = df[
    df["clean_text"] != ""
]


# ======================
# Features
# ======================

X_text = df["clean_text"]

y = df["status"]

# ======================
# Label Encoding
# ======================

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(
    df["status"]
)

print("\nClasses found:")

for i,label in enumerate(
    encoder.classes_
):

    print(
        f"{label} -> {i}"
    )


# ======================
# TF-IDF
# ======================

from feature_engineering.embeddings import (
    EmbeddingExtractor
)

extractor=EmbeddingExtractor()

X=extractor.encode(
    X_text.tolist()
)

# ======================
# Train/Test Split
# ======================

X_train,X_test,y_train,y_test=\
train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# ======================
# Linear SVM
# ======================

model = LinearSVC(
    C=2.0,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    pred
)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

# Classification Report

from sklearn.metrics import classification_report

print("\nClassification Report:\n")

print(

    classification_report(

        y_test,

        pred

    )

)

# ======================
# Save model
# ======================

import os

os.makedirs(
    "models/trained",
    exist_ok=True
)

with open(
    "models/trained/risk_classifier.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )

print(
    "Model saved successfully"
)