import streamlit as st
from sentence_transformers import (
    SentenceTransformer
)


# ======================
# Cached embedding model
# ======================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(

        "all-MiniLM-L6-v2"

    )

    return model


# ======================
# Embedding extractor
# ======================

class EmbeddingExtractor:

    def __init__(self):

        self.model = load_embedding_model()


    def encode(

        self,
        texts

    ):

        embeddings = self.model.encode(

            texts,

            convert_to_numpy=True

        )

        return embeddings