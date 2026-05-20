from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class TfidfFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1,3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )

    def fit_transform(self,text):
        return self.vectorizer.fit_transform(
            texts
        )

    def transform(self,texts):
        return self.vectorizer.transform(
            texts
        )

    def save(self,path):
        with open(path,"wb") as f:
            pickle.dump(self.vectorizer,f)

    def load(self,path):
        with open(path,"rb") as f:
            self.vectorizer = pickle.load(f)