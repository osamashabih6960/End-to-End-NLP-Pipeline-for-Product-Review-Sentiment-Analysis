"""
feature_engineering.py
-----------------------
Step 3: FEATURE ENGINEERING

Features created:
    1. TF-IDF over unigrams + bigrams (captures "not good" as one token
       thanks to the negation tagging done in preprocessing, plus general
       bigram context) -> main signal for the linear model.
    2. Meta / hand-crafted features appended as extra numeric columns:
         - review length (word count)
         - exclamation mark count
         - uppercase-word ratio (shouting / emphasis signal)
         - lexicon polarity score (simple positive/negative word counter)
    These are combined into a single sparse feature matrix using
    scipy.sparse.hstack so the whole thing plugs into any sklearn estimator.
"""

import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess

POS_LEXICON = {"good", "great", "excellent", "love", "amazing", "best", "happy", "fantastic"}
NEG_LEXICON = {"bad", "terrible", "worst", "hate", "poor", "useless", "disappointed", "angry"}


def meta_features(raw_texts):
    feats = []
    for t in raw_texts:
        t = str(t)
        words = t.split()
        n_words = max(len(words), 1)
        excl = t.count("!")
        upper_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / n_words
        lower_words = set(w.strip(string_punct()) for w in t.lower().split())
        pos_score = len(lower_words & POS_LEXICON)
        neg_score = len(lower_words & NEG_LEXICON)
        feats.append([n_words, excl, upper_ratio, pos_score, neg_score])
    return np.array(feats, dtype=float)


def string_punct():
    import string
    return string.punctuation


class FeatureBuilder:
    """Wraps a TF-IDF vectorizer + meta features into one fit/transform API."""

    def __init__(self, max_features=6000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess,
            tokenizer=str.split,      # text already tokenised by `preprocess`
            token_pattern=None,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
        )
        self._meta_mean = None
        self._meta_std = None

    def fit_transform(self, raw_texts):
        tfidf = self.vectorizer.fit_transform(raw_texts)
        meta = meta_features(raw_texts)
        self._meta_mean = meta.mean(axis=0)
        self._meta_std = meta.std(axis=0) + 1e-6
        meta_scaled = (meta - self._meta_mean) / self._meta_std
        return hstack([tfidf, csr_matrix(meta_scaled)]).tocsr()

    def transform(self, raw_texts):
        tfidf = self.vectorizer.transform(raw_texts)
        meta = meta_features(raw_texts)
        meta_scaled = (meta - self._meta_mean) / self._meta_std
        return hstack([tfidf, csr_matrix(meta_scaled)]).tocsr()
