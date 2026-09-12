"""
preprocessing.py
-----------------
Step 2: TEXT PREPARATION

2.1 Cleaning steps:
    - Lowercasing
    - HTML tag removal
    - URL removal
    - Emoji -> sentiment-word mapping (kept as signal, not discarded)
    - Punctuation / special character normalisation
    - Whitespace normalisation
    - Basic typo repair for a known noise dictionary (demo dataset only)

2.2 Preprocessing:
    - Tokenisation
    - Stopword removal (NEGATION-AWARE: "not", "no", "never" etc. are preserved
      because they flip sentiment polarity - naive stopword removal is a
      classic bug in sentiment pipelines)
    - Lightweight suffix-stripping stemmer (pure python, no external corpus
      downloads needed so the pipeline stays fully offline/reproducible)
    - Negation scope tagging: "not good" -> "not_good" so n-gram/BoW models
      can treat the negated phrase as its own unit

2.3 "Is advanced preprocessing required?"
    - For classic ML (TF-IDF + Logistic Regression/SVM) -> YES, the steps
      above (negation handling, emoji mapping) measurably help.
    - For transformer models (BERT/DistilBERT) -> NO, heavy cleaning can
      hurt; the subword tokenizer + model already model raw text well.
      Only mild cleaning (strip HTML/URLs) should be applied in that case.
      See `clean_for_transformer()` below.
"""

import re
import string

NEGATION_WORDS = {"not", "no", "never", "n't", "cannot", "cant", "dont", "isnt", "wasnt", "wont"}

# Minimal, dependency-free stopword list (avoids requiring nltk corpus downloads)
STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "of", "to", "in", "on", "at", "for",
    "with", "and", "or", "but", "it", "its", "as", "by", "from", "so", "very",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "my", "your", "our", "me", "us", "do", "does", "did", "just", "than",
} - NEGATION_WORDS  # never strip negations

EMOJI_MAP = {
    "😊": " happy ", "🔥": " excellent ", "👍": " good ", "❤️": " love ",
    "😡": " angry ", "👎": " bad ", "😢": " sad ",
}

TYPO_FIX = {"teh": "the", "realy": "really", "produtc": "product", "gr8": "great"}


def _demojize(text: str) -> str:
    for emo, word in EMOJI_MAP.items():
        text = text.replace(emo, word)
    return text


def clean_text(text: str) -> str:
    """Full cleaning pipeline for classic ML models."""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)         # URLs
    text = _demojize(text)
    text = text.lower()
    text = re.sub(r"n't\b", " not", text)                 # don't -> do not
    text = re.sub(r"[^a-z0-9_\s]", " ", text)             # punctuation/specials
    text = re.sub(r"\s+", " ", text).strip()
    words = [TYPO_FIX.get(w, w) for w in text.split()]
    return " ".join(words)


def clean_for_transformer(text: str) -> str:
    """Minimal cleaning appropriate before feeding a BERT-style tokenizer."""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = _demojize(text)
    return re.sub(r"\s+", " ", text).strip()


def _simple_stem(word: str) -> str:
    """Lightweight suffix-stripping stemmer (Porter-style, dependency-free)."""
    for suf in ("ational", "tional", "ing", "edly", "ed", "ies", "ied", "es", "s", "ly"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokenize_and_tag_negation(cleaned_text: str):
    """
    Tokenise + apply negation scope tagging + stopword removal + stemming.
    'not good product' -> ['not_good', 'produt']
    """
    tokens = cleaned_text.split()
    out = []
    negate_next = False
    for tok in tokens:
        if tok in NEGATION_WORDS:
            negate_next = True
            out.append(tok)
            continue
        if tok in STOPWORDS:
            negate_next = False
            continue
        stemmed = _simple_stem(tok)
        if negate_next:
            out.append(f"not_{stemmed}")
            negate_next = False
        else:
            out.append(stemmed)
    return out


def preprocess(text: str) -> str:
    """End-to-end: clean -> tokenize -> negation-tag -> stem -> rejoin."""
    cleaned = clean_text(text)
    tokens = tokenize_and_tag_negation(cleaned)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "This produtc is NOT good!! I don't like it at all 😡 http://x.com"
    print("raw       :", sample)
    print("cleaned   :", clean_text(sample))
    print("processed :", preprocess(sample))
