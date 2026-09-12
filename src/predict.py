"""
predict.py
----------
Loads the trained model + feature builder and exposes a single
`predict_sentiment(text)` function returning label + class probabilities.
Used by the Flask API (app.py) that powers the 3D UI.
"""

import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

_clf = None
_fb = None


def _lazy_load():
    global _clf, _fb
    if _clf is None:
        _clf = joblib.load(os.path.join(MODEL_DIR, "sentiment_clf.joblib"))
        _fb = joblib.load(os.path.join(MODEL_DIR, "feature_builder.joblib"))
    return _clf, _fb


def predict_sentiment(text: str):
    clf, fb = _lazy_load()
    X = fb.transform([text])
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_
    label = str(classes[int(np.argmax(proba))])
    scores = {str(cls): float(p) for cls, p in zip(classes, proba)}
    return {"label": label, "scores": scores}


if __name__ == "__main__":
    for t in [
        "This product is absolutely amazing, I love it!",
        "Worst purchase ever, it broke immediately.",
        "It's okay, does the job, nothing special.",
    ]:
        print(t, "->", predict_sentiment(t))
