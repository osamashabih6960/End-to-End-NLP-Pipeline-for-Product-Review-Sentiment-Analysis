"""
train_model.py
---------------
Step 4: MODELLING

Algorithm chosen: TF-IDF (+meta features) -> Logistic Regression
    - Strong, fast, interpretable baseline for short-text sentiment
    - Coefficients are inspectable per class (explainability for stakeholders)
    - In the writeup we also discuss LinearSVC / XGBoost / fine-tuned
      DistilBERT as the natural next steps once more real labelled data
      and GPU budget are available.

Intrinsic evaluation metrics reported:
    - Accuracy
    - Precision / Recall / F1 (macro + per-class, since classes matter
      individually for a 3-way sentiment task)
    - Confusion matrix

(Extrinsic metrics - business KPIs like CSAT, deflection rate, conversion
uplift - are discussed in the README since they require live A/B data
which does not exist in this offline demo.)
"""

import os
import sys
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(__file__))
from data_acquisition import acquire_data
from feature_engineering import FeatureBuilder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_data(path):
    import csv
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["sentiment"])
    return texts, labels


def main():
    if not os.path.exists(DATA_PATH):
        acquire_data(output_path=DATA_PATH, n_per_class=400)

    texts, labels = load_data(DATA_PATH)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"[train] train={len(X_train_text)} test={len(X_test_text)}")

    fb = FeatureBuilder(max_features=6000, ngram_range=(1, 2))
    X_train = fb.fit_transform(X_train_text)
    X_test = fb.transform(X_test_text)

    clf = LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    labels_sorted = sorted(set(labels))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print("\n=== INTRINSIC EVALUATION ===")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1       : {f1:.4f}")
    print("\nPer-class report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), labels =", labels_sorted)
    print(cm)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "sentiment_clf.joblib"))
    joblib.dump(fb, os.path.join(MODEL_DIR, "feature_builder.joblib"))

    metrics = {
        "accuracy": acc, "macro_precision": precision,
        "macro_recall": recall, "macro_f1": f1,
        "labels": labels_sorted, "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[train] Saved model + feature builder + metrics -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
