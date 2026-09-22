"""Train a sentiment classifier (TF-IDF + Logistic Regression) on NLTK movie_reviews.

Saves model + vectorizer to models/ and writes metrics to models/metrics.txt.
Run:  .\\.venv\\Scripts\\python.exe src\\train_model.py
"""
import os
import pickle
import sys
import time

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import tokenize_and_lemmatize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")


def load_dataset():
    """Load NLTK movie_reviews as (texts, labels)."""
    try:
        from nltk.corpus import movie_reviews
        movie_reviews.categories()
    except LookupError:
        nltk.download("movie_reviews")
        from nltk.corpus import movie_reviews

    texts, labels = [], []
    for fileid in movie_reviews.fileids():
        texts.append(movie_reviews.raw(fileid))
        labels.append(movie_reviews.categories(fileid)[0])  # 'pos' / 'neg'
    return texts, np.array(labels)


def main():
    t0 = time.time()
    print("Loading movie_reviews corpus ...")
    texts, labels = load_dataset()
    print(f"  {len(texts)} documents")

    print("Preprocessing ...")
    clean = [tokenize_and_lemmatize(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        clean, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Vectorizing (TF-IDF) ...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression ...")
    model = LogisticRegression(max_iter=1000, C=4.0)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}\n")
    print(report)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}\nConfusion matrix (rows=true neg,pos):\n{cm}\n")

    print(f"Model saved to {MODEL_DIR}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
