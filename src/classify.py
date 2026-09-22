"""Classify text as positive, negative, or neutral.

Combines two methods and reports an agreement-checked final label:
 1. VADER (rule-based lexicon)  -> label from compound score
 2. TF-IDF + Logistic Regression -> label + confidence (binary pos/neg)

Usage:
  # single text
  .\\.venv\\Scripts\\python.exe src\\classify.py "The movie was fantastic!"

  # a CSV (add --column if the text column is not named 'text')
  .\\.venv\\Scripts\\python.exe src\\classify.py data\\sample_reviews.csv --column text

  # read text lines from stdin
  Get-Content reviews.txt | .\\.venv\\Scripts\\python.exe src\\classify.py -
"""
import argparse
import csv
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import tokenize_and_lemmatize  # noqa: E402
from vader_sentiment import vader_label, vader_scores  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")


def load_ml():
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def classify(text: str, ml=None) -> dict:
    """Classify one text. Returns labels from both methods plus a final label.

    Final label logic:
      - if both methods agree, use it
      - VADER handles neutral (its compound is near 0); the ML model is binary,
        so a VADER 'neutral' wins only when the ML confidence is low (< 0.75)
      - otherwise the higher-confidence source decides (ML wins ties)
    """
    v_label = vader_label(text)
    compound = vader_scores(text)["compound"]

    ml_label, ml_conf = None, 0.0
    try:
        if ml is None:
            ml = load_ml()
        model, vectorizer = ml
        tfidf = vectorizer.transform([tokenize_and_lemmatize(text)])
        pred = model.predict(tfidf)[0]
        ml_label = "positive" if pred == "pos" else "negative"
        ml_conf = float(model.predict_proba(tfidf).max())
    except FileNotFoundError:
        pass

    if ml_label is None:
        final = v_label
    elif v_label == ml_label:
        final = v_label
    elif v_label == "neutral":
        final = "neutral" if ml_conf < 0.75 else ml_label
    else:
        final = ml_label  # disagreement on pos vs neg: trust the trained model

    return {
        "text": text,
        "vader_label": v_label,
        "vader_compound": round(compound, 3),
        "ml_label": ml_label,
        "ml_confidence": round(ml_conf, 3) if ml_label else None,
        "final_label": final,
    }


def main():
    ap = argparse.ArgumentParser(description="Classify text as positive/negative/neutral")
    # Windows consoles default to cp1252; reconfigure for unicode text
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap.add_argument("input", help='Text in quotes, path to a CSV, or "-" for stdin lines')
    ap.add_argument("--column", default="text", help="Text column name for CSV input")
    args = ap.parse_args()

    ml = None
    try:
        ml = load_ml()
    except FileNotFoundError:
        print("[warn] models/ not found - using VADER only. Run src/train_model.py\n",
              file=sys.stderr)

    rows = []
    if args.input == "-":
        rows = [classify(line.strip(), ml) for line in sys.stdin if line.strip()]
    elif os.path.isfile(args.input):
        with open(args.input, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if args.column not in reader.fieldnames:
                sys.exit(f"Column '{args.column}' not found. Available: {reader.fieldnames}")
            rows = [classify(str(r[args.column]), ml) for r in reader]
    else:
        rows = [classify(args.input, ml)]

    # print
    if rows[0]["ml_label"] is not None:
        print(f"{'final':10} {'vader':10} {'compound':>9} {'ml':10} {'conf':>6}  text")
        print("-" * 80)
        for r in rows:
            print(f"{r['final_label']:10} {r['vader_label']:10} "
                  f"{r['vader_compound']:>9} {r['ml_label']:10} {r['ml_confidence']:>6.0%}  "
                  f"{r['text'][:45]}")
    else:
        for r in rows:
            print(f"{r['final_label']:10} {r['vader_compound']:>9}  {r['text'][:60]}")

    # summary
    counts = {}
    for r in rows:
        counts[r["final_label"]] = counts.get(r["final_label"], 0) + 1
    print("\nSummary:", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(),
          key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
