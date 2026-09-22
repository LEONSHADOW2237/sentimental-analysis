"""Text preprocessing utilities for sentiment analysis."""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    _stopwords = set(stopwords.words("english"))
except LookupError:  # pragma: no cover
    nltk.download("stopwords")
    _stopwords = set(stopwords.words("english"))

# Keep negation words - they carry sentiment
_NEGATIONS = {"no", "not", "nor", "neither", "never", "n't", "won't", "don't",
              "couldn't", "shouldn't", "wouldn't", "isn't", "aren't", "wasn't"}
_stopwords = _stopwords - _NEGATIONS

_lemmatizer = WordNetLemmatizer()
_url_re = re.compile(r"https?://\S+|www\.\S+")
_html_re = re.compile(r"<.*?>")
_mention_re = re.compile(r"@\w+")
_non_alpha_re = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, strip URLs/HTML/mentions/punctuation."""
    text = str(text).lower()
    text = _url_re.sub(" ", text)
    text = _html_re.sub(" ", text)
    text = _mention_re.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _non_alpha_re.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_and_lemmatize(text: str) -> str:
    """Clean, tokenize, remove stopwords and lemmatize. Returns a string of tokens."""
    tokens = clean_text(text).split()
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 1]
    return " ".join(_lemmatizer.lemmatize(t) for t in tokens)


if __name__ == "__main__":
    sample = "I LOVED this movie!!! It was not boring at all. Check https://example.com"
    print("clean:           ", clean_text(sample))
    print("tokenized+lemma: ", tokenize_and_lemmatize(sample))
