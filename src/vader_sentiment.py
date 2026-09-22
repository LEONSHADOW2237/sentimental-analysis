"""VADER (rule-based) sentiment analysis wrapper."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def vader_scores(text: str) -> dict:
    """Return dict with neg, neu, pos, compound scores."""
    return _analyzer.polarity_scores(str(text))


def vader_label(text: str) -> str:
    """Classify text as positive / negative / neutral using VADER compound score."""
    compound = vader_scores(text)["compound"]
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


if __name__ == "__main__":
    for s in ["I love this product!", "This is terrible.", "It is a table."]:
        print(f"{s!r:35} -> {vader_label(s):10} {vader_scores(s)}")
