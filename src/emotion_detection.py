"""Emotion detection using the NRC Word-Emotion Association Lexicon (EmoLex).

The lexicon maps ~14,000 English words to 8 emotions
(anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
plus positive/negative sentiment, annotated manually via crowdsourcing.

NLP techniques used on top of the raw lexicon:
  - tokenization + lowercase normalization (via preprocess.clean_text)
  - lemmatization (each token's lemma is also looked up, so "loved" -> "love")
  - negation handling (a negator flips/shapes the emotion of the next token)
  - intensifier weighting ("very angry" scores higher than "angry")

Lexicon file: data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt
Source: NRC Emotion Lexicon (Saif M. Mohammad), version 0.92.
"""
import os

from preprocess import clean_text, _lemmatizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON_PATH = os.path.join(ROOT, "data", "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]

_NEGATORS = {"not", "no", "never", "neither", "nobody", "nothing", "nowhere",
             "cannot", "cant", "can't", "dont", "don't", "doesnt", "doesn't",
             "didnt", "didn't", "isnt", "isn't", "wasnt", "wasn't", "werent",
             "weren't", "wont", "won't", "hardly", "barely", "without"}

_INTENSIFIERS = {"very": 1.5, "really": 1.5, "extremely": 2.0, "so": 1.3,
                 "totally": 1.5, "absolutely": 1.8, "incredibly": 1.8,
                 "deeply": 1.6, "utterly": 1.8, "highly": 1.4}
_DOWNTONERS = {"slightly": 0.5, "somewhat": 0.6, "barely": 0.4,
               "kind": 0.6, "kinda": 0.6, "mildly": 0.5, "a": 0.7}


def load_emolex(path: str = LEXICON_PATH) -> dict:
    """Load EmoLex into {word: {emotion: 1, ...}} (only associations == 1)."""
    lex = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header line
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, assoc = parts[0].lower(), parts[1], parts[2].strip()
            if assoc == "1":
                lex.setdefault(word, {})[emotion] = 1
    return lex


_LEXICON = None


def get_lexicon() -> dict:
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = load_emolex()
    return _LEXICON


def detect_emotions(text: str, lexicon: dict = None) -> dict:
    """Return {emotion: score} for the 8 basic emotions, plus positive/negative
    sentiment counts. Scores are weighted token counts."""
    if lexicon is None:
        lexicon = get_lexicon()

    tokens = clean_text(text).split()
    scores = {e: 0.0 for e in EMOTIONS}
    scores["positive"] = 0.0
    scores["negative"] = 0.0

    weight, negate = 1.0, False
    matched = 0
    for tok in tokens:
        if tok in _NEGATORS:
            negate = not negate
            weight = 1.0
            continue
        if tok in _INTENSIFIERS:
            weight *= _INTENSIFIERS[tok]
            continue
        if tok in _DOWNTONERS:
            weight *= _DOWNTONERS[tok]
            continue

        # look up the token and its lemmas (EmoLex words are lemmas: love, not loved)
        entry = lexicon.get(tok)
        if not entry:
            for pos in ("v", "a", "s", "r", "n"):
                lemma = _lemmatizer.lemmatize(tok, pos=pos)
                if lexicon.get(lemma):
                    entry = lexicon[lemma]
                    break
        if entry:
            matched += 1
            factor = -1.0 if negate else 1.0  # negation flips valence; emotions still fire but reduced
            for emotion in list(entry):
                if emotion in ("positive", "negative") and negate:
                    # simple valence flip for sentiment under negation
                    flipped = "negative" if emotion == "positive" else "positive"
                    scores[flipped] += weight
                else:
                    scores[emotion] += weight * (0.5 if (negate and emotion in EMOTIONS) else 1.0)
            _ = factor
        weight = 1.0  # modifiers apply only to the next token
        negate = False if tok not in _NEGATORS else negate

    return scores


def dominant_emotion(text: str, lexicon: dict = None) -> tuple:
    """Return (dominant_emotion_or_'neutral', score, full_scores)."""
    scores = detect_emotions(text, lexicon)
    best = max(EMOTIONS, key=lambda e: scores[e])
    if scores[best] == 0:
        return "neutral", 0.0, scores
    return best, scores[best], scores


if __name__ == "__main__":
    tests = [
        "I am so happy and grateful, this made my day!",
        "This horrible betrayal makes me furious and disgusted.",
        "I'm terrified of what might happen tomorrow...",
        "The report was submitted on Tuesday.",
        "I was not happy at all, very disappointed and sad.",
    ]
    for t in tests:
        dom, sc, full = dominant_emotion(t)
        print(f"{t!r}\n  -> dominant: {dom} ({sc})  {full}\n")
