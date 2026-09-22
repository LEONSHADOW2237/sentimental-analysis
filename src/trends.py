"""Understand public opinion and trends through sentiment patterns.

Turns a classified dataset (any CSV with a 'text' column, e.g. output of
collect_data.py) into opinion insights:

  - sentiment_over_time : sentiment counts per day (trend lines)
  - opinion_by_source   : positive/negative/neutral split per source
  - keyword_drivers     : top words pushing opinion positive vs negative
  - emotion_profile     : overall emotion mix + top emotion per source
  - summary             : one-paragraph plain-language opinion summary

Usage:
  .\\.venv\\Scripts\\python.exe src\\trends.py data\\collected_iphone_16.csv
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from classify import classify as combined_classify, load_ml  # noqa: E402
from emotion_detection import EMOTIONS, detect_emotions, get_lexicon  # noqa: E402
from preprocess import clean_text, _lemmatizer  # noqa: E402

FILLER = _DE = None
try:
    from preprocess import _stopwords as _SW
except ImportError:  # pragma: no cover
    _SW = set()


def analyze_frame(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Ensure the frame has classification + emotion + time columns."""
    df = df.copy()
    if "final_label" not in df.columns:
        ml = None
        try:
            ml = load_ml()
        except FileNotFoundError:
            pass
        results = [combined_classify(str(t), ml) for t in df[text_col].astype(str)]
        df["final_label"] = [r["final_label"] for r in results]
        df["compound"] = [r["vader_compound"] for r in results]
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df.get("created_utc"), errors="coerce").dt.date
    lexicon = get_lexicon()
    if "dominant_emotion" not in df.columns:
        emo_rows = [detect_emotions(str(t), lexicon) for t in df[text_col].astype(str)]
        df["dominant_emotion"] = [
            max(EMOTIONS, key=lambda e: r[e]) if max(r[e] for e in EMOTIONS) > 0 else "neutral"
            for r in emo_rows
        ]
        for e in EMOTIONS:
            df[f"emo_{e}"] = [round(r[e], 1) for r in emo_rows]
    return df


def sentiment_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Daily counts of each sentiment label (trend over time)."""
    d = df.dropna(subset=["date"])
    pivot = (d.groupby(["date", "final_label"]).size().unstack(fill_value=0))
    return pivot.sort_index()


def sentiment_over_time_rolling(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Rolling-average share of positive vs negative sentiment (smoothed trend).

    Uses the *share* of positive (and negative) items per day so the trend is
    not distorted by volume spikes, smoothed over `window` periods.
    """
    pivot = sentiment_over_time(df)
    if pivot.empty:
        return pivot
    totals = pivot.sum(axis=1)
    shares = pd.DataFrame({
        "positive": pivot.get("positive", 0) / totals,
        "negative": pivot.get("negative", 0) / totals,
    })
    return shares.rolling(window=window, min_periods=1).mean()


def weekly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly counts of each sentiment label (better for sparse daily data)."""
    d = df.dropna(subset=["date"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["week"] = pd.to_datetime(d["date"]).dt.to_period("W").apply(lambda p: p.start_time.date())
    return (d.groupby(["week", "final_label"]).size()
            .unstack(fill_value=0).sort_index())


def momentum(df: pd.DataFrame, window: int = 7) -> dict:
    """Is opinion improving or deteriorating in the most recent period?

    Compares mean VADER compound in the last `window` days of data vs the
    preceding `window` days, plus the rising/falling share of negative posts.
    """
    d = df.dropna(subset=["date"]).copy()
    if d.empty:
        return {}
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    last_day = d["date"].max()
    recent = d[d["date"] > last_day - pd.Timedelta(days=window)]
    prior = d[(d["date"] > last_day - pd.Timedelta(days=2 * window)) &
              (d["date"] <= last_day - pd.Timedelta(days=window))]

    def neg_share(frame):
        if frame.empty:
            return None
        return float((frame["final_label"] == "negative").mean())

    recent_neg, prior_neg = neg_share(recent), neg_share(prior)
    recent_c = recent["compound"].mean() if len(recent) else None
    prior_c = prior["compound"].mean() if len(prior) else None

    direction = "insufficient data"
    if recent_neg is not None and prior_neg is not None:
        delta = prior_neg - recent_neg  # >0 means fewer negatives now
        if delta > 0.05:
            direction = "improving"
        elif delta < -0.05:
            direction = "deteriorating"
        else:
            direction = "stable"
    return {
        "direction": direction,
        "recent_negative_share": recent_neg,
        "prior_negative_share": prior_neg,
        "recent_mean_compound": recent_c,
        "prior_mean_compound": prior_c,
        "window_days": window,
    }


def volume_spikes(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """Days where discussion volume spiked (>= threshold x the average)."""
    pivot = sentiment_over_time(df)
    if pivot.empty:
        return pd.DataFrame()
    daily = pivot.sum(axis=1)
    mean = daily.mean()
    spikes = daily[daily >= max(threshold * mean, 2)]
    out = pd.DataFrame({"total": spikes, "mean": mean})
    return out


def top_emotional_posts(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Most emotionally intense posts (any emotion) — the loudest voices."""
    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    if not emo_cols:
        return pd.DataFrame()
    out = df.copy()
    out["emotion_intensity"] = out[emo_cols].sum(axis=1)
    cols = ["emotion_intensity", "final_label", "dominant_emotion", "date"]
    text_col = "text" if "text" in out.columns else out.columns[0]
    cols.append(text_col)
    return (out.sort_values("emotion_intensity", ascending=False)
            .head(n)[cols])


def opinion_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment split per source (social vs news vs reviews...)."""
    if "source" not in df.columns:
        return df["final_label"].value_counts().to_frame("count").T
    return (df.groupby("source")["final_label"].value_counts()
            .unstack(fill_value=0))


def emotion_profile(df: pd.DataFrame) -> pd.Series:
    """Aggregate emotion scores across the whole dataset."""
    cols = [c for c in df.columns if c.startswith("emo_")]
    if cols:
        return df[cols].sum().rename(index=lambda c: c[4:])
    lexicon = get_lexicon()
    totals = Counter()
    for t in df["text"].astype(str):
        totals.update(detect_emotions(t, lexicon))
    return pd.Series({e: totals[e] for e in EMOTIONS})


def keyword_drivers(df: pd.DataFrame, top_n: int = 12) -> dict:
    """Words most associated with positive vs negative opinion.

    Scores = (share in positive texts) - (share in negative texts),
    restricted to words carrying NRC emotion/sentiment associations.
    """
    lexicon = get_lexicon()
    pos_counter, neg_counter = Counter(), Counter()

    def emotional_words(text):
        out = set()
        for tok in clean_text(text).split():
            if tok in _SW:
                continue
            for form in (tok, _lemmatizer.lemmatize(tok)):
                if form in lexicon:
                    out.add(form)
                    break
        return out

    for _, row in df.iterrows():
        words = emotional_words(str(row["text"]))
        if row["final_label"] == "positive":
            pos_counter.update(words)
        elif row["final_label"] == "negative":
            neg_counter.update(words)

    def share(counter, total):
        t = sum(counter.values()) or 1
        return {w: c / t for w, c in counter.items()}

    pos_share = share(pos_counter, sum(pos_counter.values()))
    neg_share = share(neg_counter, sum(neg_counter.values()))
    all_words = set(pos_share) | set(neg_share)
    scored = sorted(all_words, key=lambda w: pos_share.get(w, 0) - neg_share.get(w, 0))
    return {
        "positive": [(w, round(pos_share.get(w, 0) - neg_share.get(w, 0), 3),
                      pos_counter[w] + neg_counter[w]) for w in reversed(scored[-top_n:])],
        "negative": [(w, round(pos_share.get(w, 0) - neg_share.get(w, 0), 3),
                      pos_counter[w] + neg_counter[w]) for w in scored[:top_n]],
    }


def summarize(df: pd.DataFrame) -> str:
    """Plain-language summary of public opinion."""
    n = len(df)
    counts = df["final_label"].value_counts()
    pct = lambda k: 100 * counts.get(k, 0) / n if n else 0
    mood = ("broadly positive" if pct("positive") > pct("negative") + 10 else
            "broadly negative" if pct("negative") > pct("positive") + 10 else
            "mixed / divided")
    emo = emotion_profile(df)
    top_emo = emo.idxmax() if emo.sum() > 0 else "none detected"
    lines = [
        f"Public opinion across {n} items is {mood}: "
        f"{pct('positive'):.0f}% positive, {pct('negative'):.0f}% negative, "
        f"{pct('neutral'):.0f}% neutral.",
        f"The strongest emotional signal is **{top_emo}**.",
    ]
    if "source" in df.columns:
        per_src = []
        for src, grp in df.groupby("source"):
            g = grp["final_label"].value_counts()
            lean = g.idxmax() if len(g) else "neutral"
            per_src.append(f"{src} leans {lean} ({100 * int(g.get(lean, 0)) / len(grp):.0f}%)")
        lines.append("By source: " + "; ".join(per_src) + ".")
    return " ".join(lines)


def topic_clusters(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """Group posts into topics (TF-IDF + KMeans) and profile each topic.

    Returns {"labels": list per row, "topics": DataFrame with top terms,
    size, dominant sentiment and dominant emotion per cluster}.
    """
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = df["text"].astype(str).tolist() if "text" in df.columns else None
    if not texts or len(texts) < n_clusters:
        return {"labels": [None] * len(df), "topics": pd.DataFrame()}

    vec = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
    X = vec.fit_transform([clean_text(t) for t in texts])
    if X.nnz == 0:
        return {"labels": [None] * len(df), "topics": pd.DataFrame()}

    n_clusters = min(n_clusters, len(texts))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    terms = np.array(vec.get_feature_names_out())

    rows = []
    for k in range(n_clusters):
        mask = labels == k
        sub = df[mask]
        centroid = km.cluster_centers_[k]
        top_terms = ", ".join(terms[centroid.argsort()[::-1][:6]])
        vc = sub["final_label"].value_counts()
        rows.append({
            "topic": k,
            "size": int(mask.sum()),
            "top_terms": top_terms,
            "dominant_sentiment": vc.idxmax() if len(vc) else "neutral",
            "pct_negative": 100 * int(vc.get("negative", 0)) / mask.sum(),
            "dominant_emotion": (sub["dominant_emotion"].mode().iat[0]
                                 if "dominant_emotion" in sub and len(sub) else "neutral"),
        })
    return {"labels": labels.tolist(), "topics": pd.DataFrame(rows)}


def anomaly_alerts(df: pd.DataFrame, z: float = 2.0) -> list:
    """Flag unusual shifts in sentiment or volume across dated data.

    Alerts fire when a day's negative share is > z std-devs above the mean,
    or when volume is a significant outlier.
    """
    pivot = sentiment_over_time(df)
    if pivot.empty:
        return []
    totals = pivot.sum(axis=1)
    neg_share = pivot.get("negative", 0) / totals
    alerts = []

    mu, sd = neg_share.mean(), neg_share.std()
    if sd > 0:
        for date, val in neg_share[neg_share > mu + z * sd].items():
            alerts.append({
                "date": date, "type": "negative-sentiment spike",
                "detail": f"negative share {val:.0%} (avg {mu:.0%}) "
                          f"with {int(totals[date])} posts",
            })

    vol_mu, vol_sd = totals.mean(), totals.std()
    if vol_sd > 0:
        for date, val in totals[totals > vol_mu + z * vol_sd].items():
            alerts.append({
                "date": date, "type": "volume spike",
                "detail": f"{int(val)} posts (avg {vol_mu:.1f})",
            })
    return sorted(alerts, key=lambda a: a["date"])


def compare_datasets(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     label_a: str = "A", label_b: str = "B") -> dict:
    """Compare opinion between two datasets (e.g. two time periods or topics)."""
    def stats(d):
        n = len(d)
        counts = d["final_label"].value_counts()
        emo = emotion_profile(d)
        return {
            "n": n,
            "pct": {k: 100 * counts.get(k, 0) / n if n else 0
                    for k in ("positive", "negative", "neutral")},
            "mean_compound": d["compound"].mean() if "compound" in d else None,
            "top_emotion": emo.idxmax() if emo.sum() > 0 else "none",
        }

    sa, sb = stats(df_a), stats(df_b)
    verdict = ("opinion improved in " + label_b
               if sb["pct"]["positive"] - sb["pct"]["negative"] >
               sa["pct"]["positive"] - sa["pct"]["negative"]
               else "opinion improved in " + label_a
               if sa["pct"]["positive"] - sa["pct"]["negative"] >
               sb["pct"]["positive"] - sb["pct"]["negative"]
               else "opinion is equivalent in both")
    table = pd.DataFrame({label_a: sa, label_b: sb}).T
    return {"table": table, "verdict": verdict, label_a: sa, label_b: sb}


def strategy_insights(df: pd.DataFrame, drivers: dict = None) -> dict:
    """Turn analysis results into actionable recommendations.

    Returns a dict with 'marketing', 'product', and 'social' lists of
    recommendation strings, each grounded in a metric from the report.
    """
    n = len(df)
    counts = df["final_label"].value_counts()
    pct = lambda k: 100 * counts.get(k, 0) / n if n else 0
    emo = emotion_profile(df).sort_values(ascending=False)
    drivers = drivers or keyword_drivers(df)

    marketing, product, social = [], [], []

    # --- Marketing: amplify what works, defend against what hurts ------
    pos_kw = [w for w, _, _ in drivers["positive"][:5]]
    if pos_kw:
        marketing.append(
            f"Lean into positive language: '{pos_kw[0]}' and {pos_kw[1] if len(pos_kw) > 1 else 'related terms'} "
            f"drive the strongest positive association — echo these in ad copy and campaigns.")
    joy = emo.get("joy", 0) / max(emo.sum(), 1)
    if joy < 0.12:
        marketing.append(
            f"Joy is only {100 * joy:.0f}% of the emotional mix — messaging is functional, not delightful. "
            "Run campaigns built around positive user stories/UGC to lift the joy signal.")
    if pct("positive") > 20:
        marketing.append(
            f"{pct('positive'):.0f}% of posts are positive — retarget/shout out these users and "
            "recruit them for testimonials or affiliate programs.")
    trust = emo.get("trust", 0)
    if trust == emo.max():
        marketing.append(
            f"Trust is the dominant emotion ({trust:.0f} pts) — brand credibility is an asset; "
            "position campaigns around reliability and social proof.")

    # --- Product: fix what drives negative opinion ---------------------
    neg_kw = [w for w, _, _ in drivers["negative"][:6]]
    if neg_kw:
        product.append(
            f"Top negative drivers are: {', '.join(neg_kw[:4])} — audit these areas first "
            "in the next product/roadmap review.")
    if emo.get("sadness", 0) + emo.get("disgust", 0) > emo.get("joy", 0) * 2:
        product.append(
            "Sadness+disgust far outweigh joy — expectations are being broken, not just unmet; "
            "review onboarding/quality-of-experience, not just features.")
    topics_res = topic_clusters(df)
    topics = topics_res.get("topics")
    if topics is not None and len(topics):
        worst = topics.sort_values("pct_negative", ascending=False).iloc[0]
        product.append(
            f"Most negative discussion cluster: '{worst['top_terms']}' ({worst['pct_negative']:.0f}% negative, "
            f"{int(worst['size'])} posts) — prioritise fixes/comms for this theme.")
        biggest = topics.sort_values("size", ascending=False).iloc[0]
        product.append(
            f"Largest conversation is '{biggest['top_terms']}' ({int(biggest['size'])} posts) — "
            "this is what the audience actually talks about; make sure the roadmap addresses it.")

    # --- Social: where to engage, when, and how ------------------------
    if "source" in df.columns:
        per_src = df.groupby("source")["final_label"].value_counts().unstack(fill_value=0)
        if len(per_src):
            neg_src = (per_src.get("negative", 0) / per_src.sum(axis=1)).idxmax()
            marketing.append(
                f"'{neg_src}' is the most negative channel — prioritise reputation management "
                "and community management responses there.")
            social.append(
                f"Source mix: {', '.join(f'{s} ({int(r.sum())} posts)' for s, r in per_src.iterrows())} "
                "— weight content efforts toward the largest channel, but watch the most negative one.")
    mom = momentum(df)
    if mom and mom.get("direction") in ("improving", "deteriorating"):
        social.append(
            f"Momentum is {mom['direction']} — {'double down on current content strategy' if mom['direction'] == 'improving' else 'pause scheduled promo content and address the complaint wave first'}.")
    spikes = volume_spikes(df)
    if len(spikes):
        social.append(
            f"Volume spikes on: {', '.join(str(d) for d in spikes.index)} — identify what triggered them "
            "and keep reactive content ready around similar events (launches, price changes).")
    alerts = anomaly_alerts(df)
    for a in [x for x in alerts if x["type"] == "negative-sentiment spike"][:2]:
        social.append(f"Alert: {a['date']} saw {a['detail']} — check replies/threads from that day "
                      "for a brewing issue worth addressing publicly.")
    if not social:
        social.append("No unusual patterns detected — maintain current posting cadence and monitoring.")

    return {"marketing": marketing, "product": product, "social": social}


def full_report(df: pd.DataFrame, text_col: str = "text") -> dict:
    df = analyze_frame(df, text_col)
    drivers = keyword_drivers(df)
    return {
        "summary": summarize(df),
        "over_time": sentiment_over_time(df),
        "over_time_rolling": sentiment_over_time_rolling(df),
        "weekly": weekly_sentiment(df),
        "momentum": momentum(df),
        "volume_spikes": volume_spikes(df),
        "top_emotional_posts": top_emotional_posts(df),
        "topics": topic_clusters(df),
        "alerts": anomaly_alerts(df),
        "insights": strategy_insights(df, drivers),
        "by_source": opinion_by_source(df),
        "emotion_profile": emotion_profile(df),
        "positive_keywords": drivers["positive"],
        "negative_keywords": drivers["negative"],
        "frame": df,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "collected_iphone_16.csv")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    rep = full_report(df)

    print("=== PUBLIC OPINION SUMMARY ===")
    print(rep["summary"])
    mom = rep["momentum"]
    if mom:
        rn = mom.get("recent_negative_share")
        pn = mom.get("prior_negative_share")
        extra = ""
        if rn is not None and pn is not None:
            extra = (f" (negative share {pn:.0%} -> {rn:.0%}, "
                     f"mean compound "
                     f"{mom.get('prior_mean_compound') or 0:+.2f} -> "
                     f"{mom.get('recent_mean_compound') or 0:+.2f})")
        print(f"Momentum: {mom['direction']}{extra}")
    print("\n=== SENTIMENT OVER TIME (rolling share) ===")
    print(rep["over_time_rolling"].to_string())
    print("\n=== WEEKLY SENTIMENT ===")
    if len(rep["weekly"]):
        print(rep["weekly"].to_string())
    print("\n=== VOLUME SPIKES ===")
    if len(rep["volume_spikes"]):
        print(rep["volume_spikes"].to_string())
    print("\n=== ANOMALY ALERTS ===")
    if rep["alerts"]:
        for a in rep["alerts"]:
            print(f"  {a['date']}  {a['type']}: {a['detail']}")
    else:
        print("  (none)")
    print("\n=== TOPIC CLUSTERS ===")
    if len(rep["topics"]["topics"]):
        print(rep["topics"]["topics"].to_string(index=False))
    print("\n=== MOST EMOTIONAL POSTS ===")
    if len(rep["top_emotional_posts"]):
        for _, row in rep["top_emotional_posts"].iterrows():
            print(f"  [{row['emotion_intensity']:.0f}] {row['final_label']:8} "
                  f"{str(row.get('text', ''))[:80]}")
    print("\n=== SENTIMENT OVER TIME (daily counts) ===")
    print(rep["over_time"].to_string())
    print("\n=== BY SOURCE ===")
    print(rep["by_source"].to_string())
    print("\n=== EMOTION PROFILE ===")
    print(rep["emotion_profile"].sort_values(ascending=False).to_string())
    print("\n=== KEYWORDS DRIVING POSITIVE OPINION ===")
    for w, s, c in rep["positive_keywords"]:
        print(f"  {w:15} +{s:.3f}  (n={c})")
    print("\n=== KEYWORDS DRIVING NEGATIVE OPINION ===")
    for w, s, c in rep["negative_keywords"]:
        print(f"  {w:15} {s:.3f}  (n={c})")
    ins = rep["insights"]
    for section, title in (("marketing", "MARKETING"), ("product", "PRODUCT DEVELOPMENT"),
                           ("social", "SOCIAL INSIGHTS")):
        print(f"\n=== RECOMMENDATIONS: {title} ===")
        for r in ins[section]:
            print(f"  - {r}")
