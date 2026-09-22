"""Streamlit dashboard for sentiment analysis.

Run:  .\\.venv\\Scripts\\python.exe -m streamlit run app.py
"""
import os
import pickle

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

from src.preprocess import tokenize_and_lemmatize
from src.vader_sentiment import vader_label, vader_scores
from src.classify import classify as combined_classify
from src.emotion_detection import EMOTIONS, detect_emotions, dominant_emotion, get_lexicon
from src.collect_data import fetch_mastodon, fetch_hackernews, fetch_news, load_amazon_csv
from src.trends import full_report, compare_datasets

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="🎬", layout="wide")


@st.cache_resource
def load_ml_model():
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def ml_predict(texts):
    model, vectorizer = load_ml_model()
    cleaned = [tokenize_and_lemmatize(t) for t in texts]
    tfidf = vectorizer.transform(cleaned)
    preds = model.predict(tfidf)
    probs = model.predict_proba(tfidf).max(axis=1)
    # map 'pos'/'neg' to full words
    return [f"{'positive' if p == 'pos' else 'negative'}" for p in preds], probs


st.title("🎬 Sentiment Analysis Dashboard")
st.caption("VADER (rule-based) + TF-IDF / Logistic Regression (trained on NLTK movie reviews, ~86.5% accuracy)")

tab_single, tab_batch, tab_emotions, tab_sources, tab_trends = st.tabs(
    ["🔍 Single text", "📊 Batch / CSV", "🎭 Emotions", "🌐 Live sources", "📈 Trends"])

with tab_single:
    text = st.text_area("Enter text to analyze:", height=150,
                        placeholder="e.g. The movie was fantastic — I loved it!")
    if st.button("Analyze", type="primary") and text.strip():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("VADER")
            scores = vader_scores(text)
            label = vader_label(text)
            emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}[label]
            st.metric("Sentiment", f"{label} {emoji}")
            st.bar_chart(pd.Series(
                {k: scores[k] for k in ("pos", "neu", "neg")}, name="score"))
            st.write(f"Compound score: **{scores['compound']:.3f}**")

        with col2:
            st.subheader("ML Model")
            try:
                label_ml, prob = ml_predict([text])
                emoji = {"positive": "😊", "negative": "😠"}[label_ml[0]]
                st.metric("Sentiment", f"{label_ml[0]} {emoji}")
                st.metric("Confidence", f"{prob[0] * 100:.1f}%")
            except FileNotFoundError:
                st.warning("Model not trained yet. Run `src/train_model.py` first.")

        # Word cloud
        cleaned = tokenize_and_lemmatize(text)
        if cleaned:
            wc = WordCloud(width=800, height=300, background_color="white").generate(cleaned)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

with tab_batch:
    uploaded = st.file_uploader("Upload a CSV with a 'text' column", type="csv")
    df = None
    if uploaded:
        df = pd.read_csv(uploaded)
    elif st.checkbox("Use sample data (data/sample_reviews.csv)"):
        df = pd.read_csv(os.path.join(ROOT, "data", "sample_reviews.csv"))

    if df is not None:
        text_col = st.selectbox("Text column:", [c for c in df.columns if df[c].dtype == object])
        if st.button("Analyze batch", type="primary") and text_col:
            texts = df[text_col].astype(str).tolist()
            # combined VADER + ML classification -> final pos/neg/neutral label
            combined = [combined_classify(t) for t in texts]
            df["final_label"] = [r["final_label"] for r in combined]
            df["vader_sentiment"] = [r["vader_label"] for r in combined]
            df["vader_compound"] = [r["vader_compound"] for r in combined]
            if combined[0]["ml_label"] is not None:
                df["ml_sentiment"] = [r["ml_label"] for r in combined]
                df["ml_confidence"] = [r["ml_confidence"] for r in combined]
            else:
                st.warning("Model not trained yet — final label uses VADER only. Run src/train_model.py")
            lexicon = get_lexicon()
            emo_rows = [detect_emotions(t, lexicon) for t in texts]
            for e in EMOTIONS:
                df[f"emo_{e}"] = [round(r[e], 1) for r in emo_rows]
            df["dominant_emotion"] = [
                max(EMOTIONS, key=lambda e: r[e]) if max(r[e] for e in EMOTIONS) > 0 else "neutral"
                for r in emo_rows]
            try:
                labels_ml, probs = ml_predict(texts)
                if "ml_sentiment" not in df:
                    df["ml_sentiment"] = labels_ml
                    df["ml_confidence"] = (probs * 100).round(1)
            except FileNotFoundError:
                pass

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Final distribution (pos/neg/neutral)")
                st.bar_chart(df["final_label"].value_counts())
            with col2:
                if "ml_sentiment" in df:
                    st.subheader("ML distribution")
                    st.bar_chart(df["ml_sentiment"].value_counts())

            st.subheader("Emotion totals (NRC EmoLex)")
            emo_totals = df[[f"emo_{e}" for e in EMOTIONS]].sum()
            st.bar_chart(emo_totals.rename(index=lambda c: c[4:]))

            if "vader_compound" in df:
                st.subheader("Compound score over rows")
                st.line_chart(df["vader_compound"])

            st.dataframe(df, use_container_width=True)
            st.download_button("Download results CSV", df.to_csv(index=False).encode(),
                               "sentiment_results.csv", "text/csv")

with tab_emotions:
    st.subheader("Emotion detection — NRC Emotion Lexicon (EmoLex)")
    st.caption("8 basic emotions (Plutchik) + positive/negative, via the NRC Word-Emotion "
               "Association Lexicon (~14k words), with lemmatization, negation handling "
               "and intensifier weighting.")

    emo_text = st.text_area("Enter text to detect emotions:", height=120,
                            value="I am so happy and grateful, but the ending left me devastated.",
                            key="emo_text")
    emo_df = None
    if st.button("Detect emotions", type="primary") and emo_text.strip():
        try:
            lexicon = get_lexicon()
        except FileNotFoundError:
            st.error("Lexicon file missing: data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
            st.stop()
        dom, sc, full = dominant_emotion(emo_text, lexicon)
        st.metric("Dominant emotion", f"{dom.capitalize()} {'' if dom == 'neutral' else '🎭'}",
                  help=f"Score: {sc}")

        emo_scores = {e: full[e] for e in EMOTIONS}
        emo_df = pd.DataFrame({"emotion": list(emo_scores), "score": list(emo_scores.values())})

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(emo_df.set_index("emotion"), color="#6a5acd")
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            nonz = emo_df[emo_df.score > 0]
            if len(nonz):
                cmap = plt.get_cmap("plasma")
                colors = cmap(np.linspace(0.1, 0.9, len(nonz)))
                ax.pie(nonz.score, labels=nonz.emotion, colors=colors,
                       autopct=lambda p: f"{p:.0f}%", startangle=90)
            else:
                ax.text(0.5, 0.5, "No emotions detected", ha="center")
            ax.axis("equal")
            st.pyplot(fig)

        st.caption("Positive / negative sentiment from EmoLex: "
                   f"positive **{full['positive']:.0f}** · negative **{full['negative']:.0f}**")

with tab_sources:
    st.subheader("Analyze text from live public sources")
    st.caption("Mastodon (social media) · Hacker News (tech discussion) · Google News RSS "
               "(news sites) · Amazon reviews via CSV import. "
               "Or use the CLI: `python src/collect_data.py '<topic>'` then "
               "`python src/classify.py <csv> --column text`.")

    src_query = st.text_input("Topic / search query:", value="iphone 16", key="src_query")
    c1, c2, c3 = st.columns(3)
    with c1:
        use_mastodon = st.checkbox("Mastodon (social)", True)
        use_hn = st.checkbox("Hacker News", True)
    with c2:
        use_news = st.checkbox("News (Google RSS)", True)
        use_amazon = st.checkbox("Amazon CSV")
    with c3:
        n_items = st.slider("Items per source", 5, 40, 15)
        amazon_file = st.file_uploader("Amazon reviews CSV", type="csv",
                                       disabled=not use_amazon)

    if st.button("Collect & analyze", type="primary") and src_query.strip():
        posts = []
        try:
            if use_mastodon:
                posts += fetch_mastodon(src_query, n_items)
            if use_hn:
                posts += fetch_hackernews(src_query, n_items)
            if use_news:
                posts += fetch_news(src_query, n_items)
            if use_amazon:
                if amazon_file:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        tmp.write(amazon_file.getvalue())
                    posts += load_amazon_csv(tmp.name, n_items)
                    os.unlink(tmp.name)
                else:
                    st.warning("Amazon selected but no CSV uploaded.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Collection error: {e}")

        if posts:
            lexicon = get_lexicon()
            for p in posts:
                res = combined_classify(p["text"])
                p["final_label"] = res["final_label"]
                p["compound"] = res["vader_compound"]
                emo = detect_emotions(p["text"], lexicon)
                top = max(EMOTIONS, key=lambda e: emo[e])
                p["dominant_emotion"] = top if emo[top] > 0 else "neutral"
            src_df = pd.DataFrame(posts)

            st.metric("Items collected", len(src_df))
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sentiment by source")
                st.bar_chart(src_df.groupby("source")["final_label"]
                             .value_counts().unstack(fill_value=0))
            with col2:
                st.subheader("Overall sentiment")
                st.bar_chart(src_df["final_label"].value_counts())

            st.subheader("Dominant emotion by source")
            st.bar_chart(src_df.groupby("source")["dominant_emotion"]
                         .value_counts().unstack(fill_value=0))

            st.dataframe(src_df[["source", "final_label", "dominant_emotion",
                                 "compound", "text", "url"]], use_container_width=True)
            st.download_button("Download collected + analyzed CSV",
                               src_df.to_csv(index=False).encode(),
                               "source_analysis.csv", "text/csv")

with tab_trends:
    st.subheader("Public opinion & trends")
    st.caption("Sentiment patterns over time, by source, and the keywords driving "
               "positive vs negative opinion — on any collected CSV "
               "(default: the most recent file in data/).")

    csv_files = sorted(
        [f for f in os.listdir(os.path.join(ROOT, "data")) if f.endswith(".csv")])
    chosen = st.selectbox("Dataset:", csv_files or ["(no CSVs in data/)"])
    if chosen != "(no CSVs in data/)" and st.button("Analyze trends", type="primary"):
        tdf = pd.read_csv(os.path.join(ROOT, "data", chosen),
                          encoding="utf-8", encoding_errors="replace")
        text_col = st.selectbox("Text column:",
                                [c for c in tdf.columns if tdf[c].dtype == object],
                                key="trend_text_col")
        rep = full_report(tdf, text_col)

        st.info(rep["summary"], icon="📌")

        mom = rep["momentum"]
        if mom and mom.get("direction"):
            icon = {"improving": "📈", "deteriorating": "📉"}.get(mom["direction"], "➡️")
            rn, pn = mom.get("recent_negative_share"), mom.get("prior_negative_share")
            detail = (f"Negative share {pn:.0%} → {rn:.0%} over the last "
                      f"{mom['window_days']} days vs the previous "
                      f"{mom['window_days']} days.") if rn is not None and pn is not None else \
                     "Not enough dated data for a before/after comparison."
            st.metric(f"{icon} Opinion momentum", mom["direction"].capitalize(),
                      detail if rn is not None and pn is not None else None)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment over time (smoothed share)")
            if len(rep["over_time_rolling"]):
                st.line_chart(rep["over_time_rolling"], color=["#2e8b57", "#b22222"])
                st.caption("Rolling average of the positive (green) vs negative (red) "
                           "share of daily posts — smooths out volume noise.")
            else:
                st.caption("No timestamp data in this dataset.")
        with c2:
            st.subheader("Opinion by source")
            st.bar_chart(rep["by_source"])

        col_w, col_s = st.columns(2)
        with col_w:
            if len(rep["weekly"]):
                st.subheader("Weekly sentiment volume")
                st.bar_chart(rep["weekly"])
        with col_s:
            if len(rep["volume_spikes"]):
                st.subheader("Volume spikes (>= 2x average daily volume)")
                st.bar_chart(rep["volume_spikes"]["total"], color="#ff8c00")
                st.caption("Days with unusually high discussion volume — often "
                           "event-driven (launches, controversies, news coverage).")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Emotion profile")
            st.bar_chart(rep["emotion_profile"])
        with col_b:
            st.subheader("Keyword drivers of opinion")
            pos_w = pd.DataFrame(rep["positive_keywords"],
                                 columns=["word", "lean", "n"]).set_index("word")["lean"]
            neg_w = pd.DataFrame(rep["negative_keywords"],
                                 columns=["word", "lean", "n"]).set_index("word")["lean"]
            st.caption("Pushing opinion **positive**:")
            st.bar_chart(pos_w, color="#2e8b57")
            st.caption("Pushing opinion **negative**:")
            st.bar_chart(neg_w, color="#b22222")

        if len(rep["top_emotional_posts"]):
            st.subheader("Most emotionally intense posts")
            st.dataframe(rep["top_emotional_posts"], use_container_width=True)

        if rep["alerts"]:
            st.subheader("⚠️ Anomaly alerts")
            st.caption("Days where negative sentiment or discussion volume were "
                       "statistically unusual (z-score >= 2).")
            st.dataframe(pd.DataFrame(rep["alerts"]), use_container_width=True)
        else:
            st.success("No anomalies detected — sentiment and volume are within "
                       "normal ranges.")

        topics = rep["topics"].get("topics")
        if topics is not None and len(topics):
            st.subheader("What people are talking about (topic clusters)")
            st.caption("Posts grouped by TF-IDF + KMeans; each topic profiled by "
                       "top terms, size and dominant sentiment.")
            st.dataframe(topics, use_container_width=True)

        ins = rep["insights"]
        st.divider()
        st.subheader("💡 Recommendations")
        st.caption("Auto-generated from the metrics above — use to inform marketing, "
                   "product development and social strategy.")
        col_m, col_p, col_s = st.columns(3)
        with col_m:
            st.markdown("**📣 Marketing**")
            for r in ins["marketing"]:
                st.markdown(f"- {r}")
        with col_p:
            st.markdown("**🛠 Product development**")
            for r in ins["product"]:
                st.markdown(f"- {r}")
        with col_s:
            st.markdown("**💬 Social insights**")
            for r in ins["social"]:
                st.markdown(f"- {r}")

        st.dataframe(rep["frame"][["final_label", "dominant_emotion", "compound",
                                   text_col]], use_container_width=True)
        st.download_button("Download analyzed dataset",
                           rep["frame"].to_csv(index=False).encode(),
                           "trend_analysis.csv", "text/csv")

    # --- Dataset comparison ---------------------------------------------
    if len(csv_files) >= 2:
        st.divider()
        st.subheader("⚖️ Compare two datasets")
        st.caption("e.g. the same topic collected at two points in time, or two "
                   "different products/topics.")
        cA, cB = st.columns(2)
        with cA:
            file_a = st.selectbox("Dataset A:", csv_files, key="cmp_a")
        with cB:
            file_b = st.selectbox("Dataset B:", csv_files, index=1, key="cmp_b")
        if st.button("Compare", type="primary") and file_a != file_b:
            df_a_raw = pd.read_csv(os.path.join(ROOT, "data", file_a),
                                   encoding="utf-8", encoding_errors="replace")
            df_b_raw = pd.read_csv(os.path.join(ROOT, "data", file_b),
                                   encoding="utf-8", encoding_errors="replace")
            col_a = "text" if "text" in df_a_raw.columns else df_a_raw.columns[0]
            col_b = "text" if "text" in df_b_raw.columns else df_b_raw.columns[0]
            df_a = full_report(df_a_raw, col_a)["frame"]
            df_b = full_report(df_b_raw, col_b)["frame"]
            cmp_res = compare_datasets(df_a, df_b, file_a, file_b)
            st.info(f"**Verdict:** {cmp_res['verdict']}", icon="⚖️")
            pct_table = pd.DataFrame({file_a: cmp_res[file_a]["pct"],
                                      file_b: cmp_res[file_b]["pct"]}).T
            st.bar_chart(pct_table)
            st.dataframe(cmp_res["table"], use_container_width=True)
