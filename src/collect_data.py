"""Collect text data from public sources for sentiment/emotion analysis.

Sources:
  - mastodon    : social media (public hashtag timeline, no API key needed;
                  note: reddit.com blocks unauthenticated JSON from most IPs)
  - hackernews  : tech discussion (Algolia public search API)
  - news        : news sites via Google News RSS (RSS per topic/query)
  - amazon      : Amazon reviews via CSV import (Amazon blocks scraping;
                  download a reviews dataset e.g. from Kaggle's "Amazon Reviews"
                  or the McAuley UCSD Amazon review dataset and pass the path)

Each fetcher returns a list of dicts with at least {"source", "text", ...}.
Collected data is saved to data/<name>.csv with a unified schema:
  source, text, author/title, url, created_utc
"""
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

UA = {"User-Agent": "Mozilla/5.0 (sentiment-analysis-project; educational use)"}


def _get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ------------------------------------------------------------- mastodon ----
def _strip_html(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def fetch_mastodon(query: str, limit: int = 25) -> list:
    """Fetch public posts for a hashtag from mastodon.social (no key needed)."""
    tag = query.replace(" ", "").replace("#", "").split()[0]
    url = f"https://mastodon.social/api/v1/timelines/tag/{urllib.parse.quote(tag)}?limit={min(limit, 40)}"
    data = json.loads(_get(url))
    posts = []
    for p in data:
        text = _strip_html(p.get("content", ""))
        if text:
            posts.append({
                "source": "mastodon",
                "text": text[:2000],
                "title": f"#{tag} post",
                "author": (p.get("account") or {}).get("acct", ""),
                "url": p.get("url", ""),
                "created_utc": (p.get("created_at") or "")[:19],
            })
    return posts


# ---------------------------------------------------------------- reddit ----
def fetch_reddit(query: str, limit: int = 25, sort: str = "relevance") -> list:
    """Search public Reddit posts (social media text)."""
    url = (f"https://www.reddit.com/search.json?q={urllib.parse.quote(query)}"
           f"&limit={limit}&sort={sort}")
    data = json.loads(_get(url))
    posts = []
    for child in data.get("data", {}).get("children", []):
        p = child["data"]
        text = f"{p.get('title', '')} {p.get('selftext', '')}".strip()
        if text:
            posts.append({
                "source": "reddit",
                "text": text[:2000],
                "title": p.get("title", ""),
                "author": p.get("author", ""),
                "url": f"https://www.reddit.com{p.get('permalink', '')}",
                "created_utc": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.gmtime(p.get("created_utc", 0))),
            })
    return posts


# ----------------------------------------------------------- hacker news ----
def fetch_hackernews(query: str, limit: int = 25) -> list:
    """Search Hacker News stories + comment text (Algolia public API)."""
    url = (f"https://hn.algolia.com/api/v1/search?query={urllib.parse.quote(query)}"
           f"&hitsPerPage={limit}")
    data = json.loads(_get(url))
    posts = []
    for hit in data.get("hits", []):
        text = hit.get("title") or hit.get("story_title") or ""
        comment = hit.get("comment_text") or ""
        body = f"{text} {comment}".strip()
        if body:
            posts.append({
                "source": "hackernews",
                "text": body[:2000],
                "title": hit.get("title") or hit.get("story_title") or "",
                "author": hit.get("author", ""),
                "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                "created_utc": (hit.get("created_at") or "")[:19],
            })
    return posts


# ------------------------------------------------------------- news rss ----
def fetch_news(query: str, limit: int = 25) -> list:
    """Fetch news headlines/descriptions from Google News RSS."""
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    root = ET.fromstring(_get(url))
    posts = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        text = f"{title}. {desc}"[:2000]
        if text:
            posts.append({
                "source": "news",
                "text": text,
                "title": title,
                "author": "",
                "url": item.findtext("link") or "",
                "created_utc": (item.findtext("pubDate") or "")[:22],
            })
        if len(posts) >= limit:
            break
    return posts


# ------------------------------------------------------------- amazon -------
def load_amazon_csv(path: str, limit: int = 100, text_col: str = None) -> list:
    """Load Amazon reviews from a CSV (e.g. Kaggle/UCSD Amazon review dataset).

    Auto-detects common text column names: reviewText, Text, review_body,
    review/text, Review, body, content, summary.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if text_col is None:
        candidates = ["reviewText", "Text", "review_body", "review/text",
                      "Review", "review", "body", "content", "summary"]
        text_col = next((c for c in candidates if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"No review text column found. Columns: {list(df.columns)}")
    posts = []
    for _, row in df.head(limit).iterrows():
        text = str(row[text_col]).strip()
        if text and text.lower() != "nan":
            posts.append({
                "source": "amazon",
                "text": text[:2000],
                "title": str(row.get("summary", row.get("Summary", "")))[:200],
                "author": "",
                "url": "",
                "created_utc": str(row.get("Time", row.get("time", ""))),
            })
    return posts


# ------------------------------------------------------------- driver -------
def save_csv(posts: list, name: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "text", "title",
                                               "author", "url", "created_utc"])
        writer.writeheader()
        writer.writerows(posts)
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Collect text from public sources")
    ap.add_argument("query", help="Search topic, e.g. 'iphone 16' or 'climate change'")
    ap.add_argument("--sources", default="mastodon,hackernews,news",
                    help="Comma list: mastodon,hackernews,news,amazon,reddit")
    ap.add_argument("--limit", type=int, default=25, help="Items per source")
    ap.add_argument("--amazon-csv", default=None,
                    help="Path to an Amazon reviews CSV (source 'amazon')")
    args = ap.parse_args()

    wanted = [s.strip().lower() for s in args.sources.split(",")]
    all_posts = []
    for src in wanted:
        try:
            if src == "mastodon":
                posts = fetch_mastodon(args.query, args.limit)
            elif src == "hackernews":
                posts = fetch_hackernews(args.query, args.limit)
            elif src == "news":
                posts = fetch_news(args.query, args.limit)
            elif src == "amazon":
                if not args.amazon_csv:
                    print("[skip] amazon: pass --amazon-csv path/to/reviews.csv")
                    continue
                posts = load_amazon_csv(args.amazon_csv, args.limit)
            else:
                print(f"[skip] unknown source {src}")
                continue
            print(f"[ok] {src}: {len(posts)} items")
            all_posts.extend(posts)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {src}: {e}")

    if all_posts:
        path = save_csv(all_posts, f"collected_{args.query.replace(' ', '_')[:30]}")
        print(f"Saved {len(all_posts)} items -> {path}")
        print(f"Next: .\\.venv\\Scripts\\python.exe src\\classify.py \"{path}\" --column text")


if __name__ == "__main__":
    main()
