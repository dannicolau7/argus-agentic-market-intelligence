"""
news_agent.py — LangGraph node: fetches Reddit posts from r/wallstreetbets,
r/stocks, and r/investing using the free public JSON API (no key required).
Scores sentiment 0-100 by counting positive vs negative words in post titles.
"""

import re
import requests

# ── Sentiment word lists ───────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "buy", "bull", "bullish", "moon", "calls", "surge", "up", "gain",
    "profit", "long", "rocket", "rally", "breakout", "pump", "squeeze",
    "rip", "green", "wins", "winner", "growth",
}

NEGATIVE_WORDS = {
    "sell", "bear", "bearish", "puts", "crash", "down", "loss", "short",
    "dump", "drop", "falling", "tank", "red", "baghold", "bagholder",
    "bankrupt", "fraud", "collapse", "bust", "rekt",
}

SUBREDDITS = ["wallstreetbets", "stocks", "investing"]

HEADERS = {"User-Agent": "stock-ai-agent/1.0 (research tool)"}


# ── Reddit fetcher ─────────────────────────────────────────────────────────────

def _fetch_reddit(ticker: str, subreddit: str, limit: int = 10) -> list[str]:
    """
    Returns a list of post titles mentioning `ticker` from the given subreddit.
    Uses the free public Reddit JSON search endpoint — no API key required.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {"q": ticker, "sort": "new", "limit": limit, "restrict_sr": "on"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        children = r.json().get("data", {}).get("children", [])
        return [c["data"]["title"] for c in children if c.get("data", {}).get("title")]
    except Exception as e:
        print(f"⚠️  [NewsAgent] Reddit r/{subreddit} fetch failed: {e}")
        return []


# ── Word-count scorer ──────────────────────────────────────────────────────────

def _score_titles(titles: list[str]) -> tuple[str, int, int, int]:
    """
    Counts positive / negative word hits across all titles.
    Returns (sentiment, score_0_100, pos_count, neg_count).
    """
    pos = 0
    neg = 0
    for title in titles:
        words = set(re.findall(r"[a-z]+", title.lower()))
        pos += len(words & POSITIVE_WORDS)
        neg += len(words & NEGATIVE_WORDS)

    total = pos + neg
    if total == 0:
        return "NEUTRAL", 50, pos, neg

    score = round((pos / total) * 100)
    if score >= 60:
        sentiment = "BULLISH"
    elif score <= 40:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    return sentiment, score, pos, neg


# ── LangGraph node ─────────────────────────────────────────────────────────────

def news_node(state: dict) -> dict:
    ticker = state["ticker"]
    print(f"📰 [NewsAgent] Fetching Reddit sentiment for {ticker}  "
          f"({', '.join('r/' + s for s in SUBREDDITS)})...")

    all_titles: list[str] = []

    for sub in SUBREDDITS:
        titles = _fetch_reddit(ticker, sub, limit=10)
        print(f"   └─ r/{sub}: {len(titles)} posts")
        all_titles.extend(titles)

    if not all_titles:
        print(f"⚠️  [NewsAgent] No Reddit posts found for {ticker}")
        return {
            **state,
            "news_sentiment":  "NEUTRAL",
            "sentiment_score": 50,
            "news_summary":    f"No Reddit posts found for {ticker}.",
        }

    sentiment, score, pos, neg = _score_titles(all_titles)

    # Build a short summary from the top 3 most relevant titles
    summary_titles = all_titles[:3]
    summary = " | ".join(t[:80] for t in summary_titles)

    emoji = "📈" if sentiment == "BULLISH" else "📉" if sentiment == "BEARISH" else "➡️"
    print(
        f"✅ [NewsAgent] {emoji} {sentiment}  score={score}/100  "
        f"pos={pos} neg={neg}  posts={len(all_titles)}"
    )

    return {
        **state,
        "news_sentiment":  sentiment,
        "sentiment_score": score,
        "news_summary":    f"Reddit ({len(all_titles)} posts) +{pos}/-{neg}: {summary}",
    }
