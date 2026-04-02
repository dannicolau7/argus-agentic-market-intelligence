"""
news_agent.py — LangGraph node: fetches news from polygon_feed, scores sentiment via Claude.
Returns: news_sentiment (BULLISH/BEARISH/NEUTRAL), sentiment_score (0-100), news_summary.
"""

import json
import anthropic
from config import ANTHROPIC_API_KEY

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def news_node(state: dict) -> dict:
    raw_news = state.get("raw_news", [])
    ticker   = state["ticker"]
    print(f"📰 [NewsAgent] Analyzing news for {ticker}  ({len(raw_news)} articles)...")

    if not raw_news:
        print(f"⚠️  [NewsAgent] No recent news found")
        return {
            **state,
            "news_sentiment":  "NEUTRAL",
            "sentiment_score": 50,
            "news_summary":    "No recent news found.",
        }

    try:
        news_texts = []
        for item in raw_news[:8]:
            title = item.get("title", "")
            desc  = (item.get("description") or "")[:250]
            pub   = (item.get("published_utc") or "")[:10]
            news_texts.append(f"[{pub}] {title}: {desc}")

        news_block = "\n".join(news_texts)

        prompt = (
            f"Analyze the following recent news articles for {ticker} stock.\n\n"
            f"Return ONLY valid JSON (no markdown fences):\n"
            f'{{"sentiment": "BULLISH" | "BEARISH" | "NEUTRAL", '
            f'"score": <integer 0-100 where 0=extremely bearish 50=neutral 100=extremely bullish>, '
            f'"summary": "<one concise sentence summarising the key themes>"}}\n\n'
            f"Articles:\n{news_block}"
        )

        client   = _get_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text  = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data      = json.loads(text)
        sentiment = str(data.get("sentiment", "NEUTRAL")).upper()
        score     = max(0, min(100, int(data.get("score", 50))))
        summary   = str(data.get("summary", ""))

        emoji = "📈" if sentiment == "BULLISH" else "📉" if sentiment == "BEARISH" else "➡️"
        print(f"✅ [NewsAgent] {emoji} {sentiment}  score={score}/100  {summary[:80]}")

        return {
            **state,
            "news_sentiment":  sentiment,
            "sentiment_score": score,
            "news_summary":    summary,
        }

    except json.JSONDecodeError as e:
        print(f"⚠️  [NewsAgent] JSON parse error: {e}")
    except Exception as e:
        print(f"❌ [NewsAgent] Error: {e}")

    return {
        **state,
        "news_sentiment":  "NEUTRAL",
        "sentiment_score": 50,
        "news_summary":    "Sentiment analysis unavailable.",
    }
