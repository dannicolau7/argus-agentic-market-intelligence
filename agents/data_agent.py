"""
data_agent.py — LangGraph node: fetches live price, OHLCV bars, snapshot from Polygon.
"""

from polygon_feed import get_daily_bars, get_snapshot, get_news


def data_node(state: dict) -> dict:
    ticker = state["ticker"]
    print(f"📡 [DataAgent] Fetching live data for {ticker}...")

    try:
        bars = get_daily_bars(ticker, days=90)
        print(f"   └─ Bars:     {len(bars)} daily candles")
    except Exception as e:
        print(f"❌ [DataAgent] Bars fetch failed: {e}")
        bars = []

    try:
        snapshot = get_snapshot(ticker)
    except Exception as e:
        print(f"❌ [DataAgent] Snapshot fetch failed: {e}")
        snapshot = {}

    try:
        news = get_news(ticker, limit=10)
        print(f"   └─ News:     {len(news)} articles")
    except Exception as e:
        print(f"❌ [DataAgent] News fetch failed: {e}")
        news = []

    # Current price: prefer last trade, fall back to day close, then last bar
    last_trade    = snapshot.get("lastTrade", {})
    current_price = float(last_trade.get("p") or 0)
    if not current_price:
        current_price = float(snapshot.get("day", {}).get("c") or 0)
    if not current_price and bars:
        current_price = float(bars[-1].get("c", 0))

    # Previous close
    prev_close = float(snapshot.get("prevDay", {}).get("c") or 0)
    if not prev_close and len(bars) >= 2:
        prev_close = float(bars[-2].get("c", 0))

    # Volume: today's accumulated, then fallback to last bar
    volume = float(snapshot.get("day", {}).get("v") or 0)
    if not volume and bars:
        volume = float(bars[-1].get("v", 0))

    # 20-bar average volume
    avg_volume = 0.0
    if len(bars) >= 20:
        avg_volume = sum(float(b.get("v", 0)) for b in bars[-20:]) / 20.0

    print(
        f"✅ [DataAgent] ${current_price:.4f}  "
        f"Δ{((current_price - prev_close) / prev_close * 100):+.2f}%  "
        f"Vol: {volume:,.0f}  AvgVol: {avg_volume:,.0f}"
    )

    return {
        **state,
        "bars":          bars,
        "current_price": current_price,
        "prev_close":    prev_close,
        "volume":        volume,
        "avg_volume":    avg_volume,
        "snapshot":      snapshot,
        "raw_news":      news,
    }
