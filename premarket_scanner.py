"""
premarket_scanner.py — Pre-market gap scanner.

Scans tickers for pre-market movers (≥3% gap, ≥30k shares) using
yfinance 1-min bars with prepost=True — free, no extra API key.

Designed to run at 8:30 AM ET via the scheduler.

Usage:
    from premarket_scanner import scan_premarket_gaps, format_premarket_msg
    gaps = scan_premarket_gaps(["AAPL", "TSLA", "NVDA", ...])
    print(format_premarket_msg(gaps[:3]))

CLI:
    python3 premarket_scanner.py
"""

import yfinance as yf
from datetime import time as dtime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

_ET = ZoneInfo("America/New_York")

# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_GAP_PCT      = 3.0      # minimum % gap to qualify
MIN_PREMARKET_VOL = 30_000   # minimum pre-market share volume
MIN_PRICE        = 0.50
MAX_PRICE        = 100.0
MAX_WORKERS      = 10       # parallel yfinance fetches


def _fetch_one(ticker: str) -> dict | None:
    """
    Fetch pre-market data for a single ticker.
    Returns a result dict if it qualifies, None otherwise.
    """
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m", prepost=True)
        if hist is None or hist.empty:
            return None

        # Normalize index to ET timezone
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC").tz_convert(_ET)
        else:
            hist.index = hist.index.tz_convert(_ET)

        market_open = dtime(9, 30)

        # Pre-market bars: any bar that starts before 09:30 ET
        pre = hist[hist.index.time < market_open]
        if pre.empty:
            return None

        # Previous close from fast_info (most reliable source)
        try:
            prev_close = float(t.fast_info["previous_close"])
        except Exception:
            return None

        if prev_close <= 0:
            return None

        premarket_price = float(pre["Close"].iloc[-1])
        premarket_vol   = int(pre["Volume"].sum())

        if premarket_price <= 0:
            return None

        gap_pct = (premarket_price - prev_close) / prev_close * 100

        # Apply filters
        if gap_pct < MIN_GAP_PCT:
            return None
        if premarket_vol < MIN_PREMARKET_VOL:
            return None
        if not (MIN_PRICE <= premarket_price <= MAX_PRICE):
            return None

        # Average daily volume for context
        try:
            avg_vol = float(t.fast_info.get("three_month_average_volume") or 0)
        except Exception:
            avg_vol = 0.0

        return {
            "ticker":          ticker,
            "gap_pct":         round(gap_pct, 2),
            "premarket_price": round(premarket_price, 4),
            "prev_close":      round(prev_close, 4),
            "premarket_vol":   premarket_vol,
            "avg_vol":         int(avg_vol),
        }
    except Exception:
        return None


def scan_premarket_gaps(tickers: List[str]) -> List[Dict]:
    """
    Scan a list of tickers for pre-market gap movers.

    Fetches in parallel (up to MAX_WORKERS concurrent requests).
    Returns list sorted by gap_pct descending.

    Each entry:
        ticker, gap_pct, premarket_price, prev_close,
        premarket_vol, avg_vol
    """
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            hit = fut.result()
            if hit:
                results.append(hit)
    results.sort(key=lambda x: x["gap_pct"], reverse=True)
    return results


def format_premarket_msg(gaps: List[Dict]) -> str:
    """Format a WhatsApp-ready message for the top pre-market gap movers."""
    if not gaps:
        return "⚡ Pre-Market (8:30 AM): No qualifying gap movers today."

    lines = ["⚡ Pre-Market Gaps (8:30 AM ET)"]
    for g in gaps:
        vol = g["premarket_vol"]
        vol_str = f"{vol/1_000:.0f}k" if vol < 1_000_000 else f"{vol/1_000_000:.1f}M"
        lines.append(
            f"{g['ticker']:6s}  +{g['gap_pct']:.1f}%  "
            f"${g['premarket_price']:.2f}  vol {vol_str}"
        )
    lines.append("💡 Confirm with real volume at the open")
    return "\n".join(lines)


if __name__ == "__main__":
    # Default: scan the top 100 tickers from the universe + some well-known names
    default_tickers = [
        "AAPL", "TSLA", "NVDA", "AMD", "META", "GOOG", "AMZN", "MSFT",
        "NFLX", "SMCI", "PLTR", "HOOD", "SOFI", "MARA", "RIOT", "COIN",
        "GME", "AMC", "BBBY", "LCID", "RIVN", "NIO", "XPEV", "F", "GM",
    ]
    try:
        from market_scanner import _load_universe
        universe = _load_universe() or []
        tickers  = list(dict.fromkeys(default_tickers + universe[:100]))
    except Exception:
        tickers = default_tickers

    print(f"Scanning {len(tickers)} tickers for pre-market gaps...")
    gaps = scan_premarket_gaps(tickers)
    print(f"\nFound {len(gaps)} qualifying gap(s):\n")
    print(format_premarket_msg(gaps[:5]))
    if gaps:
        print("\nFull results:")
        for g in gaps:
            vol_str = (f"{g['premarket_vol']/1_000:.0f}k"
                       if g["premarket_vol"] < 1_000_000
                       else f"{g['premarket_vol']/1_000_000:.1f}M")
            print(f"  {g['ticker']:6s}  gap {g['gap_pct']:+.1f}%  "
                  f"${g['premarket_price']:.2f}  premarket vol {vol_str}")
