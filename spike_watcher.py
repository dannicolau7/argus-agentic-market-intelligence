"""
spike_watcher.py — Real-time price + volume spike detection.

Runs every 60 seconds during market hours. Fetches the last few 1-minute
intraday bars for every watchlist ticker in a single yfinance batch call
(zero Polygon API usage). When a ticker spikes ≥ 2% on ≥ 2.5× average
volume in the last bar, it:

  1. Sends a preliminary WhatsApp: "⚡ AWRE spiking +4.2% on 3.8× vol"
  2. Immediately triggers the full LangGraph pipeline with news_triggered=True
     (lowers confidence threshold to 55 so early valid signals aren't rejected)

This catches moves *before* news is indexed by Polygon (30–50 min lag),
reducing alert latency from 30–50 min to under 2 min.

30-minute cooldown per ticker prevents re-triggering on the same move.
"""

import asyncio
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf

from alerts import send_whatsapp

EST              = ZoneInfo("America/New_York")
SPIKE_INTERVAL   = 60       # seconds between checks
PRICE_SPIKE_PCT  = 0.02     # 2% move in last 1-min bar triggers investigation
VOL_SPIKE_RATIO  = 2.5      # volume must be ≥ 2.5× bar average
SPIKE_COOLDOWN_S = 30 * 60  # 30-minute cooldown per ticker

_spike_alerted: dict = {}   # {ticker: time.time()} of last spike trigger


# ── Market hours ───────────────────────────────────────────────────────────────

def _market_open() -> bool:
    now = datetime.now(tz=EST)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


# ── Cooldown helpers ───────────────────────────────────────────────────────────

def _spike_in_cooldown(ticker: str) -> bool:
    last = _spike_alerted.get(ticker)
    return last is not None and (time.time() - last) < SPIKE_COOLDOWN_S


def _mark_spike(ticker: str):
    _spike_alerted[ticker] = time.time()


# ── Spike detection ────────────────────────────────────────────────────────────

def _fetch_spikes(tickers: list) -> list:
    """
    Batch-download 1-minute bars for all tickers in one HTTP call.
    Returns list of {ticker, price_chg_pct, vol_ratio, direction, price} dicts.
    """
    if not tickers:
        return []
    try:
        df = yf.download(
            tickers,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return []

        results = []
        single  = len(tickers) == 1

        for ticker in tickers:
            try:
                t_df = df if single else df.get(ticker)
                if t_df is None or t_df.empty:
                    continue

                t_df = t_df.dropna(subset=["Close", "Volume"])
                if len(t_df) < 3:
                    continue

                avg_vol  = float(t_df["Volume"].mean())
                last_bar = t_df.iloc[-1]
                prev_bar = t_df.iloc[-2]

                prev_close = float(prev_bar["Close"])
                if prev_close <= 0 or avg_vol <= 0:
                    continue

                price_chg = (float(last_bar["Close"]) - prev_close) / prev_close
                vol_ratio = float(last_bar["Volume"]) / avg_vol

                if abs(price_chg) >= PRICE_SPIKE_PCT and vol_ratio >= VOL_SPIKE_RATIO:
                    results.append({
                        "ticker":        ticker,
                        "price_chg_pct": price_chg * 100,
                        "vol_ratio":     vol_ratio,
                        "direction":     "UP" if price_chg > 0 else "DOWN",
                        "price":         float(last_bar["Close"]),
                    })

            except Exception:
                continue

        return results

    except Exception as e:
        print(f"⚠️  [SpikeWatcher] Batch fetch error: {e}")
        return []


# ── Main loop ──────────────────────────────────────────────────────────────────

async def spike_watcher_loop(run_once_fn, paper: bool, get_tickers_fn):
    """
    Async loop started in main.py lifespan.

    Args:
        run_once_fn:    main.run_once coroutine — called with (ticker, news_triggered=True)
        paper:          paper trading flag
        get_tickers_fn: callable returning current watchlist (e.g. wl.load)
    """
    mode = "PAPER" if paper else "LIVE"
    print(f"⚡ [SpikeWatcher] Started — checking every {SPIKE_INTERVAL}s | {mode}")

    loop = asyncio.get_running_loop()

    while True:
        try:
            await asyncio.sleep(SPIKE_INTERVAL)

            if not _market_open():
                continue

            tickers = get_tickers_fn()
            if not tickers:
                continue

            spikes = await loop.run_in_executor(None, _fetch_spikes, tickers)

            for s in spikes:
                ticker    = s["ticker"]
                chg_pct   = s["price_chg_pct"]
                vol_ratio = s["vol_ratio"]
                direction = s["direction"]
                price     = s["price"]

                if _spike_in_cooldown(ticker):
                    continue

                _mark_spike(ticker)

                arrow = "📈" if direction == "UP" else "📉"
                sign  = "+" if chg_pct >= 0 else ""
                msg   = (
                    f"⚡ SPIKE DETECTED [{datetime.now(tz=EST).strftime('%H:%M')}]\n"
                    f"{arrow} {ticker}  {sign}{chg_pct:.1f}%  ${price:.2f}\n"
                    f"Volume: {vol_ratio:.1f}× average\n"
                    f"Analyzing now..."
                )
                print(f"\n⚡ [SpikeWatcher] {ticker} spike: {sign}{chg_pct:.1f}%  vol={vol_ratio:.1f}×")

                if not paper:
                    try:
                        send_whatsapp(msg)
                    except Exception as e:
                        print(f"⚠️  [SpikeWatcher] WhatsApp failed: {e}")
                else:
                    print(f"   📋 [PAPER] Would send:\n{msg}")

                # Trigger full pipeline immediately with news_triggered=True
                try:
                    await run_once_fn(ticker, news_triggered=True)
                except Exception as e:
                    print(f"❌ [SpikeWatcher] Pipeline error for {ticker}: {e}")

        except asyncio.CancelledError:
            print("⚡ [SpikeWatcher] Stopped.")
            break
        except Exception as e:
            print(f"❌ [SpikeWatcher] Loop error: {e}")
            await asyncio.sleep(30)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import watchlist_manager as wl

    tickers = wl.load() or ["AWRE", "BZAI", "AAL"]
    print(f"Checking {len(tickers)} tickers for spikes: {tickers}")
    spikes = _fetch_spikes(tickers)
    if spikes:
        for s in spikes:
            print(f"  ⚡ {s['ticker']}: {s['price_chg_pct']:+.1f}%  vol={s['vol_ratio']:.1f}×  ${s['price']:.2f}")
    else:
        print("  No spikes detected right now.")
