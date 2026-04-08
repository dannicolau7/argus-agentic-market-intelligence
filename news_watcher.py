"""
news_watcher.py — 24/7 news-driven stock discovery.

Polls Polygon /v2/reference/news (ALL stocks, no ticker filter) every 5 min.
When a new article is detected for any stock:
  1. Quick gate: price $0.50–$50, min volume (yfinance, 0 Polygon calls)
  2. Cooldown: skip if already alerted for this ticker within 4 hours
  3. Full pipeline: fetch_data → news → tech → decide → alert
  4. WhatsApp sent if signal BUY/SELL and confidence >= 68

Runs 24/7 alongside monitoring_loop and scheduler_loop in main.py.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
import yfinance as yf

from config import POLYGON_API_KEY
from graph import GRAPH, make_initial_state
from alerts import send_whatsapp

BASE_URL             = "https://api.polygon.io"
POLL_INTERVAL_S      = 300          # 5 minutes between polls
ALERT_COOLDOWN_S     = 4 * 3600    # 4 hours before same ticker can alert again
CONFIDENCE_THRESHOLD = 68           # slightly above decision_agent's 65 gate
PRICE_MIN            = 0.50
PRICE_MAX            = 50.0
MIN_AVG_VOLUME       = 50_000       # ignore dead / illiquid stocks
MAX_PER_POLL         = 2            # max full-pipeline runs per cycle (rate limit)

_seen_ids:   set  = set()   # Polygon article IDs already processed
_alerted_at: dict = {}      # ticker → time.time() of last alert
_executor = ThreadPoolExecutor(max_workers=1)


# ── News fetching ──────────────────────────────────────────────────────────────

def _fetch_latest_news(limit: int = 50) -> list:
    """
    Fetch latest articles from Polygon for ALL US stocks (no ticker filter).
    Each result includes 'tickers', 'id', 'title', 'published_utc'.
    """
    try:
        r = requests.get(
            f"{BASE_URL}/v2/reference/news",
            params={"apiKey": POLYGON_API_KEY, "limit": limit, "order": "desc"},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        return r.json().get("results", [])
    except Exception as e:
        print(f"⚠️  [NewsWatcher] Fetch error: {e}")
        return []


def _filter_new(articles: list) -> list:
    """Return only articles with IDs not yet seen. Marks them as seen."""
    new = [a for a in articles if a.get("id") and a["id"] not in _seen_ids]
    for a in new:
        _seen_ids.add(a["id"])
    if len(_seen_ids) > 5000:
        _seen_ids.clear()   # reset periodically; _alerted_at prevents spam
    return new


# ── Gate helpers ───────────────────────────────────────────────────────────────

def _quick_gate(ticker: str) -> bool:
    """
    Fast price + volume check via yfinance fast_info (0 Polygon API calls).
    Returns True if the stock passes — False means skip.
    """
    try:
        fi = yf.Ticker(ticker).fast_info
        price = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
        vol   = getattr(fi, "three_month_average_volume", None)
        if price is None:
            return False
        price = float(price)
        if not (PRICE_MIN <= price <= PRICE_MAX):
            return False
        if vol and float(vol) < MIN_AVG_VOLUME:
            return False
        return True
    except Exception:
        return False


def _already_alerted(ticker: str) -> bool:
    """True if we sent a WhatsApp for this ticker within the cooldown window."""
    last = _alerted_at.get(ticker)
    return last is not None and (time.time() - last) < ALERT_COOLDOWN_S


def _mark_alerted(ticker: str):
    _alerted_at[ticker] = time.time()


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _run_pipeline(ticker: str, paper: bool) -> dict:
    return GRAPH.invoke(make_initial_state(ticker, paper_trading=paper))


def _format_alert(result: dict, headline: str) -> str:
    ticker  = result.get("ticker", "?")
    signal  = result.get("signal", "HOLD")
    price   = result.get("current_price", 0.0)
    conf    = result.get("confidence", 0)
    targets = result.get("targets", [])
    stop    = result.get("stop_loss", 0.0)
    entry   = result.get("entry_zone", "")
    reason  = result.get("reasoning", "")[:200]
    action  = result.get("action_plan", "")[:150]
    rr      = result.get("rr_ratio", 0.0)

    emoji  = "🟢 BUY" if signal == "BUY" else "🔴 SELL"
    t1     = f"${targets[0]:.2f}" if targets else "—"
    t2     = f" / ${targets[1]:.2f}" if len(targets) > 1 else ""
    now_s  = datetime.now().strftime("%H:%M")

    return (
        f"📡 NEWS ALERT [{now_s}] — {emoji}\n"
        f"{ticker} @ ${price:.2f}  conf={conf}/100  R:R {rr:.1f}x\n\n"
        f"News: {headline[:80]}\n\n"
        f"Entry: {entry}\n"
        f"Target: {t1}{t2}  |  Stop: ${stop:.2f}\n\n"
        f"Why: {reason}\n\n"
        f"Plan: {action}"
    )


# ── Main loop ──────────────────────────────────────────────────────────────────

async def news_watcher_loop(paper: bool = False):
    """
    Async loop started in main.py lifespan alongside monitoring and scheduler.
    Runs forever until cancelled.
    """
    mode = "PAPER" if paper else "LIVE"
    print(f"📡 [NewsWatcher] Started — poll every {POLL_INTERVAL_S//60}min | {mode}")

    loop = asyncio.get_running_loop()

    # ── Seed seen IDs so startup doesn't flood old articles ─────────────────
    print("📡 [NewsWatcher] Seeding article history...")
    seed = _fetch_latest_news(limit=100)
    for a in seed:
        if a.get("id"):
            _seen_ids.add(a["id"])
    print(f"📡 [NewsWatcher] Ready. Watching for new articles ({len(_seen_ids)} seeded).")

    while True:
        try:
            await asyncio.sleep(POLL_INTERVAL_S)

            # ── 1. Fetch latest news ────────────────────────────────────────
            articles = _fetch_latest_news(limit=50)
            new_arts = _filter_new(articles)

            now_str = datetime.now().strftime("%H:%M")
            if not new_arts:
                print(f"📡 [NewsWatcher] {now_str} — no new articles")
                continue

            print(f"📡 [NewsWatcher] {now_str} — {len(new_arts)} new article(s)")

            # ── 2. Collect unique candidate tickers ─────────────────────────
            candidates     = []
            seen_this_poll = set()

            for article in new_arts:
                headline = article.get("title", "")
                for ticker in article.get("tickers", []):
                    if ticker in seen_this_poll:
                        continue
                    seen_this_poll.add(ticker)
                    if _already_alerted(ticker):
                        continue
                    candidates.append({"ticker": ticker, "headline": headline})

            if not candidates:
                print(f"📡 [NewsWatcher] All candidates in cooldown — skipping")
                continue

            print(f"📡 [NewsWatcher] {len(candidates)} candidate(s) after cooldown filter")

            # ── 3. Quick gate (no Polygon calls) ────────────────────────────
            gated = []
            for c in candidates:
                if _quick_gate(c["ticker"]):
                    gated.append(c)
                    print(f"   ✅ {c['ticker']}: passed gate")
                else:
                    print(f"   ❌ {c['ticker']}: failed price/volume gate")
                if len(gated) >= MAX_PER_POLL:
                    break

            if not gated:
                continue

            # ── 4. Full pipeline for each gated candidate ────────────────────
            for i, c in enumerate(gated):
                ticker   = c["ticker"]
                headline = c["headline"]

                if i > 0:
                    # Buffer between pipeline runs to respect Polygon rate limit
                    await asyncio.sleep(30)

                print(f"\n📡 [NewsWatcher] Analyzing {ticker}...")
                print(f"   📰 {headline[:70]}")

                try:
                    result = await loop.run_in_executor(
                        _executor, _run_pipeline, ticker, paper
                    )

                    signal = result.get("signal", "HOLD")
                    conf   = result.get("confidence", 0)
                    emoji  = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"

                    print(f"   {emoji} {ticker} → {signal} {conf}/100")

                    if signal in ("BUY", "SELL") and conf >= CONFIDENCE_THRESHOLD:
                        msg = _format_alert(result, headline)
                        _mark_alerted(ticker)
                        if not paper:
                            send_whatsapp(msg)
                            print(f"   ✅ [NewsWatcher] WhatsApp sent for {ticker}")
                        else:
                            print(f"   📋 [NewsWatcher] PAPER — would send:\n{msg}")
                    else:
                        print(f"   💤 {ticker} → {signal} {conf}/100 — below threshold ({CONFIDENCE_THRESHOLD})")

                except Exception as e:
                    print(f"   ❌ [NewsWatcher] Pipeline error for {ticker}: {e}")

        except asyncio.CancelledError:
            print("📡 [NewsWatcher] Stopped.")
            break
        except Exception as e:
            print(f"❌ [NewsWatcher] Loop error: {e}")
            await asyncio.sleep(60)
