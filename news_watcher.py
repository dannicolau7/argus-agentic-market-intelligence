"""
news_watcher.py — 24/7 news-driven stock discovery.

Three complementary news sources, all sharing the same 4-hour cooldown:

  1. Polygon /v2/reference/news (ALL stocks, every 5 min) — broad market coverage,
     ~15–45 min lag from press release publication.

  2. Yahoo Finance news (watchlist tickers only, every 90 s) — faster source,
     typically 2–5 min lag. yf_news_watcher_loop() runs alongside.

  3. SEC EDGAR 8-K RSS (edgar_watcher.py, every 60 s) — catches FDA decisions,
     earnings, contract wins at filing time, before any news aggregator.

Polygon watcher kept for broad-market discovery (finds stocks not on watchlist).
Yahoo Finance + EDGAR fill the latency gap for watchlist tickers.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
import yfinance as yf

import watchlist_manager as wl
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

_seen_ids:    set  = set()   # Polygon article IDs already processed
_yf_seen_ids: set  = set()   # Yahoo Finance article UUIDs already processed
_alerted_at:  dict = {}      # ticker → time.time() of last alert (shared across all sources)
_executor = ThreadPoolExecutor(max_workers=2)


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

def _run_pipeline(ticker: str, paper: bool, news_triggered: bool = True) -> dict:
    """Run the full LangGraph pipeline. news_triggered=True lowers confidence threshold to 55."""
    state = make_initial_state(ticker, paper_trading=paper)
    state["news_triggered"] = news_triggered
    return GRAPH.invoke(state)


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


# ── Yahoo Finance news watcher (90-second poll, watchlist tickers only) ────────

async def _check_yf_ticker_news(ticker: str, paper: bool, loop) -> None:
    """
    Fetch Yahoo Finance news for one ticker and run the pipeline if a new
    article is found. Shares _alerted_at and _quick_gate with Polygon watcher.
    """
    try:
        arts = await loop.run_in_executor(
            None, lambda: yf.Ticker(ticker).news or []
        )
    except Exception:
        return

    new_arts = []
    for art in arts:
        uid = art.get("uuid") or art.get("id") or ""
        if not uid or uid in _yf_seen_ids:
            continue
        _yf_seen_ids.add(uid)
        age_s = time.time() - art.get("providerPublishTime", 0)
        if age_s > 7200:   # ignore articles older than 2 hours
            continue
        new_arts.append(art)

    if not new_arts:
        return
    if _already_alerted(ticker):
        return
    if not _quick_gate(ticker):
        return

    headline = new_arts[0].get("title", "")
    now_s    = datetime.now().strftime("%H:%M")
    print(f"\n📰 [YF-News] {ticker}: {headline[:70]}")

    try:
        result = await loop.run_in_executor(
            _executor, _run_pipeline, ticker, paper, True
        )
        signal = result.get("signal", "HOLD")
        conf   = result.get("confidence", 0)
        emoji  = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
        print(f"   {emoji} [YF-News] {ticker} → {signal} {conf}/100")

        if signal in ("BUY", "SELL") and conf >= CONFIDENCE_THRESHOLD:
            msg = _format_alert(result, headline)
            _mark_alerted(ticker)
            if not paper:
                send_whatsapp(msg)
                print(f"   ✅ [YF-News] Alert sent for {ticker}")
            else:
                print(f"   📋 [YF-News] PAPER — would send:\n{msg}")
        else:
            print(f"   💤 [YF-News] {ticker} → below threshold ({CONFIDENCE_THRESHOLD})")
    except Exception as e:
        print(f"   ❌ [YF-News] Pipeline error for {ticker}: {e}")


async def yf_news_watcher_loop(paper: bool = False):
    """
    Polls Yahoo Finance news for every watchlist ticker every 90 seconds.
    Much faster than Polygon (2–5 min lag vs 15–45 min).
    Started as a separate asyncio task alongside news_watcher_loop.
    """
    mode = "PAPER" if paper else "LIVE"
    print(f"📰 [YF-News] Started — poll every 90s | {mode}")
    loop = asyncio.get_running_loop()

    # Seed: mark current articles as seen so startup doesn't flood old news
    print("📰 [YF-News] Seeding Yahoo Finance article history...")
    init_tickers = wl.load()
    for ticker in init_tickers:
        try:
            arts = await loop.run_in_executor(
                None, lambda t=ticker: yf.Ticker(t).news or []
            )
            for art in arts:
                uid = art.get("uuid") or art.get("id") or ""
                if uid:
                    _yf_seen_ids.add(uid)
        except Exception:
            pass
    print(f"📰 [YF-News] Seeded {len(_yf_seen_ids)} articles. Watching {len(init_tickers)} ticker(s).")

    while True:
        try:
            await asyncio.sleep(90)
            tickers = wl.load()
            for ticker in tickers:
                await _check_yf_ticker_news(ticker, paper, loop)
        except asyncio.CancelledError:
            print("📰 [YF-News] Stopped.")
            break
        except Exception as e:
            print(f"❌ [YF-News] Loop error: {e}")
            await asyncio.sleep(30)


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
