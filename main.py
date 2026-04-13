"""
main.py — entry point for the Stock AI Agent.

Usage:
  python main.py                         # monitor TICKER from .env
  python main.py --ticker AAPL           # override ticker
  python main.py --ticker BZAI --paper   # paper trading (no real alerts)
  python main.py --interval 120          # check every 2 minutes
  python main.py --port 8080             # dashboard on custom port

Features:
  - Market hours awareness (9:30 AM - 4:00 PM EST)
  - Signal memory + stop loss / target monitoring
  - Daily summary report at 4:30 PM EST
  - Paper trading mode (logs only, no SMS/push)
  - FastAPI dashboard at http://localhost:{port}
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
import math
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import config
from graph import GRAPH, make_initial_state
import watchlist_manager as wl
import logger


# ── CLI args ───────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Stock AI Agent")
    p.add_argument("--ticker",   nargs="+", default=None,  help="One or more ticker symbols (e.g. --ticker BZAI AWRE)")
    p.add_argument("--interval", type=int, default=300, help="Scan interval seconds (default 300 = 5 min)")
    p.add_argument("--paper",    action="store_true",   help="Paper trading — no real SMS/push alerts")
    p.add_argument("--port",     type=int, default=8000, help="Dashboard port (default 8000)")
    # Watchlist management (these exit immediately without starting the server)
    p.add_argument("--add",    metavar="TICKER", default=None, help="Add ticker to watchlist and exit")
    p.add_argument("--remove", metavar="TICKER", default=None, help="Remove ticker from watchlist and exit")
    p.add_argument("--list",   action="store_true",            help="List watchlist tickers and exit")
    args, _ = p.parse_known_args()
    return args

# ── Safe module-level defaults (populated by _setup_from_args at startup) ──────
TICKERS  = ["BZAI"]
TICKER   = "BZAI"
INTERVAL = 300
PAPER    = False
PORT     = 8000
EST      = ZoneInfo("America/New_York")


def _setup_from_args() -> None:
    """Parse CLI args and populate module-level config. Called only from __main__."""
    global TICKERS, TICKER, INTERVAL, PAPER, PORT

    args = _parse_args()

    # Handle watchlist management commands immediately (no server needed)
    if args.add:
        wl.add(args.add)
        raise SystemExit(0)
    if args.remove:
        wl.remove(args.remove)
        raise SystemExit(0)
    if args.list:
        wl.list_tickers()
        raise SystemExit(0)

    # Resolve tickers: CLI flag → .env → watchlist → default
    watchlist   = wl.load()
    cli_tickers = args.ticker
    if cli_tickers:
        TICKERS = [t.upper() for t in cli_tickers]
    elif watchlist:
        TICKERS = watchlist
    elif config.TICKER:
        TICKERS = [t.strip().upper() for t in config.TICKER.split() if t.strip()]
    else:
        TICKERS = ["BZAI"]

    TICKER   = TICKERS[0]
    INTERVAL = args.interval
    PAPER    = args.paper
    PORT     = args.port


# ── Market hours helpers ───────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(tz=EST)
    if now.weekday() >= 5:            # Saturday / Sunday
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


def is_report_window() -> bool:
    """True for INTERVAL seconds after 4:30 PM EST on weekdays."""
    now = datetime.now(tz=EST)
    if now.weekday() >= 5:
        return False
    report_t = now.replace(hour=16, minute=30, second=0, microsecond=0)
    delta = (now - report_t).total_seconds()
    return 0 <= delta < INTERVAL


# ── Runtime state ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    """
    All mutable runtime state in one place.
    Swap with AppState() in tests for full isolation — no module-level patching needed.
    """
    ticker_states:    dict = field(default_factory=dict)   # {ticker: state_dict}
    histories:        dict = field(default_factory=dict)   # {ticker: [last 200]}
    bars_map:         dict = field(default_factory=dict)   # {ticker: bars_list}
    news_map:         dict = field(default_factory=dict)   # {ticker: news_list}
    signal_memory:    dict = field(default_factory=dict)   # {ticker: {signal, price, ...}}
    daily_log:        list = field(default_factory=list)   # BUY/SELL events today
    report_sent_date: str  = ""                            # "YYYY-MM-DD"


_app_state = AppState()
_executor  = ThreadPoolExecutor(max_workers=1)


# ── Graph execution ────────────────────────────────────────────────────────────

def _run_sync(ticker: str, paper: bool) -> dict:
    return GRAPH.invoke(make_initial_state(ticker, paper_trading=paper))


def _store_result(result: dict):
    ticker = result.get("ticker", TICKER)

    bars  = result.get("bars", [])
    news  = result.get("raw_news", [])
    state = {k: v for k, v in result.items() if k not in ("bars", "snapshot", "raw_news")}
    state["timestamp"] = datetime.now().isoformat()

    _app_state.bars_map[ticker]       = bars
    _app_state.news_map[ticker]       = news
    _app_state.ticker_states[ticker]  = state

    hist = _app_state.histories.setdefault(ticker, [])
    hist.append({
        "timestamp":  state["timestamp"],
        "price":      state.get("current_price", 0),
        "signal":     state.get("signal", "HOLD"),
        "confidence": state.get("confidence", 0),
        "rsi":        state.get("rsi", 50),
    })
    if len(hist) > 200:
        hist.pop(0)


def _update_signal_memory(result: dict):
    ticker  = result.get("ticker", TICKER)
    signal  = result.get("signal", "HOLD")
    price   = result.get("current_price", 0.0)

    if signal in ("BUY", "SELL"):
        _app_state.signal_memory[ticker] = {
            "signal":    signal,
            "price":     price,
            "stop_loss": result.get("stop_loss", 0.0),
            "targets":   result.get("targets", []),
            "timestamp": datetime.now().isoformat(),
        }
        _app_state.daily_log.append({
            "signal":     signal,
            "ticker":     ticker,
            "price":      price,
            "confidence": result.get("confidence", 0),
            "timestamp":  datetime.now().isoformat(),
        })
        # Persist signal to CSV log
        logger.log_signal(result)


def _check_exits(ticker: str, price: float) -> str | None:
    mem = _app_state.signal_memory.get(ticker)
    if not mem or mem.get("signal") != "BUY":
        return None
    stop        = mem.get("stop_loss", 0.0)
    entry_price = mem.get("price", 0.0)
    # Only count targets that are genuinely above the entry price
    targets = [t for t in mem.get("targets", []) if t > entry_price]
    if stop and price <= stop:
        return "STOP_LOSS"
    if targets and price >= min(targets):
        return "TARGET_HIT"
    return None


async def run_once(ticker: str = None):
    ticker = ticker or TICKER
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run_sync, ticker, PAPER)

    _store_result(result)
    _update_signal_memory(result)

    # Stop loss / target monitoring
    price       = result.get("current_price", 0.0)
    exit_reason = _check_exits(ticker, price)
    if exit_reason:
        icon = "🛑" if exit_reason == "STOP_LOSS" else "🎯"
        print(f"{icon} [Monitor] {exit_reason} triggered for {ticker} @ ${price:.4f}")
        if not PAPER:
            try:
                from alerts import send_push, send_whatsapp
                label = "🛑 STOP LOSS HIT" if exit_reason == "STOP_LOSS" else "🎯 TARGET HIT"
                msg   = f"{label}\n{ticker} @ ${price:.4f}"
                send_push(f"{exit_reason} — {ticker}", msg)
                send_whatsapp(msg)
            except Exception as e:
                print(f"❌ [Monitor] Exit alert failed: {e}")
        _app_state.signal_memory.pop(ticker, None)

    sig  = result.get("signal", "HOLD")
    conf = result.get("confidence", 0)
    icon = "🟢" if sig == "BUY" else "🔴" if sig == "SELL" else "🟡"
    print(
        f"{icon} [Monitor] {sig}  conf={conf}/100  "
        f"${price:.4f}  RSI={result.get('rsi', 0):.1f}"
        + ("  📋 PAPER" if PAPER else "")
    )


# ── Daily report ───────────────────────────────────────────────────────────────

async def _send_daily_report():
    today    = datetime.now(tz=EST).strftime("%Y-%m-%d")
    today_signals = [e for e in _app_state.daily_log if e["timestamp"][:10] == today]

    lines = [f"📅 Daily Report — {TICKER} — {today}"]
    if not today_signals:
        lines.append("No BUY/SELL signals fired today.")
    else:
        lines.append(f"Signals fired: {len(today_signals)}")
        for s in today_signals:
            icon = "🟢" if s["signal"] == "BUY" else "🔴"
            lines.append(f"  {icon} {s['signal']} @ ${s['price']:.4f}  conf={s['confidence']}")
    if PAPER:
        lines.append("\n📋 PAPER TRADING MODE")

    report = "\n".join(lines)
    print("\n" + "─" * 50)
    print(report)
    print("─" * 50 + "\n")

    if not PAPER:
        try:
            from alerts import send_push, send_whatsapp
            send_push(f"Daily Report — {TICKER}", report)
            send_whatsapp(report)
        except Exception as e:
            print(f"❌ [Monitor] Daily report delivery failed: {e}")


# ── Monitoring loop ────────────────────────────────────────────────────────────

async def monitoring_loop():
    mode = "📋 PAPER" if PAPER else "🔴 LIVE"
    print(f"🚀 Stock AI Agent starting  [{mode}]  tickers={', '.join(TICKERS)}  interval={INTERVAL}s")
    print(f"📊 Dashboard → http://localhost:{PORT}")

    # Print watchlist on startup
    saved = wl.load()
    if saved:
        print(f"📋 Watchlist: {', '.join(saved)}")
    print()

    while True:
        try:
            if is_market_open():
                for i, t in enumerate(TICKERS):
                    if i > 0:
                        await asyncio.sleep(20)  # avoid Polygon 5 req/min rate limit
                    await run_once(t)
            else:
                now_est  = datetime.now(tz=EST)
                now_date = now_est.strftime("%Y-%m-%d")
                print(
                    f"🕐 [Monitor] Market closed "
                    f"({now_est.strftime('%a %H:%M EST')}) — "
                    f"next check in {INTERVAL}s"
                )
        except Exception as e:
            print(f"❌ [Monitor] Loop error: {e}")

        await asyncio.sleep(INTERVAL)


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from scheduler import scheduler_loop
    from news_watcher import news_watcher_loop
    task_monitor   = asyncio.create_task(monitoring_loop())
    task_scheduler = asyncio.create_task(
        scheduler_loop(paper=PAPER, daily_log=_app_state.daily_log,
                        signal_memory=_app_state.signal_memory)
    )
    task_watcher   = asyncio.create_task(news_watcher_loop(paper=PAPER))
    yield
    task_monitor.cancel()
    task_scheduler.cancel()
    task_watcher.cancel()


app = FastAPI(title=f"Stock AI Agent — {TICKER}", lifespan=lifespan)


def _json_safe(value):
    """Recursively replace NaN/Inf values so FastAPI JSON responses stay valid."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    try:
        with open("dashboard/index.html") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>dashboard/index.html not found</h2>"


@app.get("/api/state")
async def api_state(ticker: str = None):
    t = (ticker or TICKER).upper()
    state   = _app_state.ticker_states.get(t, {})
    history = _app_state.histories.get(t, [])
    return JSONResponse(_json_safe({
        "state":         state,
        "history":       history,
        "paper_trading": PAPER,
        "ticker":        t,
        "tickers":       TICKERS,
        "market_open":   is_market_open(),
    }))


@app.get("/api/bars")
async def api_bars(ticker: str = None):
    t    = (ticker or TICKER).upper()
    bars = _app_state.bars_map.get(t, [])
    tv_bars = [
        {
            "time":   b.get("t", 0) // 1000,
            "open":   b.get("o"),
            "high":   b.get("h"),
            "low":    b.get("l"),
            "close":  b.get("c"),
            "volume": b.get("v"),
        }
        for b in bars
    ]
    return JSONResponse(_json_safe({"bars": tv_bars, "ticker": t}))


@app.get("/api/news")
async def api_news():
    raw  = _app_state.news_map.get(TICKER, [])
    news = [
        {
            "title":     n.get("title", ""),
            "url":       n.get("article_url", ""),
            "published": (n.get("published_utc") or "")[:10],
            "source":    (n.get("publisher") or {}).get("name", ""),
        }
        for n in raw[:10]
    ]
    return JSONResponse(_json_safe({"news": news, "ticker": TICKER}))


@app.get("/api/watchlist")
async def api_watchlist():
    rows = []
    for t in TICKERS:
        s = _app_state.ticker_states.get(t, {})
        rows.append({
            "ticker":     t,
            "signal":     s.get("signal", "—"),
            "confidence": s.get("confidence", 0),
            "price":      s.get("current_price", 0),
            "rsi":        s.get("rsi", 0),
        })
    return JSONResponse(_json_safe({"watchlist": rows, "memory": _app_state.signal_memory}))


@app.post("/api/run")
async def api_trigger_run():
    async def _run_all():
        for i, t in enumerate(TICKERS):
            if i > 0:
                await asyncio.sleep(20)
            await run_once(t)
    asyncio.create_task(_run_all())
    return {"status": "triggered", "tickers": TICKERS}


@app.get("/api/log")
async def api_log():
    """Return last 100 rows of signals_log.csv for the dashboard."""
    return JSONResponse(_json_safe({"log": logger.read_log(limit=100)}))


@app.get("/api/watchlist/saved")
async def api_watchlist_saved():
    """Return the persisted watchlist.json tickers."""
    return JSONResponse(_json_safe({"tickers": wl.load()}))


@app.post("/api/watchlist/add/{ticker}")
async def api_watchlist_add(ticker: str):
    updated = wl.add(ticker.upper())
    return {"tickers": updated}


@app.delete("/api/watchlist/remove/{ticker}")
async def api_watchlist_remove(ticker: str):
    updated = wl.remove(ticker.upper())
    return {"tickers": updated}


if __name__ == "__main__":
    _setup_from_args()
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)
