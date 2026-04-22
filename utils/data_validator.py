"""
utils/data_validator.py — Hard data-quality gates.

Prevents the pipeline from sending signals when market data is missing,
stale, or inconsistent.  All public functions are safe to call from any
agent — they never raise, only return error codes and log internally.

Public API
----------
validate_market_data(ticker, price, volume, ohlcv) -> list[str]
    Returns a list of error codes (empty = data is valid).

check_market_hours() -> tuple[bool, str]
    Returns (allowed, reason). BUY/SELL signals are only allowed
    9:30 AM–4:15 PM ET on weekdays.

is_market_open() -> bool
    True if market is currently open (Polygon API + time fallback).

log_validation_failure(ticker, errors, context) -> None
    Appends a JSON line to data/errors.log.

send_admin_alert(ticker, errors) -> None
    Sends a single WhatsApp admin notice (rate-limited to 1 per 30 min).
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

_EST = ZoneInfo("America/New_York")
_ERRORS_LOG    = Path(__file__).parent.parent / "data" / "errors.log"
_MIN_BARS      = 20      # minimum daily OHLCV bars required for technical analysis
_PRICE_MAX_DEV = 0.05    # 5% maximum allowed deviation vs independent yfinance quote
_STATUS_TTL    = 300     # seconds — Polygon market-status cache lifetime
_ADMIN_COOLDOWN = 1800   # seconds — minimum gap between admin WhatsApp alerts

_market_cache: dict = {"open": None, "ts": 0.0}
_last_admin:   dict = {"ts": 0.0}


# ── Market hours ───────────────────────────────────────────────────────────────

def _polygon_market_open() -> Optional[bool]:
    """Query Polygon /v1/marketstatus/now. Returns None if unavailable."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return None
    try:
        import requests
        r = requests.get(
            "https://api.polygon.io/v1/marketstatus/now",
            params={"apiKey": api_key},
            timeout=5,
        )
        if r.ok:
            return r.json().get("market") == "open"
    except Exception:
        pass
    return None


def is_market_open() -> bool:
    """
    True if the US equities market is currently open.
    Uses Polygon API (cached 5 min); falls back to time-based check.
    """
    now_ts = time.monotonic()
    if now_ts - _market_cache["ts"] < _STATUS_TTL and _market_cache["open"] is not None:
        return _market_cache["open"]

    result = _polygon_market_open()
    if result is None:
        est = datetime.now(tz=_EST)
        if est.weekday() >= 5:
            result = False
        else:
            open_t  = est.replace(hour=9,  minute=30, second=0, microsecond=0)
            close_t = est.replace(hour=16, minute=0,  second=0, microsecond=0)
            result  = open_t <= est <= close_t

    _market_cache.update({"open": result, "ts": now_ts})
    return result


def check_market_hours() -> tuple:
    """
    Returns (allowed: bool, reason: str).

    BUY/SELL signals are only generated 9:30 AM–4:15 PM ET, Mon–Fri.
    This window extends 15 min past the close so that 4:00 PM prints
    (which sometimes arrive at 4:01–4:02) can still be processed.
    """
    est = datetime.now(tz=_EST)
    wd  = est.weekday()

    if wd >= 5:
        return False, f"WEEKEND ({est.strftime('%A')})"

    open_t  = est.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = est.replace(hour=16, minute=15, second=0, microsecond=0)

    if est < open_t:
        mins = int((open_t - est).total_seconds() / 60)
        return False, f"PRE_MARKET: {mins}min until open"

    if est > close_t:
        mins = int((est - close_t).total_seconds() / 60)
        return False, f"AFTER_HOURS: {mins}min since close"

    return True, "MARKET_OPEN"


# ── Data validation ────────────────────────────────────────────────────────────

def validate_market_data(
    ticker: str,
    price:  float,
    volume: float,
    ohlcv:  list,
) -> list:
    """
    Cross-check market data quality.  Returns a list of error code strings.
    Empty list means the data is valid.

    Checks:
      MISSING_PRICE        — price is None / 0
      PRICE_MISMATCH       — >5% deviation from independent yfinance quote
      MISSING_VOLUME       — volume is None / 0
      INSUFFICIENT_HISTORY — fewer than 20 daily bars
    """
    errors: list = []

    # ── Check 1: price must exist ──────────────────────────────────────────────
    if not price or price <= 0:
        errors.append("MISSING_PRICE")
        return errors   # cannot cross-check without a price

    # ── Check 2: cross-verify against yfinance (independent source) ────────────
    try:
        import yfinance as yf
        fi = yf.Ticker(ticker).fast_info
        yf_price = float(fi["last_price"] or 0)
        if yf_price > 0:
            deviation = abs(price - yf_price) / yf_price
            if deviation > _PRICE_MAX_DEV:
                errors.append(
                    f"PRICE_MISMATCH: pipeline=${price:.4f} "
                    f"yfinance=${yf_price:.4f} "
                    f"({deviation * 100:.1f}% deviation)"
                )
    except Exception:
        pass   # yfinance unavailable — not a blocking error by itself

    # ── Check 3: volume must be real ───────────────────────────────────────────
    if not volume or volume <= 0:
        errors.append("MISSING_VOLUME")

    # ── Check 4: need enough history for technical analysis ────────────────────
    actual = len(ohlcv) if ohlcv else 0
    if actual < _MIN_BARS:
        errors.append(f"INSUFFICIENT_HISTORY: {actual}/{_MIN_BARS} daily bars")

    return errors


# ── Logging & admin alerts ─────────────────────────────────────────────────────

def log_validation_failure(
    ticker:  str,
    errors:  list,
    context: Optional[dict] = None,
) -> None:
    """Appends a structured JSON line to data/errors.log."""
    try:
        _ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts":     datetime.now(tz=timezone.utc).isoformat(),
            "ticker": ticker,
            "errors": errors,
            "price":  (context or {}).get("current_price", 0),
            "volume": (context or {}).get("volume", 0),
            "bars":   len((context or {}).get("bars", [])),
        }
        with open(_ERRORS_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"⚠️  [DataValidator] errors.log write failed: {e}")


def send_admin_alert(ticker: str, errors: list) -> None:
    """
    Send a single WhatsApp admin notice when validation fails.
    Rate-limited to at most one message per 30 minutes.
    """
    now = time.monotonic()
    if now - _last_admin["ts"] < _ADMIN_COOLDOWN:
        return

    _last_admin["ts"] = now
    try:
        from alerts import send_whatsapp
        msg = (
            f"⚠️ Signal blocked — data validation failed\n"
            f"Ticker: {ticker}\n"
            f"Errors: {', '.join(errors)}"
        )
        send_whatsapp(msg)
        print(f"📲 [DataValidator] Admin alert sent for {ticker}: {errors}")
    except Exception as e:
        print(f"⚠️  [DataValidator] Admin alert failed: {e}")
