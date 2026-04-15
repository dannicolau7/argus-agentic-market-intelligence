"""
agents/execution_agent.py — LangGraph node: final execution gate.

Runs after sizing_agent, before alert_agent.
Last check before an alert fires — converts sized trades into actionable orders
or blocks them if conditions have changed since decision_agent ran.

Checks (in order):
  1. Signal staleness   — timestamp > SIGNAL_MAX_AGE_MIN → too old, convert to WATCH
  2. Price extended     — price > entry_high × (1 + ENTRY_ZONE_SLACK) → chasing, block
  3. Price retreated    — price < entry_low × (1 - 3%) → setup invalidated, block
  4. Volume thin        — volume < 40% of avg → warn (proceed with caution, don't block)

Order type recommendation:
  STAGGERED  — scale_in=True; enter in thirds at entry, +2%, +4%
  MARKET     — news_triggered and price still in zone; fill immediately
  LIMIT      — standard; use limit at entry midpoint

On block: signal → "WATCH", should_alert → False, order_type → "NONE"
On pass:  executable → True, order_type set, should_alert preserved
"""

from datetime import datetime, timezone

from config import ENTRY_ZONE_SLACK, SIGNAL_MAX_AGE_MIN

_PRICE_RETREAT_PCT = 0.03   # price dropped 3% below entry zone low → invalidated
_MIN_VOLUME_RATIO  = 0.40   # volume < 40% of avg → warn (low conviction)


def execution_node(state: dict) -> dict:
    ticker        = state["ticker"]
    signal        = state.get("signal", "HOLD")
    risk_approved = state.get("risk_approved", True)
    max_shares    = state.get("max_shares", 0)

    # Only gate risk-approved BUY signals that have shares allocated
    if signal != "BUY" or not risk_approved or max_shares == 0:
        reason = "Not a risk-approved BUY with shares allocated"
        print(f"⏭️  [ExecutionAgent] {ticker}: skipped — {reason}")
        return {**state,
                "executable":       False,
                "execution_reason": reason,
                "order_type":       "NONE"}

    price          = float(state.get("current_price", 0.0))
    volume         = float(state.get("volume", 0.0))
    avg_volume     = float(state.get("avg_volume", 0.0))
    entry_zone     = state.get("entry_zone", "")
    timestamp      = state.get("timestamp", "")
    news_triggered = state.get("news_triggered", False)
    scale_in       = state.get("scale_in", False)

    if price <= 0:
        return _block(state, ticker, "Price unavailable — cannot validate entry")

    # ── Parse entry zone ──────────────────────────────────────────────────────
    try:
        parts      = entry_zone.replace("$", "").split("-")
        entry_low  = float(parts[0].strip())
        entry_high = float(parts[1].strip())
    except Exception:
        entry_low = entry_high = price   # fallback: treat current price as zone

    # ── 1. Signal staleness ───────────────────────────────────────────────────
    age_min = 0.0
    try:
        signal_time = datetime.fromisoformat(timestamp)
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)
        age_min = (datetime.now(timezone.utc) - signal_time).total_seconds() / 60
        if age_min > SIGNAL_MAX_AGE_MIN:
            return _block(state, ticker,
                f"Signal stale ({age_min:.0f} min > {SIGNAL_MAX_AGE_MIN} min limit) — "
                f"conditions may have changed")
    except Exception:
        pass   # timestamp parse failure → treat as fresh

    # ── 2. Price extended above entry zone ────────────────────────────────────
    extended_ceiling = entry_high * (1 + ENTRY_ZONE_SLACK)
    if price > extended_ceiling:
        chased_pct = (price / entry_high - 1) * 100
        return _block(state, ticker,
            f"Price ${price:.2f} too extended (+{chased_pct:.1f}% above zone top "
            f"${entry_high:.2f}) — don't chase")

    # ── 3. Price retreated below entry zone ───────────────────────────────────
    retreat_floor = entry_low * (1 - _PRICE_RETREAT_PCT)
    if price < retreat_floor:
        drop_pct = (entry_low / price - 1) * 100
        return _block(state, ticker,
            f"Price ${price:.2f} retreated {drop_pct:.1f}% below zone bottom "
            f"${entry_low:.2f} — setup invalidated")

    # ── 4. Volume confirmation (warn only, don't block) ───────────────────────
    low_volume  = avg_volume > 0 and volume < avg_volume * _MIN_VOLUME_RATIO
    volume_note = ""
    if low_volume:
        vol_ratio   = volume / avg_volume if avg_volume > 0 else 0
        volume_note = f" ⚠️ low vol {vol_ratio:.1f}×"

    # ── Order type selection ──────────────────────────────────────────────────
    if scale_in:
        order_type = "STAGGERED"   # high ATR → enter in thirds
    elif news_triggered:
        order_type = "MARKET"      # catalyst move — fill immediately
    else:
        order_type = "LIMIT"       # standard — limit at entry midpoint

    reason = (
        f"${price:.2f} in zone ${entry_low:.2f}–${entry_high:.2f}  "
        f"age {age_min:.0f}min  → {order_type}{volume_note}"
    )

    print(f"✅ [ExecutionAgent] {ticker}: EXECUTABLE  {order_type}  {reason}")

    return {
        **state,
        "executable":       True,
        "execution_reason": reason,
        "order_type":       order_type,
    }


def _block(state: dict, ticker: str, reason: str) -> dict:
    print(f"🚫 [ExecutionAgent] {ticker}: BLOCKED — {reason}")
    return {
        **state,
        "executable":       False,
        "execution_reason": reason,
        "order_type":       "NONE",
        "should_alert":     False,
        "signal":           "WATCH",   # valid idea, not actionable right now
    }
