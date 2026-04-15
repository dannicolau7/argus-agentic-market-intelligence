"""
agents/sizing_agent.py — LangGraph node: position sizing.

Runs after risk_agent (only if risk_approved=True).
Determines exactly how large the position should be.

Sizing hierarchy (each layer can only reduce, never increase):
  1. Stop-based risk sizing  — risk_dollars / (entry - stop) = max_shares
  2. Conviction multiplier   — confidence 90+ → full, 65 → 60%
  3. Volatility multiplier   — from risk_agent's risk_multiplier
  4. Liquidity cap           — never exceed 5% of avg daily volume
  5. Portfolio cap           — never exceed 25% of total portfolio in one name
  6. Scale-in decision       — if ATR > 5% or market is choppy → scale in thirds

Output fields added to state:
  position_size_pct   float   fraction of portfolio to deploy (0.0 – 0.25)
  position_size_usd   float   dollar amount
  max_shares          int     at the given entry price
  risk_dollars        float   amount at risk (entry_mid - stop) * shares
  scale_in            bool    True = enter in thirds (immediate, +2%, +4%)
  size_reasoning      str     one-line explanation
"""

from config import PORTFOLIO_SIZE, MAX_RISK_PER_TRADE

MAX_POSITION_PCT  = 0.25     # hard cap: never more than 25% of portfolio in one name
LIQUIDITY_CAP_PCT = 0.05     # never more than 5% of avg daily volume
MIN_POSITION_PCT  = 0.01     # below 1% of portfolio is noise — skip the trade

# Conviction bands (confidence → size multiplier)
_CONVICTION = [
    (90, 1.00),
    (80, 0.85),
    (70, 0.70),
    (65, 0.55),
    (0,  0.40),   # floor
]


def _conviction_mult(confidence: float) -> float:
    for threshold, mult in _CONVICTION:
        if confidence >= threshold:
            return mult
    return 0.40


def sizing_node(state: dict) -> dict:
    ticker       = state["ticker"]
    signal       = state.get("signal", "HOLD")
    risk_approved = state.get("risk_approved", True)

    # Only size BUY signals that passed the risk gate
    if signal != "BUY" or not risk_approved:
        return {**state,
                "position_size_pct": 0.0, "position_size_usd": 0.0,
                "max_shares": 0, "risk_dollars": 0.0,
                "scale_in": False, "size_reasoning": "No sizing — not a risk-approved BUY"}

    confidence      = float(state.get("confidence", 65))
    price           = float(state.get("current_price", 0.0))
    stop_loss       = float(state.get("stop_loss", 0.0))
    atr             = float(state.get("atr", 0.0))
    avg_volume      = float(state.get("avg_volume", 0.0))
    risk_multiplier = float(state.get("risk_multiplier", 1.0))
    entry_zone      = state.get("entry_zone", "")

    if price <= 0:
        return _zero_size(state, "Price unavailable — cannot size")

    # ── Resolve entry midpoint ─────────────────────────────────────────────────
    try:
        parts      = entry_zone.replace("$", "").split("-")
        entry_mid  = (float(parts[0].strip()) + float(parts[1].strip())) / 2
    except Exception:
        entry_mid  = price

    # ── 1. Stop-based risk sizing ──────────────────────────────────────────────
    risk_per_share = entry_mid - stop_loss
    if risk_per_share <= 0:
        risk_per_share = atr if atr > 0 else entry_mid * 0.05   # fallback: 5% or 1× ATR

    max_risk_dollars = PORTFOLIO_SIZE * MAX_RISK_PER_TRADE       # e.g. $500 on $25k account
    stop_based_shares = max_risk_dollars / risk_per_share

    # ── 2. Conviction multiplier ───────────────────────────────────────────────
    conviction = _conviction_mult(confidence)
    sized_shares = stop_based_shares * conviction

    # ── 3. Volatility / regime multiplier (from risk_agent) ───────────────────
    sized_shares *= risk_multiplier

    # ── 4. Liquidity cap — never exceed 5% of avg daily volume ────────────────
    if avg_volume > 0:
        liquidity_cap = avg_volume * LIQUIDITY_CAP_PCT
        if sized_shares > liquidity_cap:
            sized_shares = liquidity_cap

    # ── 5. Portfolio cap — never exceed 25% of total portfolio ────────────────
    max_by_portfolio = (PORTFOLIO_SIZE * MAX_POSITION_PCT) / entry_mid
    sized_shares = min(sized_shares, max_by_portfolio)

    max_shares = max(1, int(sized_shares))
    position_usd = max_shares * entry_mid
    position_pct = position_usd / PORTFOLIO_SIZE
    actual_risk  = max_shares * risk_per_share

    # ── Below minimum — skip ───────────────────────────────────────────────────
    if position_pct < MIN_POSITION_PCT:
        return _zero_size(state, f"Position too small ({position_pct*100:.2f}% < 1% minimum)")

    # ── 6. Scale-in decision ───────────────────────────────────────────────────
    atr_pct   = atr / price if price > 0 else 0
    scale_in  = atr_pct > 0.05   # scale in thirds if daily ATR > 5%

    # ── Build reasoning ────────────────────────────────────────────────────────
    reasoning = (
        f"${actual_risk:.0f} at risk ({MAX_RISK_PER_TRADE*100:.0f}% rule) × "
        f"{conviction:.0%} conviction × ×{risk_multiplier:.1f} regime = "
        f"{max_shares} shares @ ${entry_mid:.2f} "
        f"(${position_usd:,.0f} / {position_pct*100:.1f}% of portfolio)"
    )
    if scale_in:
        reasoning += " — SCALE IN thirds (high ATR)"

    print(f"📐 [SizingAgent] {ticker}: {max_shares} shares  "
          f"${position_usd:,.0f}  ({position_pct*100:.1f}%)  "
          f"risk=${actual_risk:.0f}  {'SCALE-IN' if scale_in else 'FULL-ENTRY'}")

    return {
        **state,
        "position_size_pct": round(position_pct, 4),
        "position_size_usd": round(position_usd, 2),
        "max_shares":        max_shares,
        "risk_dollars":      round(actual_risk, 2),
        "scale_in":          scale_in,
        "size_reasoning":    reasoning,
    }


def _zero_size(state: dict, reason: str) -> dict:
    print(f"📐 [SizingAgent] {state['ticker']}: {reason}")
    return {
        **state,
        "position_size_pct": 0.0,
        "position_size_usd": 0.0,
        "max_shares":        0,
        "risk_dollars":      0.0,
        "scale_in":          False,
        "size_reasoning":    reason,
        "should_alert":      False,
    }
