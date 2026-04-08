"""
setups/first_pullback.py — First Pullback setup.

In an established uptrend (EMA9 > EMA21 > EMA50), price pulls back
to the 9-EMA on declining volume — the first retest of a key level.

Hard filters:
  EMA9 > EMA21 > EMA50   (bullish stack)
  price within 5% of EMA9 (pulling back to, not blown through)
  rsi in [38, 55]         (mild pullback, not trend reversal)

Score focuses on:
  trend strength   — how cleanly aligned the EMAs are
  pullback quality — how close to EMA9 and with declining volume
  RSI position    — 45–55 is ideal (mild dip in uptrend)
"""

import numpy as np
from agents.tech_agent import _ema as ema_fn


def is_first_pullback(data: dict, rsi: float, rvol: float, gap_pct: float) -> bool:
    closes = data["closes"]
    if len(closes) < 51:
        return False
    e9  = ema_fn(closes, 9)[-1]
    e21 = ema_fn(closes, 21)[-1]
    e50 = ema_fn(closes, 50)[-1]
    price = float(closes[-1])
    if not (e9 > e21 > e50):
        return False
    pct_from_e9 = abs(price - e9) / e9
    return pct_from_e9 <= 0.05 and 38.0 <= rsi <= 55.0


def score_first_pullback(data: dict, news: dict, rsi: float, rvol: float, gap_pct: float) -> dict:
    """Max ~90 pts for an ideal first-pullback setup."""
    closes = data["closes"]
    price  = float(closes[-1])
    score  = 0
    signals = []

    e9  = ema_fn(closes, 9)[-1]
    e21 = ema_fn(closes, 21)[-1]
    e50 = ema_fn(closes, 50)[-1]

    # ── EMA stack strength (max 30) ───────────────────────────────────────────
    spread_9_21  = (e9 - e21) / e21 * 100 if e21 > 0 else 0
    spread_21_50 = (e21 - e50) / e50 * 100 if e50 > 0 else 0
    stack_strength = spread_9_21 + spread_21_50
    if stack_strength >= 5:
        score += 30; signals.append("strong EMA stack 🔥")
    elif stack_strength >= 2:
        score += 20; signals.append("EMA stack ✅")
    else:
        score += 10; signals.append("EMA stack")

    # ── Price proximity to EMA9 (max 30) ──────────────────────────────────────
    pct_from_e9 = abs(price - e9) / e9 * 100
    if pct_from_e9 <= 0.5:
        score += 30; signals.append("at EMA9 📍")
    elif pct_from_e9 <= 1.5:
        score += 22; signals.append(f"{pct_from_e9:.1f}% from EMA9")
    elif pct_from_e9 <= 3.0:
        score += 12; signals.append(f"{pct_from_e9:.1f}% from EMA9")

    # ── RSI pullback quality (max 20) ─────────────────────────────────────────
    if 45 <= rsi <= 52:
        score += 20; signals.append(f"RSI {rsi:.0f} ideal pullback")
    elif 38 <= rsi < 45:
        score += 12; signals.append(f"RSI {rsi:.0f}")
    elif 52 < rsi <= 55:
        score += 8;  signals.append(f"RSI {rsi:.0f}")

    # ── Declining volume on pullback (good sign) ── (max 10) ─────────────────
    volumes = data["volumes"]
    if len(volumes) >= 5:
        recent_vol = float(np.mean(volumes[-3:]))
        prior_vol  = float(np.mean(volumes[-8:-3]))
        if prior_vol > 0 and recent_vol < prior_vol * 0.75:
            score += 10; signals.append("vol declining on pullback ✅")

    return {"score": score, "signals": signals}
