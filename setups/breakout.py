"""
setups/breakout.py — Breakout setup.

A stock closes at or above its 20-day high with elevated volume,
confirming institutional buying at resistance.

Hard filters:
  price >= 20-day high of prior bars   (actual breakout)
  rvol  >= 2x                          (volume confirmation)
  rsi   in [50, 67]                    (momentum zone, not overbought)

Score focuses on:
  breakout strength  — how far above resistance
  volume confirmation — how much heavier than average
  trend alignment    — EMA / MACD setup
"""

import numpy as np


def is_breakout(data: dict, rsi: float, rvol: float, gap_pct: float) -> bool:
    closes = data["closes"]
    if len(closes) < 21:
        return False
    prior_high = float(np.max(closes[-21:-1]))   # 20-day high excluding today
    price      = float(closes[-1])
    return price >= prior_high and rvol >= 2.0 and 50.0 <= rsi <= 67.0


def score_breakout(data: dict, news: dict, rsi: float, rvol: float, gap_pct: float) -> dict:
    """
    Max ~100 pts for an ideal breakout.
    Score weights volume and breakout distance.
    """
    closes  = data["closes"]
    price   = float(closes[-1])
    score   = 0
    signals = []

    # ── Breakout distance above 20-day high (max 40) ──────────────────────────
    prior_high = float(np.max(closes[-21:-1])) if len(closes) >= 21 else price
    if prior_high > 0:
        pct_above = (price - prior_high) / prior_high * 100
        if pct_above >= 5:
            score += 40; signals.append(f"+{pct_above:.1f}% above resistance 🔥")
        elif pct_above >= 2:
            score += 25; signals.append(f"+{pct_above:.1f}% breakout")
        elif pct_above >= 0.5:
            score += 15; signals.append("at resistance")

    # ── Volume confirmation (max 40) ──────────────────────────────────────────
    if rvol >= 5:
        score += 40; signals.append(f"RVOL {rvol:.1f}x 🔥")
    elif rvol >= 3:
        score += 28; signals.append(f"RVOL {rvol:.1f}x")
    else:
        score += 15; signals.append(f"RVOL {rvol:.1f}x")

    # ── RSI momentum zone (max 20) ────────────────────────────────────────────
    if 55 <= rsi <= 65:
        score += 20; signals.append(f"RSI {rsi:.0f} breakout zone")
    elif 50 <= rsi < 55:
        score += 10; signals.append(f"RSI {rsi:.0f}")

    # ── News support (max 15 — not required but helpful) ──────────────────────
    hours_old = news.get("hours_old", 999)
    if news.get("has_recent") and hours_old < 12:
        score += 15; signals.append(f"news {hours_old:.0f}h ago")

    return {"score": score, "signals": signals}
