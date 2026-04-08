"""
setups/oversold_bounce.py — Oversold Bounce setup.

A deeply oversold stock showing early signs of reversal:
volume surge signals institutional buyers stepping in.

Hard filters:
  rsi  <= 35         (genuinely oversold)
  rvol >= 2x         (buyers emerging — not quiet bleed)
  price near support (not in free fall, bouncing from a level)

Score focuses on:
  RSI extreme     — more oversold = more snap-back potential
  support quality — how close to a meaningful floor
  volume surge    — conviction of the buyers
"""

import numpy as np


def is_oversold_bounce(data: dict, rsi: float, rvol: float, gap_pct: float) -> bool:
    closes = data["closes"]
    lows   = data["lows"]
    price  = float(closes[-1])
    if rsi > 35.0 or rvol < 2.0:
        return False
    # Must be within 5% of 20-bar support
    support = float(np.min(lows[-20:])) if len(lows) >= 20 else float(np.min(lows))
    if support <= 0:
        return False
    return abs(price - support) / price <= 0.05


def score_oversold_bounce(data: dict, news: dict, rsi: float, rvol: float, gap_pct: float) -> dict:
    """Max ~110 pts for an ideal oversold-bounce setup."""
    closes  = data["closes"]
    lows    = data["lows"]
    volumes = data["volumes"]
    price   = float(closes[-1])
    score   = 0
    signals = []

    # ── RSI extreme (max 40) ──────────────────────────────────────────────────
    if rsi <= 20:
        score += 40; signals.append(f"RSI {rsi:.0f} extreme oversold 🔥")
    elif rsi <= 25:
        score += 32; signals.append(f"RSI {rsi:.0f} very oversold")
    elif rsi <= 30:
        score += 22; signals.append(f"RSI {rsi:.0f} oversold")
    else:
        score += 12; signals.append(f"RSI {rsi:.0f}")

    # ── Support proximity (max 35) ────────────────────────────────────────────
    support = float(np.min(lows[-20:])) if len(lows) >= 20 else 0
    if support > 0:
        pct_from_sup = (price - support) / price * 100
        if pct_from_sup <= 0.5:
            score += 35; signals.append("at support 📍")
        elif pct_from_sup <= 2.0:
            score += 25; signals.append(f"{pct_from_sup:.1f}% above support")
        elif pct_from_sup <= 4.0:
            score += 12; signals.append(f"near support ({pct_from_sup:.1f}%)")

    # ── Volume surge (max 35) — the key signal that buyers are in ─────────────
    if rvol >= 10:
        score += 35; signals.append(f"RVOL {rvol:.1f}x 🔥🔥")
    elif rvol >= 5:
        score += 25; signals.append(f"RVOL {rvol:.1f}x 🔥")
    elif rvol >= 3:
        score += 15; signals.append(f"RVOL {rvol:.1f}x")
    else:
        score += 8;  signals.append(f"RVOL {rvol:.1f}x")

    # ── News catalyst bonus (max 15 — not required but confirms reversal) ─────
    hours_old = news.get("hours_old", 999)
    if news.get("has_recent") and hours_old < 4:
        score += 15; signals.append(f"news <{hours_old:.0f}h 📰")

    return {"score": score, "signals": signals}
