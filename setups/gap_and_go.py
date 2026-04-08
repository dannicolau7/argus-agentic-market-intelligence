"""
setups/gap_and_go.py — Gap and Go setup.

A stock opens significantly higher than previous close (gap up) on news,
with heavy volume confirming institutional participation.

Hard filters:
  gap_pct >= 3%      (meaningful overnight gap)
  rvol    >= 3x      (heavy volume — not a quiet drift)
  rsi     <= 67      (not already in overbought territory)

Score focuses on:
  gap size       — larger gap = stronger catalyst
  news recency   — fresher news = more reaction potential
  rvol strength  — more volume = more conviction
"""

import numpy as np


def is_gap_and_go(data: dict, rsi: float, rvol: float, gap_pct: float) -> bool:
    return gap_pct >= 3.0 and rvol >= 3.0 and rsi <= 67.0


def score_gap_and_go(data: dict, news: dict, rsi: float, rvol: float, gap_pct: float) -> dict:
    """
    Max ~120 pts for an ideal gap-and-go setup.
    Score weights gap size and volume heavier than RSI position.
    """
    score   = 0
    signals = []

    # ── Gap size (max 60) ──────────────────────────────────────────────────────
    if gap_pct >= 15:
        score += 60; signals.append(f"gap +{gap_pct:.1f}% 🔥🔥")
    elif gap_pct >= 10:
        score += 50; signals.append(f"gap +{gap_pct:.1f}% 🔥")
    elif gap_pct >= 7:
        score += 40; signals.append(f"gap +{gap_pct:.1f}%")
    elif gap_pct >= 5:
        score += 30; signals.append(f"gap +{gap_pct:.1f}%")
    else:
        score += 20; signals.append(f"gap +{gap_pct:.1f}%")

    # ── RVOL (max 40) ─────────────────────────────────────────────────────────
    if rvol >= 10:
        score += 40; signals.append(f"RVOL {rvol:.1f}x 🔥🔥")
    elif rvol >= 5:
        score += 30; signals.append(f"RVOL {rvol:.1f}x 🔥")
    else:
        score += 20; signals.append(f"RVOL {rvol:.1f}x")

    # ── News recency (max 20) — very fresh news = active market attention ──────
    hours_old = news.get("hours_old", 999)
    if hours_old < 1:
        score += 20; signals.append("news <1h ⚡")
    elif hours_old < 4:
        score += 15; signals.append(f"news {hours_old:.0f}h ago")
    elif hours_old < 12:
        score += 8;  signals.append(f"news {hours_old:.0f}h ago")

    return {"score": score, "signals": signals}
