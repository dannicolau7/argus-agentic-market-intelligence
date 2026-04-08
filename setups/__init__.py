"""
setups/ — Setup detection and scoring for market_scanner.py.

Public API (single function replaces detect_setup + score_setup):

  detect_and_score(data, news, rsi, rvol, gap_pct) -> dict
      Scores ALL setups whose hard filters pass, returns the highest-scoring one.
      This prevents the first-match bias where gap_and_go would always shadow
      a breakout that actually scores higher on the same stock.

  Returns:
      setup_type  : str   — name of best-matching setup
      score       : int   — setup layer score
      signals     : list  — human-readable signal labels
"""

from setups.gap_and_go      import is_gap_and_go,      score_gap_and_go
from setups.breakout        import is_breakout,         score_breakout
from setups.first_pullback  import is_first_pullback,   score_first_pullback
from setups.oversold_bounce import is_oversold_bounce,  score_oversold_bounce

__all__ = ["detect_and_score"]

_SETUPS = [
    ("gap_and_go",      is_gap_and_go,      score_gap_and_go),
    ("oversold_bounce", is_oversold_bounce,  score_oversold_bounce),
    ("breakout",        is_breakout,         score_breakout),
    ("first_pullback",  is_first_pullback,   score_first_pullback),
]


def _general_score(data: dict, news: dict, rsi: float, rvol: float) -> dict:
    """Fallback for stocks that match no named setup."""
    score   = 0
    signals = []
    if rvol >= 5:
        score += 30; signals.append(f"RVOL {rvol:.1f}x")
    elif rvol >= 3:
        score += 20; signals.append(f"RVOL {rvol:.1f}x")
    elif rvol >= 2:
        score += 10; signals.append(f"RVOL {rvol:.1f}x")
    hours_old = news.get("hours_old", 999)
    if hours_old < 4:
        score += 20; signals.append(f"news {hours_old:.0f}h ago")
    elif hours_old < 12:
        score += 10; signals.append(f"news {hours_old:.0f}h ago")
    return {"setup_type": "general", "score": score, "signals": signals}


def detect_and_score(data: dict, news: dict,
                     rsi: float, rvol: float, gap_pct: float) -> dict:
    """
    Score ALL setups whose hard filter passes, return the highest-scoring one.

    A stock qualifying as both gap_and_go AND breakout gets both scored;
    the winner is whichever setup produces the stronger signal, not the
    first one in the priority list.
    """
    best_type   = None
    best_score  = -1
    best_signals = []

    for name, is_fn, score_fn in _SETUPS:
        try:
            if not is_fn(data, rsi, rvol, gap_pct):
                continue
        except Exception:
            continue
        try:
            result = score_fn(data, news, rsi, rvol, gap_pct)
        except Exception:
            continue
        if result["score"] > best_score:
            best_type    = name
            best_score   = result["score"]
            best_signals = result["signals"]

    if best_type is None:
        return _general_score(data, news, rsi, rvol)

    return {"setup_type": best_type, "score": best_score, "signals": best_signals}
