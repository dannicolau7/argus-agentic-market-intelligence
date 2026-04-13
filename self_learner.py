"""
self_learner.py — Reads best_picks_log.csv and calculates adaptive weight
adjustments based on historical signal win rates.

How it works:
  - For each past pick, check which signals were active (MACD, RSI zone,
    VWAP, volume spike, sentiment) and whether the trade was profitable.
  - Calculate win rate per signal.
  - Return a weight multiplier per signal: high win rate → boost,
    low win rate → reduce.

Called by analyzer.py at the start of each analysis.
"""

import csv
import os
from collections import defaultdict

from langsmith import Client
from features.news_classifier import BULLISH_CATEGORIES

BEST_PICKS_LOG = "best_picks_log.csv"

# Signals tracked in best_picks_log.csv — these weights are actually learned.
# Signals NOT in this list (vwap, bollinger, support, sentiment, smart_money,
# float_rot) are always 1.0 in analyzer.py until we log them too.
DEFAULT_WEIGHTS = {
    "volume":         1.0,   # inferred from rvol >= 2.0
    "rsi_bounce":     1.0,   # inferred from rsi 30–50
    "rsi_momentum":   1.0,   # inferred from rsi 50–65
    "macd":           1.0,   # logged as macd_cross (0/1)
    "ema_stack":      1.0,   # logged as ema_cross (0/1)
    "gap":            1.0,   # logged as gap_pct > 2%
    "bullish_news":   1.0,   # logged as news_category in BULLISH_CATEGORIES
    # Additional signals tracked by _compute_score_breakdown
    "bollinger":      1.0,
    "float_rot":      1.0,
    "sentiment":      1.0,
    "smart_money":    1.0,
    "support":        1.0,
    "vwap":           1.0,
}

MIN_SAMPLES = 5   # need at least 5 picks to trust a win rate


def log_outcome_to_langsmith(run_id: str, ticker: str, actual_gain_pct: float, setup_type: str):
    try:
        client = Client()
        client.create_feedback(
            run_id=run_id,
            key="1d_return",
            score=1 if actual_gain_pct > 0 else 0,
            value=actual_gain_pct,
            comment=f"{ticker} | {setup_type} | actual 1d return: {actual_gain_pct:.2f}%"
        )
    except Exception as e:
        print(f"LangSmith feedback logging failed: {e}")


def load_win_rates() -> dict:
    """
    Reads best_picks_log.csv and returns signal win rates.
    A trade is a "win" if actual_gain_loss_pct > 0.
    Returns dict: signal_name → {"wins": int, "total": int, "win_rate": float}
    """
    if not os.path.exists(BEST_PICKS_LOG):
        return {}

    wins  = defaultdict(int)
    total = defaultdict(int)

    try:
        with open(BEST_PICKS_LOG, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gain_str = row.get("actual_gain_loss_pct", "")
                if not gain_str:
                    continue   # not yet resolved
                try:
                    gain = float(gain_str.replace("%", "").replace("+", ""))
                except ValueError:
                    continue
                won = gain > 0

                # Infer which signals were active from available columns
                try:
                    rvol = float(row.get("rvol", 0) or 0)
                    rsi  = float(row.get("rsi", 50) or 50)
                except ValueError:
                    rvol = rsi = 0.0

                try:
                    macd_cross = int(row.get("macd_cross", 0) or 0)
                    ema_cross  = int(row.get("ema_cross",  0) or 0)
                    gap_pct    = float(row.get("gap_pct",  0) or 0)
                except ValueError:
                    macd_cross = ema_cross = 0
                    gap_pct    = 0.0

                signals = {}
                if rvol >= 2.0:
                    signals["volume"] = True
                if 30 <= rsi <= 50:
                    signals["rsi_bounce"] = True
                elif 50 < rsi <= 65:
                    signals["rsi_momentum"] = True
                if macd_cross:
                    signals["macd"] = True
                if ema_cross:
                    signals["ema_stack"] = True
                if gap_pct >= 2.0:
                    signals["gap"] = True

                news_cat = row.get("news_category", "general")
                if news_cat in BULLISH_CATEGORIES:
                    signals["bullish_news"] = True

                for sig in signals:
                    total[sig] += 1
                    if won:
                        wins[sig] += 1

    except Exception as e:
        print(f"⚠️  [SelfLearner] Error reading log: {e}")
        return {}

    result = {}
    for sig in total:
        if total[sig] >= MIN_SAMPLES:
            result[sig] = {
                "wins":     wins[sig],
                "total":    total[sig],
                "win_rate": round(wins[sig] / total[sig], 3),
            }
    return result


def get_weight_adjustments() -> dict:
    """
    Returns a dict of weight multipliers for each signal.
    win_rate > 0.65 → multiplier 1.3 (boost)
    win_rate 0.50–0.65 → multiplier 1.0 (neutral)
    win_rate < 0.50 → multiplier 0.7 (reduce)
    No data → multiplier 1.0 (default)
    """
    win_rates   = load_win_rates()
    adjustments = dict(DEFAULT_WEIGHTS)  # start with all 1.0

    for signal, data in win_rates.items():
        wr = data["win_rate"]
        if wr >= 0.65:
            adjustments[signal] = 1.3
        elif wr >= 0.50:
            adjustments[signal] = 1.0
        else:
            adjustments[signal] = 0.7

    return adjustments


def get_summary() -> str:
    """Human-readable summary of learned weights for logging."""
    win_rates   = load_win_rates()
    adjustments = get_weight_adjustments()
    if not win_rates:
        return "No history yet — using default weights."

    lines = ["📚 Self-Learner weights (from history):"]
    for sig, adj in adjustments.items():
        wr_data = win_rates.get(sig)
        if wr_data:
            icon = "⬆️" if adj > 1.0 else "⬇️" if adj < 1.0 else "➡️"
            lines.append(
                f"  {icon} {sig}: ×{adj}  "
                f"({wr_data['wins']}/{wr_data['total']} wins, "
                f"{wr_data['win_rate']*100:.0f}% win rate)"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_summary())
    print("\nAdjustments:", get_weight_adjustments())
