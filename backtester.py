"""
backtester.py — Walk-forward backtest engine.

Uses rule-based scoring (same signals as market_scanner) — no Claude API calls.
Simulates BUY at the next bar's open when score ≥ 60.
Exits on: stop loss (-5%), target (+8%), or 5-bar timeout.

Usage:
    from backtester import backtest
    results = backtest(["AAPL", "TSLA", "NVDA"], period="1y")

CLI:
    python3 backtester.py AAPL TSLA NVDA --period 1y
"""

import numpy as np
import yfinance as yf
from typing import List, Dict

# ── Simulation parameters ──────────────────────────────────────────────────────
BUY_THRESHOLD = 60     # minimum score to trigger a simulated BUY
TARGET_PCT    = 0.08   # +8% take-profit
STOP_PCT      = 0.05   # -5% hard stop
MAX_HOLD      = 5      # maximum bars to hold before forced exit


# ── Inline indicators (no external imports — self-contained) ───────────────────

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    k   = 2.0 / (period + 1)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    diffs  = np.diff(closes[-(period + 1):])
    gains  = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_g  = float(np.mean(gains))
    avg_l  = float(np.mean(losses))
    if avg_g == 0 and avg_l == 0:
        return 50.0
    if avg_l == 0:
        return 100.0
    return round(100.0 - 100.0 / (1.0 + avg_g / avg_l), 2)


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    trs = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    return float(np.mean(trs[-period:]))


def _score_bar(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
               volumes: np.ndarray, opens: np.ndarray, i: int) -> int:
    """
    Score bar i using only data up to and including bar i (no lookahead).
    Returns integer score; ≥ BUY_THRESHOLD triggers a simulated BUY.
    """
    if i < 26:
        return 0

    c = closes[:i + 1]
    h = highs[:i + 1]
    l = lows[:i + 1]
    v = volumes[:i + 1]
    score = 0

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi = _rsi(c)
    if 30 <= rsi <= 50:
        score += 20   # bounce zone — best entry
    elif 50 < rsi <= 65:
        score += 10   # momentum zone
    elif rsi > 70:
        score -= 20   # overbought penalty

    # ── MACD histogram ────────────────────────────────────────────────────────
    ema12      = _ema(c, 12)
    ema26      = _ema(c, 26)
    macd_line  = ema12 - ema26
    signal_ln  = _ema(macd_line, 9)
    histogram  = float(macd_line[-1] - signal_ln[-1])
    if histogram > 0:
        score += 20
    else:
        score -= 10

    # ── EMA alignment ────────────────────────────────────────────────────────
    if len(c) >= 50:
        ema9  = float(_ema(c, 9)[-1])
        ema21 = float(_ema(c, 21)[-1])
        ema50 = float(_ema(c, 50)[-1])
        if ema9 > ema21 > ema50:
            score += 15
        elif ema9 < ema21 < ema50:
            score -= 15

    # ── Volume ratio ─────────────────────────────────────────────────────────
    hist_v  = v[-21:-1] if len(v) >= 21 else v[:-1]
    avg_vol = float(np.mean(hist_v)) if len(hist_v) > 0 else 0.0
    if avg_vol > 0:
        vol_ratio = v[-1] / avg_vol
        if vol_ratio >= 2.0:
            score += 15
        elif vol_ratio >= 1.5:
            score += 8

    # ── Gap (today open vs prev close) ────────────────────────────────────────
    if i > 0 and closes[i - 1] > 0:
        gap_pct = (opens[i] - closes[i - 1]) / closes[i - 1] * 100
        if 2.0 <= gap_pct <= 10.0:
            score += 10
        elif gap_pct > 10.0:
            score -= 10

    # ── Bollinger lower band (oversold) ───────────────────────────────────────
    if len(c) >= 20:
        sma20 = float(np.mean(c[-20:]))
        std20 = float(np.std(c[-20:]))
        if c[-1] < sma20 - 2 * std20:
            score += 15

    # ── Near 20-day support ───────────────────────────────────────────────────
    if len(l) >= 20:
        support = float(np.min(l[-20:]))
        if support > 0 and (c[-1] - support) / c[-1] <= 0.03:
            score += 10

    return score


# ── Per-ticker backtester ──────────────────────────────────────────────────────

def _backtest_ticker(ticker: str, period: str) -> Dict:
    """Walk-forward backtest for a single ticker."""
    data = yf.download(ticker, period=period, interval="1d",
                       auto_adjust=True, progress=False)
    if data is None or len(data) < 55:
        return {"ticker": ticker, "trades": [], "error": "insufficient data"}

    closes  = data["Close"].values.flatten().astype(float)
    highs   = data["High"].values.flatten().astype(float)
    lows    = data["Low"].values.flatten().astype(float)
    volumes = data["Volume"].values.flatten().astype(float)
    opens   = data["Open"].values.flatten().astype(float)
    dates   = [str(d.date()) for d in data.index]

    trades      = []
    in_position = False
    entry_price = stop_price = target_price = 0.0
    entry_date  = ""
    bars_held   = 0

    for i in range(50, len(closes) - 1):
        if not in_position:
            score = _score_bar(closes, highs, lows, volumes, opens, i)
            if score >= BUY_THRESHOLD:
                entry_price  = float(opens[i + 1])   # buy at next bar's open (no lookahead)
                if entry_price <= 0:
                    continue
                atr          = _atr(highs[:i + 1], lows[:i + 1], closes[:i + 1])
                stop_price   = max(entry_price * (1 - STOP_PCT), entry_price - atr)
                target_price = entry_price * (1 + TARGET_PCT)
                entry_date   = dates[i + 1]
                in_position  = True
                bars_held    = 0
        else:
            bars_held += 1
            exit_reason = None
            exit_price  = float(closes[i])

            if float(lows[i]) <= stop_price:
                exit_reason = "STOP_LOSS"
                exit_price  = stop_price
            elif float(highs[i]) >= target_price:
                exit_reason = "TARGET_HIT"
                exit_price  = target_price
            elif bars_held >= MAX_HOLD:
                exit_reason = "TIME_STOP"
                exit_price  = float(closes[i])

            if exit_reason:
                gain_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "ticker":        ticker,
                    "entry_date":    entry_date,
                    "exit_date":     dates[i],
                    "entry_price":   round(float(entry_price), 4),
                    "exit_price":    round(float(exit_price), 4),
                    "gain_loss_pct": round(float(gain_pct), 2),
                    "exit_reason":   exit_reason,
                })
                in_position = False

    return {"ticker": ticker, "trades": trades}


# ── Public API ─────────────────────────────────────────────────────────────────

def backtest(tickers: List[str], period: str = "1y") -> Dict:
    """
    Run walk-forward backtest across a list of tickers.

    Args:
        tickers: list of ticker symbols
        period:  yfinance period string — "3mo", "6mo", "1y", "2y"

    Returns dict with:
        win_rate, avg_gain_pct, avg_loss_pct, total_pnl_pct,
        max_drawdown_pct, total_trades, trades[], per_ticker{}
    """
    all_trades: List[Dict] = []
    per_ticker: Dict       = {}

    for ticker in tickers:
        result = _backtest_ticker(ticker.upper().strip(), period)
        per_ticker[ticker] = result
        all_trades.extend(result.get("trades", []))

    if not all_trades:
        return {
            "win_rate": None, "avg_gain_pct": None, "avg_loss_pct": None,
            "total_pnl_pct": 0.0, "max_drawdown_pct": 0.0,
            "total_trades": 0, "trades": [], "per_ticker": per_ticker,
        }

    gains  = [t["gain_loss_pct"] for t in all_trades if t["gain_loss_pct"] > 0]
    losses = [t["gain_loss_pct"] for t in all_trades if t["gain_loss_pct"] <= 0]
    total  = len(all_trades)

    # Peak-to-trough max drawdown on cumulative P&L
    cum = peak = max_dd = 0.0
    for t in sorted(all_trades, key=lambda x: x["entry_date"]):
        cum  += t["gain_loss_pct"]
        peak  = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return {
        "win_rate":         round(len(gains) / total * 100, 1),
        "avg_gain_pct":     round(float(np.mean(gains)), 2)  if gains  else None,
        "avg_loss_pct":     round(float(np.mean(losses)), 2) if losses else None,
        "total_pnl_pct":    round(sum(t["gain_loss_pct"] for t in all_trades), 2),
        "max_drawdown_pct": round(float(max_dd), 2),
        "total_trades":     total,
        "trades":           sorted(all_trades, key=lambda x: x["entry_date"], reverse=True),
        "per_ticker":       per_ticker,
    }


if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser(description="Walk-forward backtester")
    parser.add_argument("tickers", nargs="*", default=["AAPL", "TSLA", "NVDA"])
    parser.add_argument("--period", default="1y",
                        help="yfinance period: 3mo, 6mo, 1y, 2y (default: 1y)")
    args = parser.parse_args()

    print(f"\nRunning backtest: {args.tickers}  period={args.period}")
    result = backtest(args.tickers, args.period)

    print(f"\n{'='*52}")
    print(f"Total trades:     {result['total_trades']}")
    print(f"Win rate:         {result['win_rate']}%")
    print(f"Avg gain:         +{result['avg_gain_pct']}%")
    print(f"Avg loss:         {result['avg_loss_pct']}%")
    print(f"Total P&L:        {result['total_pnl_pct']:+.1f}%")
    print(f"Max drawdown:     -{result['max_drawdown_pct']:.1f}%")
    print(f"\nRecent trades:")
    for t in result["trades"][:15]:
        gl = t['gain_loss_pct']
        marker = "✅" if gl > 0 else "❌"
        print(f"  {marker} {t['entry_date']}  {t['ticker']:6s}  "
              f"${t['entry_price']:.2f} → ${t['exit_price']:.2f}  "
              f"{gl:+.1f}%  [{t['exit_reason']}]")
