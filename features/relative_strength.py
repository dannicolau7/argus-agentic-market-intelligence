"""
features/relative_strength.py — Multi-horizon relative strength vs benchmarks.

Computes stock RS vs SPY (broad market), QQQ (growth/tech), and an optional
sector ETF over 1d, 5d, and 20d lookbacks.

Scoring uses a weighted composite:
  horizon weights: 1d 50%  |  5d 35%  |  20d 15%
  benchmark weights: SPY 60%  |  QQQ 20%  |  sector 20%

A stock with sustained outperformance across horizons and benchmarks scores
higher than a one-day spike that disappears over 5d.
"""

import numpy as np
import yfinance as yf


# ── Sector ETF map ─────────────────────────────────────────────────────────────
# Maps yfinance sector strings → SPDR sector ETF symbol.
# Used to select the right third benchmark per stock in compute_rs().

SECTOR_ETFS: dict[str, str] = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Energy":                 "XLE",
    "Financial Services":     "XLF",
    "Financials":             "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Utilities":              "XLU",
    "Basic Materials":        "XLB",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
}

# ── Benchmark fetching ─────────────────────────────────────────────────────────

def fetch_benchmarks(period: str = "2mo",
                     include_sectors: bool = False) -> dict[str, np.ndarray]:
    """
    Download SPY, QQQ, and optionally all SPDR sector ETFs.

    Call once per scan; pass the result to compute_rs() for each stock
    to avoid repeated API calls.

    Returns {"spy": array, "qqq": array, "xlk": array, ...}
    Keys are lowercase. Any download failure leaves an empty array.
    """
    symbols = [("SPY", "spy"), ("QQQ", "qqq")]
    if include_sectors:
        seen: set[str] = set()
        for etf in SECTOR_ETFS.values():
            if etf not in seen:
                symbols.append((etf, etf.lower()))
                seen.add(etf)

    out = {key: np.array([]) for _, key in symbols}
    for symbol, key in symbols:
        try:
            df = yf.download(symbol, period=period, interval="1d",
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                out[key] = df["Close"].values.astype(float)
        except Exception as e:
            print(f"⚠️  [RS] {symbol} download failed: {e}")
    return out


def fetch_spy_closes(period: str = "2mo") -> np.ndarray:
    """Backward-compatible helper — fetches SPY only."""
    return fetch_benchmarks(period)["spy"]


# ── Core computation ───────────────────────────────────────────────────────────

def _pct_change(arr: np.ndarray, n: int) -> float:
    """Return (arr[-1] / arr[-n-1] - 1) * 100, or 0.0 if not enough data."""
    if len(arr) > n and float(arr[-n - 1]) > 0:
        return float((arr[-1] / arr[-n - 1] - 1) * 100)
    return 0.0


def _rs_vs(stock_closes: np.ndarray, bench_closes: np.ndarray) -> dict[str, float]:
    """RS of stock vs one benchmark for 1d, 5d, 20d."""
    if len(bench_closes) < 2:
        return {"1d": 0.0, "5d": 0.0, "20d": 0.0}
    return {
        "1d":  round(_pct_change(stock_closes, 1)  - _pct_change(bench_closes, 1),  2),
        "5d":  round(_pct_change(stock_closes, 5)  - _pct_change(bench_closes, 5),  2),
        "20d": round(_pct_change(stock_closes, 20) - _pct_change(bench_closes, 20), 2),
    }


def compute_rs(stock_closes: np.ndarray,
               spy_closes: np.ndarray,
               qqq_closes: np.ndarray = None,
               sector_closes: np.ndarray = None) -> dict:
    """
    Compute multi-horizon, multi-benchmark relative strength.

    Parameters
    ----------
    stock_closes  : daily close array for the stock (most recent last)
    spy_closes    : daily close array for SPY
    qqq_closes    : daily close array for QQQ (optional)
    sector_closes : daily close array for the sector ETF (optional)

    Returns
    -------
    rs_1d         : 1-day composite RS (weighted across benchmarks)
    rs_5d         : 5-day composite RS
    rs_20d        : 20-day composite RS
    rs_composite  : weighted average across all horizons (main ranking signal)
    rs_score      : integer points for scanner scoring (can be negative)
    rs_label      : human-readable string for prompts / WhatsApp
    rs_vs_spy     : raw 1d RS vs SPY alone (for logging)
    """
    if len(stock_closes) < 2:
        return {"rs_1d": 0.0, "rs_5d": 0.0, "rs_20d": 0.0,
                "rs_composite": 0.0, "rs_score": 0, "rs_label": "",
                "rs_vs_spy": 0.0}

    vs_spy = _rs_vs(stock_closes, spy_closes)

    # Benchmark weights: SPY 60%, QQQ 20%, sector 20%
    # Where optional benchmarks are missing, redistribute weight to SPY
    spy_w, qqq_w, sec_w = 1.0, 0.0, 0.0
    if qqq_closes is not None and len(qqq_closes) >= 2:
        spy_w, qqq_w = 0.6, 0.2
    if sector_closes is not None and len(sector_closes) >= 2:
        spy_w = 0.6 if qqq_w else 0.8
        sec_w = 0.2

    def composite_for_horizon(n: str) -> float:
        v = vs_spy[n] * spy_w
        if qqq_w:
            v += _rs_vs(stock_closes, qqq_closes)[n] * qqq_w
        if sec_w:
            v += _rs_vs(stock_closes, sector_closes)[n] * sec_w
        return round(v, 2)

    rs_1d  = composite_for_horizon("1d")
    rs_5d  = composite_for_horizon("5d")
    rs_20d = composite_for_horizon("20d")

    # Horizon weights: 1d 50%, 5d 35%, 20d 15%
    rs_composite = round(rs_1d * 0.50 + rs_5d * 0.35 + rs_20d * 0.15, 2)

    # ── Scoring ────────────────────────────────────────────────────────────────
    # Use composite RS for scoring — penalises one-day spikes that reverse on 5d+
    score = 0
    label = ""

    if rs_composite >= 12:
        score = 30; label = f"RS +{rs_composite:.1f}% sustained 🔥"
    elif rs_composite >= 7:
        score = 22; label = f"RS +{rs_composite:.1f}% vs mkt"
    elif rs_composite >= 3:
        score = 12; label = f"RS +{rs_composite:.1f}% vs mkt"
    elif rs_composite > 0:
        score = 5;  label = f"RS +{rs_composite:.1f}% vs mkt"
    elif rs_composite <= -7:
        score = -20; label = f"RS {rs_composite:.1f}% lagging ⚠️"
    elif rs_composite <= -3:
        score = -10; label = f"RS {rs_composite:.1f}% lagging"

    return {
        "rs_1d":        rs_1d,
        "rs_5d":        rs_5d,
        "rs_20d":       rs_20d,
        "rs_composite": rs_composite,
        "rs_score":     score,
        "rs_label":     label,
        "rs_vs_spy":    vs_spy["1d"],   # raw 1d SPY comparison for logging
    }
