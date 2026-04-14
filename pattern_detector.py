"""
pattern_detector.py — Purely algorithmic chart pattern detection.

Detects 5 patterns from daily OHLCV arrays:
  bull_flag, double_bottom, ascending_triangle, cup_handle, breakout

No API calls, no ML — deterministic and fully testable.

Usage:
    from pattern_detector import detect_patterns
    patterns = detect_patterns(closes, highs, lows, volumes)
    # [{"pattern": "bull_flag", "confidence": 0.88, "description": "..."}]
"""

import numpy as np
from typing import List, Dict


def detect_patterns(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> List[Dict]:
    """
    Detect chart patterns from daily OHLCV arrays (minimum 20 bars required).
    Returns list of detected patterns sorted by confidence descending.
    Each entry: {"pattern": str, "confidence": float 0-1, "description": str}
    """
    closes  = np.asarray(closes,  dtype=float)
    highs   = np.asarray(highs,   dtype=float)
    lows    = np.asarray(lows,    dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    if len(closes) < 20:
        return []

    results = []
    for fn in (_bull_flag, _double_bottom, _ascending_triangle, _cup_handle, _breakout):
        hit = fn(closes, highs, lows, volumes)
        if hit:
            results.append(hit)

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ── Individual detectors ───────────────────────────────────────────────────────

def _bull_flag(closes, highs, lows, volumes) -> dict | None:
    """
    Bull flag: strong pole (≥8% gain over 3–5 bars prior to flag),
    followed by tight consolidation (last 5 bars range < 3%, declining volume).
    """
    if len(closes) < 12:
        return None

    flag_bars = closes[-5:]
    flag_vols = volumes[-5:]

    # Flag: tight consolidation — range < 3% of current price
    if closes[-1] <= 0:
        return None
    flag_range = (float(np.max(flag_bars)) - float(np.min(flag_bars))) / closes[-1]
    if flag_range >= 0.03:
        return None

    # Flag: volume declining during consolidation (first bar > last bar)
    if flag_vols[0] <= flag_vols[-1]:
        return None

    # Pole: look for ≥8% gain in 3–5 bars before the flag window
    best_pole = 0.0
    pole_bars = 0
    for n in range(3, 6):
        if len(closes) < n + 5:
            continue
        pole_start = closes[-(n + 5)]
        pole_end   = closes[-6]   # bar before flag starts
        if pole_start <= 0:
            continue
        gain = (pole_end - pole_start) / pole_start
        if gain > best_pole:
            best_pole = gain
            pole_bars = n

    if best_pole < 0.08:
        return None

    conf = min(0.95, 0.65 + best_pole + (0.03 - flag_range) * 5)
    return {
        "pattern":     "bull_flag",
        "confidence":  round(conf, 2),
        "description": f"{pole_bars}-bar pole +{best_pole*100:.1f}%, tight {flag_range*100:.1f}% flag",
    }


def _double_bottom(closes, highs, lows, volumes) -> dict | None:
    """
    Double bottom: two lows within 3% of each other in last 30 bars,
    recovery ≥5% between them, current price above both lows.
    """
    n   = min(30, len(lows))
    seg = lows[-n:]
    cls = closes[-n:]

    if len(seg) < 10:
        return None

    # Find the lowest point
    idx1 = int(np.argmin(seg))
    low1 = float(seg[idx1])
    if low1 <= 0:
        return None

    # Find second low: at least 5 bars away, within 3% of first low
    idx2 = -1
    low2 = np.inf
    for i, v in enumerate(seg):
        if abs(i - idx1) < 5:
            continue
        if abs(v - low1) / low1 <= 0.03 and v < low2:
            low2 = float(v)
            idx2 = i

    if idx2 == -1:
        return None

    # Recovery ≥5% between the two bottoms
    b_start = min(idx1, idx2) + 1
    b_end   = max(idx1, idx2)
    if b_end <= b_start:
        return None
    peak_between = float(np.max(cls[b_start:b_end]))
    trough = min(low1, low2)
    if trough <= 0:
        return None
    recovery = (peak_between - trough) / trough
    if recovery < 0.05:
        return None

    # Current price above both bottoms
    if closes[-1] <= trough:
        return None

    proximity = abs(low1 - low2) / low1
    conf = min(0.92, 0.65 + recovery + (0.03 - proximity) * 3)
    return {
        "pattern":     "double_bottom",
        "confidence":  round(conf, 2),
        "description": f"Two bottoms ~${trough:.2f}, recovery {recovery*100:.1f}%",
    }


def _ascending_triangle(closes, highs, lows, volumes) -> dict | None:
    """
    Ascending triangle: flat resistance (highs within 2%) and rising lows
    over the last 15 bars, price currently approaching resistance.
    """
    n     = min(20, len(highs))
    seg_h = highs[-n:]
    seg_l = lows[-n:]

    if len(seg_h) < 10:
        return None

    h_max = float(np.max(seg_h))
    h_min = float(np.min(seg_h))
    if h_min <= 0:
        return None

    # Flat resistance: all highs within 2% of the max high
    if (h_max - h_min) / h_min > 0.02:
        return None

    # Rising lows: positive linear slope on lows
    x     = np.arange(len(seg_l), dtype=float)
    slope = float(np.polyfit(x, seg_l, 1)[0])
    if slope <= 0:
        return None

    # Price approaching resistance (within 3%)
    if closes[-1] < h_max * 0.97:
        return None

    flatness = 1.0 - (h_max - h_min) / h_min
    conf = min(0.90, 0.55 + flatness * 0.2 + min(slope / closes[-1] * 50, 0.15))
    return {
        "pattern":     "ascending_triangle",
        "confidence":  round(conf, 2),
        "description": f"Flat resistance ~${h_max:.2f}, rising lows — breakout imminent",
    }


def _cup_handle(closes, highs, lows, volumes) -> dict | None:
    """
    Cup & handle: U-shaped base over 15–30 bars, price near prior high,
    followed by a 3–5 bar handle whose depth < 50% of the cup depth.
    """
    if len(closes) < 22:
        return None

    # Cup window: [-30:-5] (excluding the last 5 bars which form the handle)
    cup_start = max(-30, -len(closes))
    cup       = closes[cup_start:-5]

    if len(cup) < 10:
        return None

    cup_left  = float(cup[0])
    cup_right = float(cup[-1])
    cup_low   = float(np.min(cup))
    if cup_left <= 0 or cup_low <= 0:
        return None

    rim       = (cup_left + cup_right) / 2
    cup_depth = rim - cup_low

    # Cup depth ≥5%
    if cup_depth / cup_low < 0.05:
        return None

    # Left and right rims within 5% of each other
    if abs(cup_left - cup_right) / cup_left > 0.05:
        return None

    # Handle: last 5 bars, drop < 50% of cup depth
    handle      = closes[-5:]
    handle_drop = float(np.max(handle)) - float(np.min(handle))
    if handle_drop > cup_depth * 0.5:
        return None

    # Price near the rim
    if closes[-1] < rim * 0.97:
        return None

    depth_ratio = cup_depth / cup_low
    handle_ratio = handle_drop / cup_depth if cup_depth > 0 else 1.0
    conf = min(0.88, 0.55 + depth_ratio * 0.5 + (1 - handle_ratio) * 0.2)
    return {
        "pattern":     "cup_handle",
        "confidence":  round(conf, 2),
        "description": f"Cup depth {depth_ratio*100:.1f}%, handle {handle_drop/cup_low*100:.1f}% — near breakout",
    }


def _breakout(closes, highs, lows, volumes) -> dict | None:
    """
    Breakout: today's high crossed above the 20-day high on volume ≥ 1.5× avg.
    """
    if len(closes) < 22:
        return None

    prior_high = float(np.max(highs[-21:-1]))   # 20-day high, excluding today
    today_high = float(highs[-1])
    today_vol  = float(volumes[-1])
    avg_vol    = float(np.mean(volumes[-21:-1])) if len(volumes) >= 21 else 1.0

    if today_high <= prior_high:
        return None

    vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1.0
    if vol_ratio < 1.5:
        return None

    breakout_pct = (today_high - prior_high) / prior_high
    conf = min(0.95, 0.60 + min(breakout_pct * 10, 0.20) + min((vol_ratio - 1.5) * 0.10, 0.15))
    return {
        "pattern":     "breakout",
        "confidence":  round(conf, 2),
        "description": f"Breaking ${prior_high:.2f} 20-day high on {vol_ratio:.1f}× volume",
    }


if __name__ == "__main__":
    import yfinance as yf

    for ticker in ["NVDA", "AAPL", "TSLA", "AMD"]:
        data = yf.download(ticker, period="6mo", interval="1d",
                           auto_adjust=True, progress=False)
        if data is None or data.empty:
            continue
        closes  = data["Close"].values.flatten()
        highs   = data["High"].values.flatten()
        lows    = data["Low"].values.flatten()
        volumes = data["Volume"].values.flatten()
        patterns = detect_patterns(closes, highs, lows, volumes)
        print(f"\n{ticker} ({len(closes)} bars):")
        if patterns:
            for p in patterns:
                print(f"  [{p['confidence']:.0%}] {p['pattern']:20s} {p['description']}")
        else:
            print("  No patterns detected")
