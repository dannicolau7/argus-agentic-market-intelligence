"""
tech_agent.py — LangGraph node: RSI, MACD, Bollinger Bands, ATR, support/resistance,
volume spike detection (>2x 20-bar average).
"""

import numpy as np


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    k      = 2 / (period + 1)
    result = [float(arr[0])]
    for v in arr[1:]:
        result.append(float(v) * k + result[-1] * (1 - k))
    return np.array(result)


def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _calc_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> dict:
    if len(closes) < slow + sig:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
    macd_line   = _ema(closes, fast) - _ema(closes, slow)
    signal_line = _ema(macd_line, sig)
    histogram   = macd_line - signal_line
    return {
        "macd":      round(float(macd_line[-1]),   6),
        "signal":    round(float(signal_line[-1]), 6),
        "histogram": round(float(histogram[-1]),   6),
    }


def _calc_bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> dict:
    if len(closes) < period:
        last = float(closes[-1]) if len(closes) else 0.0
        return {"upper": last, "middle": last, "lower": last, "bandwidth": 0.0}
    recent = closes[-period:]
    middle = float(np.mean(recent))
    std    = float(np.std(recent, ddof=0))
    upper  = round(middle + num_std * std, 6)
    lower  = round(middle - num_std * std, 6)
    middle = round(middle, 6)
    bw     = round((upper - lower) / middle if middle else 0.0, 6)
    return {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bw}


def _calc_atr(bars: list, period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    trs = [
        max(
            bars[i]["h"] - bars[i]["l"],
            abs(bars[i]["h"] - bars[i - 1]["c"]),
            abs(bars[i]["l"] - bars[i - 1]["c"]),
        )
        for i in range(1, len(bars))
    ]
    recent = trs[-period:]
    return round(sum(recent) / len(recent), 6) if recent else 0.0


def _find_sr(bars: list, lookback: int = 20) -> dict:
    recent = bars[-min(lookback, len(bars)):]
    highs  = [b["h"] for b in recent]
    lows   = [b["l"] for b in recent]
    return {
        "support":    round(min(lows), 6),
        "resistance": round(max(highs), 6),
    }


# ── LangGraph node ─────────────────────────────────────────────────────────────

def tech_node(state: dict) -> dict:
    bars           = state.get("bars", [])
    ticker         = state["ticker"]
    current_volume = state.get("volume", 0.0)
    avg_volume     = state.get("avg_volume", 0.0)

    print(f"📊 [TechAgent] Calculating indicators for {ticker}  ({len(bars)} bars)...")

    try:
        if len(bars) < 5:
            raise ValueError(f"Not enough bars: {len(bars)}")

        closes = np.array([float(b["c"]) for b in bars], dtype=float)

        rsi      = _calc_rsi(closes)
        macd     = _calc_macd(closes)
        bollinger= _calc_bollinger(closes)
        atr      = _calc_atr(bars)
        sr       = _find_sr(bars)

        # Volume spike: current volume > 2× 20-bar average
        vol_ratio    = (current_volume / avg_volume) if avg_volume > 0 else 1.0
        volume_spike = vol_ratio >= 2.0

        price = float(closes[-1])
        bb_tag = (
            "ABOVE_BB🔴" if price > bollinger["upper"]
            else "BELOW_BB🟢" if price < bollinger["lower"]
            else "inside_BB"
        )

        print(
            f"✅ [TechAgent] RSI={rsi:.1f}  "
            f"MACD={macd['histogram']:+.6f}  "
            f"BB={bb_tag}  "
            f"ATR={atr:.4f}  "
            f"VolSpike={'🔥 YES (' + str(round(vol_ratio, 1)) + 'x)' if volume_spike else 'no'}"
        )

        return {
            **state,
            "rsi":               rsi,
            "macd":              macd,
            "bollinger":         bollinger,
            "atr":               atr,
            "support":           sr["support"],
            "resistance":        sr["resistance"],
            "volume_spike":      volume_spike,
            "volume_spike_ratio": round(vol_ratio, 2),
        }

    except Exception as e:
        print(f"❌ [TechAgent] Error: {e}")
        price = state.get("current_price", 0.0)
        return {
            **state,
            "rsi":               50.0,
            "macd":              {"macd": 0.0, "signal": 0.0, "histogram": 0.0},
            "bollinger":         {"upper": price, "middle": price, "lower": price, "bandwidth": 0.0},
            "atr":               0.0,
            "support":           0.0,
            "resistance":        0.0,
            "volume_spike":      False,
            "volume_spike_ratio": 1.0,
        }
