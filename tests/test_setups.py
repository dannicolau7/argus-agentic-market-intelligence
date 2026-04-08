"""
tests/test_setups.py — Unit tests for setups/ modules.

Covers:
  - is_* hard filters (True / False cases)
  - score_* scoring functions (score > 0, key signals present)
  - detect_and_score() best-match behaviour (highest score wins, not first match)

Run with:  pytest tests/test_setups.py -v
"""

import numpy as np
import pytest

from setups.gap_and_go      import is_gap_and_go,      score_gap_and_go
from setups.breakout        import is_breakout,         score_breakout
from setups.first_pullback  import is_first_pullback,   score_first_pullback
from setups.oversold_bounce import is_oversold_bounce,  score_oversold_bounce
from setups                 import detect_and_score


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _data(n: int = 30, price: float = 5.0, volumes: float = 1_000_000,
          trending: bool = False, trend_pct: float = 0.01) -> dict:
    """Minimal data dict with flat or gently trending prices."""
    if trending:
        closes = np.array([price * (1 + trend_pct) ** i for i in range(n)], dtype=float)
    else:
        closes = np.full(n, price, dtype=float)
    return {
        "closes":  closes,
        "opens":   closes * 0.99,
        "highs":   closes * 1.02,
        "lows":    closes * 0.97,
        "volumes": np.full(n, volumes, dtype=float),
        "price":   float(closes[-1]),
    }


def _news(hours_old: float = 2.0, has_recent: bool = True) -> dict:
    return {"hours_old": hours_old, "has_recent": has_recent}


# ── gap_and_go ─────────────────────────────────────────────────────────────────

class TestGapAndGo:
    def test_passes_hard_filters(self):
        assert is_gap_and_go(_data(), rsi=50, rvol=4.0, gap_pct=5.0) is True

    def test_fails_low_rvol(self):
        assert is_gap_and_go(_data(), rsi=50, rvol=2.5, gap_pct=5.0) is False

    def test_fails_small_gap(self):
        assert is_gap_and_go(_data(), rsi=50, rvol=4.0, gap_pct=2.9) is False

    def test_fails_overbought_rsi(self):
        assert is_gap_and_go(_data(), rsi=68, rvol=4.0, gap_pct=5.0) is False

    def test_score_returns_positive(self):
        result = score_gap_and_go(_data(), _news(), rsi=50, rvol=5.0, gap_pct=8.0)
        assert result["score"] > 0
        assert isinstance(result["signals"], list)

    def test_large_gap_scores_higher_than_small_gap(self):
        low  = score_gap_and_go(_data(), _news(has_recent=False), rsi=50, rvol=4.0, gap_pct=3.5)
        high = score_gap_and_go(_data(), _news(has_recent=False), rsi=50, rvol=4.0, gap_pct=16.0)
        assert high["score"] > low["score"]

    def test_high_rvol_scores_higher(self):
        low  = score_gap_and_go(_data(), _news(has_recent=False), rsi=50, rvol=3.5, gap_pct=5.0)
        high = score_gap_and_go(_data(), _news(has_recent=False), rsi=50, rvol=12.0, gap_pct=5.0)
        assert high["score"] > low["score"]

    def test_fresh_news_adds_points(self):
        stale = score_gap_and_go(_data(), _news(hours_old=999, has_recent=False), rsi=50, rvol=4.0, gap_pct=5.0)
        fresh = score_gap_and_go(_data(), _news(hours_old=0.5, has_recent=True),  rsi=50, rvol=4.0, gap_pct=5.0)
        assert fresh["score"] > stale["score"]


# ── breakout ───────────────────────────────────────────────────────────────────

class TestBreakout:
    def _breakout_data(self, n: int = 25) -> dict:
        """Price today is above the prior 20-day high."""
        closes = np.full(n, 5.0, dtype=float)
        closes[-1] = 5.5   # today breaks above 5.0 resistance
        return {
            "closes":  closes,
            "opens":   closes * 0.99,
            "highs":   closes * 1.02,
            "lows":    closes * 0.97,
            "volumes": np.full(n, 1_000_000, dtype=float),
            "price":   5.5,
        }

    def test_passes_hard_filters(self):
        assert is_breakout(self._breakout_data(), rsi=57, rvol=3.0, gap_pct=2.0) is True

    def test_fails_price_below_20d_high(self):
        # Today's close is BELOW the prior 20-day high
        n = 25
        closes = np.full(n, 5.5, dtype=float)
        closes[-1] = 5.0   # today is below recent resistance of 5.5
        d = {
            "closes":  closes,
            "opens":   closes * 0.99,
            "highs":   closes * 1.02,
            "lows":    closes * 0.97,
            "volumes": np.full(n, 1_000_000, dtype=float),
            "price":   5.0,
        }
        assert is_breakout(d, rsi=57, rvol=3.0, gap_pct=2.0) is False

    def test_fails_rsi_too_low(self):
        assert is_breakout(self._breakout_data(), rsi=45, rvol=3.0, gap_pct=2.0) is False

    def test_fails_rsi_overbought(self):
        assert is_breakout(self._breakout_data(), rsi=70, rvol=3.0, gap_pct=2.0) is False

    def test_fails_insufficient_bars(self):
        d = _data(n=10)
        assert is_breakout(d, rsi=57, rvol=3.0, gap_pct=2.0) is False

    def test_score_positive(self):
        result = score_breakout(self._breakout_data(), _news(has_recent=False), rsi=60, rvol=3.5, gap_pct=2.0)
        assert result["score"] > 0

    def test_higher_rvol_scores_more(self):
        d = self._breakout_data()
        low  = score_breakout(d, _news(has_recent=False), rsi=60, rvol=2.5, gap_pct=2.0)
        high = score_breakout(d, _news(has_recent=False), rsi=60, rvol=6.0, gap_pct=2.0)
        assert high["score"] > low["score"]


# ── first_pullback ─────────────────────────────────────────────────────────────

class TestFirstPullback:
    def _pullback_data(self, n: int = 55) -> dict:
        """
        Uptrending close array that yields EMA9 > EMA21 > EMA50,
        with the last bar pulling back slightly toward EMA9.
        """
        # Steadily rising series so EMAs stack correctly
        closes = np.array([5.0 * (1.005 ** i) for i in range(n)], dtype=float)
        # Nudge last bar down 1% so price is near EMA9 (which is near the recent average)
        closes[-1] = closes[-1] * 0.99
        return {
            "closes":  closes,
            "opens":   closes * 0.99,
            "highs":   closes * 1.01,
            "lows":    closes * 0.98,
            "volumes": np.concatenate([
                np.full(n - 3, 1_500_000),   # higher volume before pullback
                np.full(3, 800_000),          # declining volume on pullback
            ]),
            "price":   float(closes[-1]),
        }

    def test_passes_hard_filters(self):
        d = self._pullback_data()
        assert is_first_pullback(d, rsi=48, rvol=1.5, gap_pct=0.5) is True

    def test_fails_insufficient_bars(self):
        d = _data(n=30)
        assert is_first_pullback(d, rsi=48, rvol=1.5, gap_pct=0.5) is False

    def test_fails_rsi_too_high(self):
        d = self._pullback_data()
        assert is_first_pullback(d, rsi=60, rvol=1.5, gap_pct=0.5) is False

    def test_fails_rsi_too_low(self):
        d = self._pullback_data()
        assert is_first_pullback(d, rsi=35, rvol=1.5, gap_pct=0.5) is False

    def test_fails_flat_ema_stack(self):
        d = _data(n=60)  # flat closes — EMA9 ≈ EMA21 ≈ EMA50, not strictly stacked
        assert is_first_pullback(d, rsi=48, rvol=1.5, gap_pct=0.5) is False

    def test_score_positive(self):
        d = self._pullback_data()
        result = score_first_pullback(d, _news(has_recent=False), rsi=48, rvol=1.5, gap_pct=0.5)
        assert result["score"] > 0

    def test_declining_volume_signal_present(self):
        d = self._pullback_data()
        result = score_first_pullback(d, _news(has_recent=False), rsi=48, rvol=1.5, gap_pct=0.5)
        assert any("vol declining" in s for s in result["signals"])


# ── oversold_bounce ────────────────────────────────────────────────────────────

class TestOversoldBounce:
    def _bounce_data(self, n: int = 25) -> dict:
        """Price near 20-bar support (min of lows)."""
        closes = np.full(n, 3.0, dtype=float)
        lows   = np.full(n, 2.85, dtype=float)   # support ~5% below price
        closes[-1] = 2.9                           # now within 5% of support
        return {
            "closes":  closes,
            "opens":   closes * 1.01,
            "highs":   closes * 1.03,
            "lows":    lows,
            "volumes": np.full(n, 1_000_000, dtype=float),
            "price":   float(closes[-1]),
        }

    def test_passes_hard_filters(self):
        assert is_oversold_bounce(self._bounce_data(), rsi=28, rvol=3.0, gap_pct=0) is True

    def test_fails_rsi_not_oversold(self):
        assert is_oversold_bounce(self._bounce_data(), rsi=40, rvol=3.0, gap_pct=0) is False

    def test_fails_low_rvol(self):
        assert is_oversold_bounce(self._bounce_data(), rsi=28, rvol=1.5, gap_pct=0) is False

    def test_fails_price_far_from_support(self):
        d = _data(n=25, price=10.0)  # flat at 10 — lows also ~10, so support ~10, price == support
        # Make support low relative to price
        d["lows"]  = np.full(25, 5.0)
        d["price"] = 10.0
        assert is_oversold_bounce(d, rsi=28, rvol=3.0, gap_pct=0) is False

    def test_score_positive(self):
        d = self._bounce_data()
        result = score_oversold_bounce(d, _news(has_recent=False), rsi=28, rvol=4.0, gap_pct=0)
        assert result["score"] > 0

    def test_more_oversold_scores_higher(self):
        d = self._bounce_data()
        mild    = score_oversold_bounce(d, _news(has_recent=False), rsi=33, rvol=3.0, gap_pct=0)
        extreme = score_oversold_bounce(d, _news(has_recent=False), rsi=18, rvol=3.0, gap_pct=0)
        assert extreme["score"] > mild["score"]


# ── detect_and_score — best-match, not first-match ────────────────────────────

class TestDetectAndScore:
    """
    detect_and_score() must return the highest-scoring setup, not the first
    one that passes its hard filter.
    """

    def _gap_breakout_data(self, n: int = 25) -> dict:
        """
        Data that qualifies as BOTH gap_and_go AND breakout:
          - today is above 20-day high  (breakout hard filter)
          - gap_pct >= 3% and rvol >= 3x (gap_and_go hard filter)
          - RSI in 50–67 (both setups allow this)
        """
        closes = np.full(n, 5.0, dtype=float)
        closes[-1] = 5.5     # above prior-day high → breakout
        return {
            "closes":  closes,
            "opens":   np.full(n, 5.3, dtype=float),   # gap up from yesterday's close of 5.0
            "highs":   closes * 1.03,
            "lows":    closes * 0.97,
            "volumes": np.full(n, 1_000_000, dtype=float),
            "price":   5.5,
        }

    def test_returns_dict_with_required_keys(self):
        result = detect_and_score(_data(), _news(), rsi=50, rvol=2.0, gap_pct=2.0)
        assert "setup_type" in result
        assert "score"      in result
        assert "signals"    in result

    def test_no_match_returns_general(self):
        """Flat stock, low RVOL, mid RSI — no named setup should pass."""
        result = detect_and_score(_data(), _news(has_recent=False), rsi=50, rvol=1.0, gap_pct=0.5)
        assert result["setup_type"] == "general"

    def test_gap_and_go_detected(self):
        """Big gap + high RVOL + no breakout → gap_and_go."""
        d = _data(n=25)   # flat — no breakout
        result = detect_and_score(d, _news(), rsi=50, rvol=5.0, gap_pct=10.0)
        assert result["setup_type"] == "gap_and_go"
        assert result["score"] > 0

    def test_oversold_bounce_detected(self):
        d = TestOversoldBounce()._bounce_data()
        result = detect_and_score(d, _news(), rsi=28, rvol=4.0, gap_pct=0.5)
        assert result["setup_type"] == "oversold_bounce"

    def test_best_score_wins_not_first_match(self):
        """
        When a stock qualifies for multiple setups, the one with the higher
        score must win — not the first in _SETUPS priority order.

        gap_and_go is listed first in _SETUPS.  We construct a scenario where
        breakout would score higher (large breakout distance, high RVOL, RSI
        in the ideal breakout zone 55–65) so the winner must be breakout.
        """
        d = self._gap_breakout_data(n=25)
        # RSI 60: ideal for breakout (55–65 zone gets +20), fine for gap_and_go too
        # gap_pct 3.5: qualifies gap_and_go (min 3%) but only earns +20 (3–5% tier)
        # rvol 3.5: gap_and_go gets +20, breakout gets +28 (3–5x tier)
        # breakout distance: 5.5 vs prior high 5.0 = +10%, earns +40
        # Total breakout: 40 (dist) + 28 (rvol) + 20 (rsi zone) = 88
        # Total gap_and_go: 20 (gap 3–5%) + 20 (rvol 3–5x) = 40 (no news)
        result = detect_and_score(d, _news(has_recent=False), rsi=60, rvol=3.5, gap_pct=3.5)
        assert result["setup_type"] == "breakout", (
            f"Expected 'breakout' to win on score, got '{result['setup_type']}' "
            f"with score {result['score']}"
        )

    def test_score_is_non_negative_for_named_setup(self):
        d = _data(n=25)
        result = detect_and_score(d, _news(), rsi=50, rvol=5.0, gap_pct=8.0)
        if result["setup_type"] != "general":
            assert result["score"] > 0
