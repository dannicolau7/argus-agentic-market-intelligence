"""
tests/test_scanner.py — Integration test for scan_best_of_day().

All external I/O is mocked so no network calls are made:
  - yfinance / Polygon  via _bulk_download_batched, fetch_benchmarks, _check_news
  - Anthropic Claude    via _claude_rank
  - CSV / file system   via update_pick_accuracy, log_best_pick, _load_alerted_today

Run with:  pytest tests/test_scanner.py -v
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import market_scanner
from market_scanner import scan_best_of_day, MIN_SCORE


# ── Fake OHLCV data ────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 60, price: float = 5.30, gap_pct: float = 8.0,
                rvol_spike: float = 5.0) -> dict:
    """
    Build a synthetic OHLCV dict that passes all five scanner gates:

    Gate 0  dollar vol ≥ $100k  — volumes*price >> threshold
    Gate 1  RVOL ≥ 2.0          — last bar volume is rvol_spike × avg
    Gate 3  RSI 28–67            — zigzag closes + moderate final move ≈ RSI 61
    Gate 4  price $0.50–$50     — price = 5.30
    Gate 5  not alerted today    — mocked to empty set
    """
    # Zigzag around 5.0 to keep RSI ~50 before the final bar
    closes  = np.array([5.0 + (0.05 if i % 2 == 0 else -0.05)
                        for i in range(n)], dtype=float)
    closes[-2] = 5.0    # prior close (denominator for gap %)
    closes[-1] = price  # today's close

    # Gap open: opens[-1] is gap_pct above prior close
    opens       = closes.copy()
    opens[-1]   = round(closes[-2] * (1 + gap_pct / 100), 4)

    highs       = closes + 0.20
    highs[-1]   = price + 0.40
    lows        = np.maximum(closes - 0.20, 0.01)
    lows[-1]    = price - 0.15

    # Volume spike only on the last bar so RVOL > 2
    base_vol    = 1_000_000.0
    volumes     = np.ones(n, dtype=float) * base_vol
    volumes[-1] = base_vol * rvol_spike

    return {
        "closes":  closes,
        "opens":   opens,
        "highs":   highs,
        "lows":    lows,
        "volumes": volumes,
        "price":   price,
    }


def _make_benchmarks(n: int = 60) -> dict:
    """SPY slightly declining so BZAI clearly outperforms (positive RS)."""
    spy = np.linspace(452.0, 448.0, n)   # down ~0.9% over period
    qqq = np.linspace(380.0, 377.0, n)   # down ~0.8%
    return {"spy": spy, "qqq": qqq}


FAKE_CLAUDE = {
    "rank":          ["BZAI"],
    "why":           "Earnings beat with 8% gap and 5x volume confirms institutional demand.",
    "expected_move": "+8-12% in 1-2 days",
    "key_risk":      "Low float — use limit orders.",
    "entry_low":     5.25,
    "entry_high":    5.40,
    "target":        5.80,
    "stop_loss":     4.90,
}


# ── Integration test ───────────────────────────────────────────────────────────

class TestScanBestOfDay:
    """
    Verifies the full scan_best_of_day() pipeline with all I/O mocked.
    Checks that a qualifying stock returns the expected winner structure.
    """

    @pytest.fixture
    def winner(self):
        """Run the pipeline once and return the winner dict."""
        ohlcv = _make_ohlcv()

        patches = {
            "market_scanner.update_pick_accuracy":    MagicMock(),
            "market_scanner._load_universe":          MagicMock(return_value=["BZAI"]),
            "market_scanner.fetch_benchmarks":        MagicMock(return_value=_make_benchmarks()),
            "market_scanner._bulk_download_batched":  MagicMock(return_value={"BZAI": ohlcv}),
            "market_scanner._load_alerted_today":     MagicMock(return_value=set()),
            "market_scanner._check_news": MagicMock(return_value={
                "has_recent": True,
                "hours_old":  2.0,
                "headline":   "BZAI beats estimates and raises full-year guidance",
            }),
            "market_scanner._get_ticker_info": MagicMock(return_value={
                "market_cap": 100_000_000.0,
                "sector":     "Technology",
            }),
            "market_scanner._claude_rank":  MagicMock(return_value=FAKE_CLAUDE),
            "market_scanner.log_best_pick": MagicMock(),
        }

        with patch.multiple("market_scanner", **{
            k.split(".", 1)[1]: v for k, v in patches.items()
        }):
            result = scan_best_of_day(paper=True)

        return result

    def test_returns_non_empty_dict(self, winner):
        assert isinstance(winner, dict)
        assert winner, "Expected a winner but got empty dict"

    def test_winner_is_bzai(self, winner):
        assert winner["ticker"] == "BZAI"

    def test_score_at_or_above_min_threshold(self, winner):
        assert winner["score"] >= MIN_SCORE, (
            f"score={winner['score']} is below MIN_SCORE={MIN_SCORE}"
        )

    def test_required_keys_present(self, winner):
        for key in ("ticker", "price", "score", "setup_type",
                    "rvol", "rsi", "news_headline", "whatsapp_msg"):
            assert key in winner, f"missing key: {key}"

    def test_whatsapp_msg_non_empty_string(self, winner):
        assert isinstance(winner["whatsapp_msg"], str)
        assert len(winner["whatsapp_msg"]) > 50

    def test_setup_type_is_valid(self, winner):
        valid = {"gap_and_go", "breakout", "first_pullback", "oversold_bounce", "general"}
        assert winner["setup_type"] in valid

    def test_rvol_reflects_spike(self, winner):
        # With 5x volume spike, RVOL should be well above the gate minimum
        assert winner["rvol"] >= 2.0

    def test_no_qualifying_stock_returns_empty(self):
        """When no stock passes gate filtering, scan returns {}."""
        with patch.multiple(
            "market_scanner",
            update_pick_accuracy=MagicMock(),
            _load_universe=MagicMock(return_value=["BZAI"]),
            fetch_benchmarks=MagicMock(return_value=_make_benchmarks()),
            _bulk_download_batched=MagicMock(return_value={}),  # empty — no data
            _load_alerted_today=MagicMock(return_value=set()),
        ):
            result = scan_best_of_day(paper=True)
        assert result == {}
