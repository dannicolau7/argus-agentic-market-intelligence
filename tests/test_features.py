"""
tests/test_features.py — Unit tests for features/news_classifier.py
and features/relative_strength.py.

Run with:  pytest tests/test_features.py -v
"""

import numpy as np
import pytest

from features.news_classifier import classify, score, label, CATEGORY_SCORES
from features.relative_strength import compute_rs, SECTOR_ETFS


# ── news_classifier ────────────────────────────────────────────────────────────

class TestClassify:
    """classify(headline) → correct category."""

    def test_fda_approval(self):
        assert classify("FDA approves new cancer drug from Acme Bio") == "fda_approval"
        assert classify("Company receives FDA clearance for device") == "fda_approval"
        assert classify("Breakthrough designation granted by FDA") == "fda_approval"

    def test_fda_rejection(self):
        assert classify("FDA rejects application for new drug") == "fda_rejection"
        assert classify("Firm receives complete response letter from FDA") == "fda_rejection"

    def test_earnings_beat(self):
        assert classify("BZAI beats estimates on record revenue") == "earnings_beat"
        assert classify("Company tops expectations, raises guidance") == "earnings_beat"
        assert classify("Revenue surged 40% beating consensus") == "earnings_beat"

    def test_earnings_miss(self):
        assert classify("Company misses estimates, cuts outlook") == "earnings_miss"
        assert classify("Earnings miss sends stock lower") == "earnings_miss"
        assert classify("Swings to loss on weak demand") == "earnings_miss"

    def test_offering_dilution(self):
        assert classify("Company announces secondary offering of 5M shares") == "offering_dilution"
        assert classify("Files prospectus supplement for direct offering") == "offering_dilution"
        assert classify("Firm prices registered direct at discount") == "offering_dilution"

    def test_partnership(self):
        assert classify("Signs strategic partnership with Big Tech Corp") == "partnership"
        assert classify("Enters joint venture for commercialization agreement") == "partnership"

    def test_contract_win(self):
        assert classify("Wins contract with Department of Defense") == "contract_win"
        assert classify("Awarded government contract worth $200M") == "contract_win"

    def test_upgrade(self):
        assert classify("Analyst upgrades to buy with raised price target") == "upgrade"
        assert classify("Goldman initiates with buy rating") == "upgrade"

    def test_downgrade(self):
        assert classify("Barclays downgrades to sell on valuation") == "downgrade"
        assert classify("Price target cut to $5") == "downgrade"

    def test_general_fallback(self):
        assert classify("") == "general"
        assert classify("CEO discusses long-term vision at conference") == "general"
        assert classify("Company updates investor relations website") == "general"


class TestNoFalsePositives:
    """Phrase-level matching — single ambiguous words must NOT trigger a category."""

    def test_offering_not_triggered_by_cloud_offering(self):
        # "offering" alone would match if keyword matching were single-word
        assert classify("Expanding cloud offering to enterprise customers") == "general"

    def test_beat_not_triggered_by_deadbeat(self):
        assert classify("CEO is a deadbeat with a troubled past") == "general"

    def test_beat_not_triggered_by_heartbeat(self):
        assert classify("New heartbeat monitoring device approved") != "earnings_beat"

    def test_miss_not_triggered_by_dismiss(self):
        assert classify("Court dismisses lawsuit against company") == "general"

    def test_downgrade_not_triggered_by_upgrade_word(self):
        # An upgrade headline should not match downgrade
        result = classify("Analyst upgrades to buy, raises price target")
        assert result == "upgrade"

    def test_dilution_not_triggered_by_partner_offering(self):
        # "offering" in a partnership context should not trigger dilution
        assert classify("Partners with cloud offering from Amazon") == "general"


class TestDilutionBeatsPartnership:
    """Offering_dilution must rank above partnership (priority order check)."""

    def test_private_placement_not_partnership(self):
        # "private placement" is a dilution phrase; headline also mentions partnership
        headline = "Announces private placement to fund strategic partnership"
        assert classify(headline) == "offering_dilution"


class TestScoreAndLabel:
    """score() and label() return correct values for every category."""

    def test_scores_match_constants(self):
        for cat, expected in CATEGORY_SCORES.items():
            assert score(cat) == expected, f"score({cat!r}) mismatch"

    def test_unknown_category_scores_zero(self):
        assert score("made_up_category") == 0

    def test_label_returns_string(self):
        for cat in CATEGORY_SCORES:
            lbl = label(cat)
            assert isinstance(lbl, str) and len(lbl) > 0

    def test_bullish_categories_positive(self):
        for cat in ("fda_approval", "earnings_beat", "contract_win", "upgrade", "partnership"):
            assert score(cat) > 0

    def test_bearish_categories_negative(self):
        for cat in ("fda_rejection", "earnings_miss", "offering_dilution", "downgrade"):
            assert score(cat) < 0

    def test_general_is_zero(self):
        assert score("general") == 0


# ── relative_strength ──────────────────────────────────────────────────────────

def _flat(n: int, v: float = 10.0) -> np.ndarray:
    """All-equal closes — RS vs same benchmark = 0."""
    return np.full(n, v)


def _trending(n: int, start: float = 10.0, pct_per_day: float = 0.01) -> np.ndarray:
    """Prices growing at pct_per_day each bar."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + pct_per_day))
    return np.array(prices)


class TestComputeRS:
    """compute_rs() returns correct structure and plausible values."""

    def test_returns_all_keys(self):
        stock = _trending(50, pct_per_day=0.01)
        spy   = _trending(50, pct_per_day=0.005)
        result = compute_rs(stock, spy)
        for key in ("rs_1d", "rs_5d", "rs_20d", "rs_composite",
                    "rs_score", "rs_label", "rs_vs_spy"):
            assert key in result, f"missing key: {key}"

    def test_zero_rs_when_stock_matches_spy(self):
        closes = _flat(50, 10.0)
        result = compute_rs(closes, closes)
        assert result["rs_1d"]        == pytest.approx(0.0, abs=0.01)
        assert result["rs_5d"]        == pytest.approx(0.0, abs=0.01)
        assert result["rs_composite"] == pytest.approx(0.0, abs=0.01)
        assert result["rs_score"]     == 0

    def test_positive_rs_when_stock_outperforms(self):
        stock = _trending(50, pct_per_day=0.02)   # 2%/day
        spy   = _trending(50, pct_per_day=0.005)  # 0.5%/day
        result = compute_rs(stock, spy)
        assert result["rs_1d"]        > 0
        assert result["rs_composite"] > 0
        assert result["rs_score"]     > 0

    def test_negative_rs_when_stock_underperforms(self):
        stock = _trending(50, pct_per_day=0.001)  # barely moves
        spy   = _trending(50, pct_per_day=0.02)   # strong
        result = compute_rs(stock, spy)
        assert result["rs_composite"] < 0

    def test_rs_vs_spy_is_raw_spy_not_composite(self):
        """rs_vs_spy must equal _rs_vs(stock, spy)['1d'], not the composite."""
        stock = _trending(50, pct_per_day=0.02)
        spy   = _trending(50, pct_per_day=0.005)
        qqq   = _trending(50, pct_per_day=0.015)
        result = compute_rs(stock, spy, qqq)
        # rs_vs_spy is the raw 1-day SPY-only diff — composite will be different
        # because it blends SPY+QQQ; they should NOT be equal when QQQ is provided
        raw_spy_1d = float((stock[-1] / stock[-2] - 1) * 100) - float((spy[-1] / spy[-2] - 1) * 100)
        assert result["rs_vs_spy"] == pytest.approx(raw_spy_1d, abs=0.05)

    def test_qqq_composite_differs_from_spy_only(self):
        """Adding QQQ must change the composite score when QQQ differs from SPY."""
        stock = _trending(50, pct_per_day=0.02)
        spy   = _trending(50, pct_per_day=0.005)
        qqq   = _trending(50, pct_per_day=0.025)  # outperforms SPY
        only_spy = compute_rs(stock, spy)
        with_qqq  = compute_rs(stock, spy, qqq)
        assert only_spy["rs_composite"] != with_qqq["rs_composite"]

    def test_insufficient_data_returns_zeros(self):
        result = compute_rs(np.array([10.0]), np.array([10.0]))
        assert result["rs_1d"]    == 0.0
        assert result["rs_score"] == 0

    def test_empty_spy_returns_zeros(self):
        stock  = _trending(50)
        result = compute_rs(stock, np.array([]))
        assert result["rs_composite"] == 0.0
        assert result["rs_score"]     == 0


class TestSectorETFs:
    """SECTOR_ETFS map sanity checks."""

    def test_common_sectors_present(self):
        for sector in ("Technology", "Healthcare", "Energy", "Industrials"):
            assert sector in SECTOR_ETFS, f"{sector} missing from SECTOR_ETFS"

    def test_etf_values_are_uppercase_strings(self):
        for sector, etf in SECTOR_ETFS.items():
            assert isinstance(etf, str) and etf == etf.upper(), (
                f"ETF for {sector!r} is not uppercase: {etf!r}"
            )
