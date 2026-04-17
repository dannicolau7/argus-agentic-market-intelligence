"""
tests/test_system.py — Integration tests for system-level modules.

Covers:
  - circuit_breaker: threshold logic, fail-open, caching
  - world_context: thread-safe reads/writes, build_prompt_section
  - performance_tracker: DB init, signal logging, open signals query
  - alert idempotency: same (ticker, signal, day) blocked on second call
  - task_supervisor: registration, health reporting, cancel_all
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
import threading
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── make sure project root is on sys.path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# circuit_breaker
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    def setup_method(self):
        """Clear the module-level cache before each test."""
        import circuit_breaker as cb
        cb._cache_ts     = 0.0
        cb._cache_result = {}

    def test_safe_when_vix_low_spy_normal(self):
        import circuit_breaker as cb
        with patch.object(cb, "_fetch_vix", return_value=18.0):
            result = cb.check_market(spy_day_chg=0.5)
        assert result["safe"] is True
        assert result["vix"]  == 18.0

    def test_triggered_when_vix_above_threshold(self):
        import circuit_breaker as cb
        with patch.object(cb, "_fetch_vix", return_value=30.0):
            result = cb.check_market(spy_day_chg=0.0)
        assert result["safe"] is False
        assert "VIX" in result["reason"]

    def test_triggered_when_spy_drops_sharply(self):
        import circuit_breaker as cb
        with patch.object(cb, "_fetch_vix", return_value=20.0):
            result = cb.check_market(spy_day_chg=-2.5)
        assert result["safe"] is False
        assert "SPY" in result["reason"]

    def test_fails_open_on_vix_fetch_error(self):
        """If VIX fetch throws, circuit breaker must NOT block signals."""
        import circuit_breaker as cb
        with patch.object(cb, "_fetch_vix", side_effect=Exception("network")):
            result = cb.check_market(spy_day_chg=0.0)
        assert result["safe"] is True

    def test_result_is_cached(self):
        import circuit_breaker as cb
        call_count = {"n": 0}
        def counting_fetch():
            call_count["n"] += 1
            return 18.0
        with patch.object(cb, "_fetch_vix", side_effect=counting_fetch):
            cb.check_market(spy_day_chg=0.0)
            cb.check_market(spy_day_chg=0.0)
        assert call_count["n"] == 1   # second call served from cache


# ═══════════════════════════════════════════════════════════════════════════════
# world_context
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorldContext:
    def setup_method(self):
        import world_context as wctx
        import copy
        # Reset entire context to defaults so disk-persisted data doesn't bleed in
        with wctx._lock:
            wctx._ctx = copy.deepcopy(wctx._CTX_DEFAULT)

    def test_update_and_read_macro(self):
        import world_context as wctx
        wctx.update_macro({"regime": "BULL", "bias": "BULLISH", "vix": 18.5})
        ctx = wctx.get()
        assert ctx["macro"]["regime"] == "BULL"
        assert ctx["macro"]["vix"]    == 18.5

    def test_update_and_read_geo(self):
        import world_context as wctx
        wctx.update_geo({
            "overall_bias": "BEARISH",
            "hot_sectors":  ["XLE"],
            "cold_sectors": ["XLK"],
            "risk_summary": "Energy in focus.",
            "events":       [],
        })
        ctx = wctx.get()
        assert ctx["geo"]["overall_bias"] == "BEARISH"
        assert "XLE" in ctx["geo"]["hot_sectors"]

    def test_get_returns_deep_copy(self):
        """Modifying the returned dict must not mutate internal state."""
        import world_context as wctx
        wctx.update_macro({"regime": "BULL"})
        ctx = wctx.get()
        ctx["macro"]["regime"] = "BEAR"       # mutate the copy
        assert wctx.get()["macro"]["regime"] == "BULL"  # original unchanged

    def test_prompt_section_empty_when_not_populated(self):
        import world_context as wctx
        section = wctx.build_prompt_section()
        assert section == ""   # nothing updated → empty string

    def test_prompt_section_includes_macro(self):
        import world_context as wctx
        wctx.update_macro({
            "regime": "BULL", "fed_stance": "PAUSED",
            "bias": "BULLISH", "vix": 18.0, "yield_10y": 4.2,
            "yield_2y": 3.6, "yield_curve": 60, "summary": "Test summary."
        })
        section = wctx.build_prompt_section()
        assert "BULL" in section
        assert "PAUSED" in section

    def test_thread_safe_concurrent_writes(self):
        """Multiple threads writing simultaneously must not corrupt state."""
        import world_context as wctx
        errors = []
        def writer(i):
            try:
                wctx.update_macro({"regime": f"REGIME_{i}"})
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []
        ctx = wctx.get()
        assert ctx["macro"]["regime"].startswith("REGIME_")


# ═══════════════════════════════════════════════════════════════════════════════
# performance_tracker
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceTracker:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path, monkeypatch):
        """Redirect DB to a temp file so tests don't touch the real DB."""
        import performance_tracker as pt
        monkeypatch.setattr(pt, "DB_PATH", tmp_path / "test_perf.db")
        pt.init_db()

    def _fake_state(self, ticker="AAPL", signal="BUY", price=170.0):
        import world_context as wctx
        # Ensure macro/geo/breadth have some data
        wctx.update_macro({"regime": "BULL", "bias": "BULLISH"})
        wctx.update_geo({"overall_bias": "NEUTRAL"})
        wctx.update_breadth({"health": "HEALTHY"})
        return {
            "ticker": ticker, "signal": signal, "current_price": price,
            "confidence": 70, "stop_loss": price * 0.95,
            "targets": [price * 1.05, price * 1.12, price * 1.22],
            "trade_horizon": "swing", "news_triggered": False,
            "rsi": 52.0, "volume_spike": True,
            "news_sentiment": "BULLISH", "reasoning": "Unit test signal",
        }

    def test_init_creates_tables(self, tmp_path):
        import performance_tracker as pt
        conn = sqlite3.connect(str(pt.DB_PATH))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "signals"  in tables
        assert "outcomes" in tables

    def test_record_buy_signal(self):
        import performance_tracker as pt
        state = self._fake_state()
        sig_id = pt.record_signal(state)
        assert isinstance(sig_id, int) and sig_id > 0

        conn = sqlite3.connect(str(pt.DB_PATH))
        row = conn.execute("SELECT * FROM signals WHERE id=?", (sig_id,)).fetchone()
        assert row is not None
        assert row[1] == "AAPL"   # ticker
        assert row[2] == "BUY"    # signal

    def test_record_creates_three_outcome_rows(self):
        import performance_tracker as pt
        sig_id = pt.record_signal(self._fake_state())
        conn = sqlite3.connect(str(pt.DB_PATH))
        rows = conn.execute("SELECT checkpoint FROM outcomes WHERE signal_id=?", (sig_id,)).fetchall()
        checkpoints = {r[0] for r in rows}
        assert checkpoints == {"1d", "3d", "7d"}

    def test_hold_signal_not_recorded(self):
        import performance_tracker as pt
        state = self._fake_state(signal="HOLD")
        result = pt.record_signal(state)
        assert result is None

    def test_get_open_signals_returns_recent_buys(self):
        import performance_tracker as pt
        pt.record_signal(self._fake_state(ticker="AAPL"))
        pt.record_signal(self._fake_state(ticker="MSFT"))
        open_sigs = pt.get_open_signals()
        tickers = {s["ticker"] for s in open_sigs}
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_get_stats_no_data(self):
        import performance_tracker as pt
        stats = pt.get_stats(lookback_days=30)
        assert stats["total"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# alert idempotency
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlertIdempotency:
    """
    Idempotency is now backed by SQLite via performance_tracker.
    Tests use a temporary in-memory DB to avoid touching real data.
    """
    def setup_method(self):
        """Patch performance_tracker to use a shared in-memory DB for each test."""
        import performance_tracker as pt
        import sqlite3
        self._real_get_conn = pt._get_conn
        # One shared connection so mark/check see the same data
        shared = sqlite3.connect(":memory:")
        shared.row_factory = sqlite3.Row
        shared.executescript("""
            CREATE TABLE IF NOT EXISTS fired_alerts (
                ticker TEXT NOT NULL, signal TEXT NOT NULL,
                paper INTEGER NOT NULL DEFAULT 0, fired_date TEXT NOT NULL,
                PRIMARY KEY (ticker, signal, paper, fired_date)
            );
        """)
        pt._get_conn = lambda: shared

    def teardown_method(self):
        import performance_tracker as pt
        pt._get_conn = self._real_get_conn

    def test_first_alert_not_duplicate(self):
        import performance_tracker as pt
        assert pt.is_alert_fired("AAPL", "BUY") is False

    def test_second_alert_is_duplicate(self):
        import performance_tracker as pt
        pt.mark_alert_fired("AAPL", "BUY")
        assert pt.is_alert_fired("AAPL", "BUY") is True

    def test_different_signal_not_duplicate(self):
        import performance_tracker as pt
        pt.mark_alert_fired("AAPL", "BUY")
        assert pt.is_alert_fired("AAPL", "SELL") is False

    def test_different_ticker_not_duplicate(self):
        import performance_tracker as pt
        pt.mark_alert_fired("AAPL", "BUY")
        assert pt.is_alert_fired("MSFT", "BUY") is False



# ═══════════════════════════════════════════════════════════════════════════════
# task_supervisor
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskSupervisor:
    def setup_method(self):
        import task_supervisor as sup
        sup._tasks.clear()
        sup._health.clear()

    def test_task_registers_and_reports_health(self):
        import task_supervisor as sup

        async def run():
            async def noop(): await asyncio.sleep(0)
            sup.start("test_task", noop)
            await asyncio.sleep(0.05)
            health = sup.get_health()
            assert "test_task" in health
            await sup.cancel_all()

        asyncio.run(run())

    def test_crashed_task_restarts(self):
        import task_supervisor as sup

        crash_count = {"n": 0}

        async def run():
            async def crasher():
                crash_count["n"] += 1
                if crash_count["n"] < 2:
                    raise RuntimeError("intentional crash")
                await asyncio.sleep(10)

            # Override backoff to 0 for speed
            sup.start("crasher", crasher, max_backoff=0)
            await asyncio.sleep(0.1)
            assert sup._health["crasher"]["restarts"] >= 1
            await sup.cancel_all()

        asyncio.run(run())

    def test_cancel_all_drains_cleanly(self):
        import task_supervisor as sup

        async def run():
            async def long_task():
                await asyncio.sleep(60)

            sup.start("long", long_task)
            await asyncio.sleep(0.02)
            await sup.cancel_all()   # must not raise
            assert sup._tasks == {}

        asyncio.run(run())
