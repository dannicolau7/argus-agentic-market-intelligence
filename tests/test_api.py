"""
tests/test_api.py — HTTP endpoint tests for main.py FastAPI routes.

Tests /api/state, /api/watchlist, and /api/bars without starting the full
server or triggering the lifespan (monitoring loop, scheduler, news watcher).

Run with:  pytest tests/test_api.py -v
"""

import math
import pytest
import main
from fastapi.testclient import TestClient

# Single shared client — lifespan is NOT triggered without `with` block
client = TestClient(main.app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def reset_state():
    """Replace the shared AppState with a fresh instance before each test."""
    main._app_state = main.AppState()
    main.TICKER  = "BZAI"
    main.TICKERS = ["BZAI"]


# ── /api/state ─────────────────────────────────────────────────────────────────

class TestApiState:
    def test_returns_200_on_empty_state(self):
        assert client.get("/api/state").status_code == 200

    def test_required_keys_present(self):
        data = client.get("/api/state").json()
        for key in ("state", "history", "market_open", "tickers", "ticker"):
            assert key in data, f"missing key: {key}"

    def test_reflects_injected_signal(self):
        main._app_state.ticker_states["BZAI"] = {
            "ticker":        "BZAI",
            "signal":        "BUY",
            "confidence":    78,
            "current_price": 2.45,
            "rsi":           42.0,
        }
        data = client.get("/api/state").json()
        assert data["state"]["signal"]        == "BUY"
        assert data["state"]["confidence"]    == 78
        assert data["state"]["current_price"] == pytest.approx(2.45)

    def test_ticker_query_param_routes_to_correct_state(self):
        main.TICKERS = ["BZAI", "SOUN"]
        main._app_state.ticker_states["SOUN"] = {
            "ticker":        "SOUN",
            "signal":        "HOLD",
            "current_price": 3.10,
        }
        data = client.get("/api/state?ticker=SOUN").json()
        assert data["ticker"]          == "SOUN"
        assert data["state"]["signal"] == "HOLD"

    def test_nan_in_state_sanitized_to_null(self):
        """_json_safe() must replace NaN/Inf so FastAPI can serialise the response."""
        main._app_state.ticker_states["BZAI"] = {
            "ticker":        "BZAI",
            "signal":        "BUY",
            "confidence":    72,
            "current_price": float("nan"),
            "atr":           float("inf"),
        }
        r = client.get("/api/state")
        assert r.status_code == 200          # would 500 if NaN leaked to JSON
        data = r.json()
        assert data["state"]["current_price"] is None
        assert data["state"]["atr"]           is None

    def test_history_list_returned(self):
        main._app_state.histories["BZAI"] = [
            {"timestamp": "2024-01-01T10:00:00", "price": 2.45,
             "signal": "BUY", "confidence": 78, "rsi": 42.0},
        ]
        data = client.get("/api/state").json()
        assert isinstance(data["history"], list)
        assert len(data["history"]) == 1


# ── /api/watchlist ─────────────────────────────────────────────────────────────

class TestApiWatchlist:
    def test_returns_200(self):
        assert client.get("/api/watchlist").status_code == 200

    def test_required_keys_present(self):
        data = client.get("/api/watchlist").json()
        assert "watchlist" in data
        assert "memory"    in data

    def test_all_tickers_listed(self):
        main.TICKERS = ["BZAI", "SOUN"]
        main._app_state.ticker_states["BZAI"] = {"signal": "BUY",  "confidence": 78,
                                                   "current_price": 2.45, "rsi": 42.0}
        main._app_state.ticker_states["SOUN"] = {"signal": "HOLD", "confidence": 50,
                                                   "current_price": 3.10, "rsi": 55.0}
        data = client.get("/api/watchlist").json()
        tickers = [row["ticker"] for row in data["watchlist"]]
        assert "BZAI" in tickers
        assert "SOUN" in tickers

    def test_watchlist_row_shape(self):
        main._app_state.ticker_states["BZAI"] = {
            "signal": "BUY", "confidence": 78, "current_price": 2.45, "rsi": 42.0,
        }
        row = client.get("/api/watchlist").json()["watchlist"][0]
        for key in ("ticker", "signal", "confidence", "price", "rsi"):
            assert key in row, f"missing key: {key}"

    def test_signal_memory_reflected(self):
        main._app_state.signal_memory["BZAI"] = {
            "signal":    "BUY",
            "price":     2.45,
            "stop_loss": 2.20,
            "targets":   [2.75, 3.00],
        }
        data = client.get("/api/watchlist").json()
        assert "BZAI" in data["memory"]
        assert data["memory"]["BZAI"]["signal"] == "BUY"
        assert data["memory"]["BZAI"]["stop_loss"] == pytest.approx(2.20)


# ── /api/bars ─────────────────────────────────────────────────────────────────

class TestApiBars:
    def test_returns_200(self):
        assert client.get("/api/bars").status_code == 200

    def test_empty_bars_when_no_data(self):
        data = client.get("/api/bars").json()
        assert data["bars"] == []

    def test_bar_shape_correct(self):
        main._app_state.bars_map["BZAI"] = [
            {"t": 1_700_000_000_000, "o": 2.40, "h": 2.50, "l": 2.35,
             "c": 2.45, "v": 500_000},
            {"t": 1_700_086_400_000, "o": 2.45, "h": 2.60, "l": 2.40,
             "c": 2.55, "v": 750_000},
        ]
        data = client.get("/api/bars").json()
        assert len(data["bars"]) == 2
        bar = data["bars"][0]
        for key in ("time", "open", "high", "low", "close", "volume"):
            assert key in bar, f"missing key: {key}"

    def test_timestamp_converted_to_seconds(self):
        main._app_state.bars_map["BZAI"] = [
            {"t": 1_700_000_000_000, "o": 2.40, "h": 2.50, "l": 2.35,
             "c": 2.45, "v": 500_000},
        ]
        bar = client.get("/api/bars").json()["bars"][0]
        assert bar["time"] == 1_700_000_000_000 // 1000  # ms → s

    def test_ticker_query_param(self):
        main.TICKERS = ["BZAI", "SOUN"]
        main._app_state.bars_map["SOUN"] = [
            {"t": 1_700_000_000_000, "o": 3.00, "h": 3.20,
             "l": 2.95, "c": 3.10, "v": 200_000},
        ]
        data = client.get("/api/bars?ticker=SOUN").json()
        assert data["ticker"] == "SOUN"
        assert len(data["bars"]) == 1
        assert data["bars"][0]["close"] == pytest.approx(3.10)

    def test_nan_bars_sanitized(self):
        main._app_state.bars_map["BZAI"] = [
            {"t": 1_700_000_000_000, "o": float("nan"), "h": 2.50,
             "l": 2.35, "c": 2.45, "v": 500_000},
        ]
        r = client.get("/api/bars")
        assert r.status_code == 200
        assert r.json()["bars"][0]["open"] is None
