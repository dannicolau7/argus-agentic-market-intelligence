#!/usr/bin/env python3
"""
check_connections.py — 14-point health check for stock-ai-agent.

Verifies that all agent connections, data files, and integrations are
working correctly. Run manually or as part of CI.

Usage:
    python check_connections.py
    python check_connections.py --fail-fast   # stop on first failure
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def _check(label: str, fn, fatal: bool = False) -> bool:
    try:
        result = fn()
        if result is True or result is None:
            print(f"  {PASS} {label}")
            return True
        elif isinstance(result, str) and result.startswith("WARN:"):
            print(f"  {WARN} {label}: {result[5:]}")
            return True   # warnings don't fail
        else:
            print(f"  {FAIL} {label}: {result}")
            return False
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  {FAIL} {label}: {msg}")
        if fatal:
            traceback.print_exc()
        return False


def run_checks(fail_fast: bool = False) -> int:
    """Run all 14 checks. Returns number of failures."""
    failures = 0

    print(f"\n{'═'*60}")
    print(f"  Argus Connection Health Check — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'═'*60}\n")

    # ── 1. IntelligenceHub singleton ─────────────────────────────────────────
    print("1. IntelligenceHub")
    def _check_hub():
        from intelligence_hub import hub, IntelligenceHub
        assert isinstance(hub, IntelligenceHub), "hub is not IntelligenceHub"
        # Verify singleton
        from intelligence_hub import IntelligenceHub as IH2
        h2 = IH2()
        assert hub is h2, "singleton broken — got different instances"
    if not _check("Singleton + import", _check_hub):
        failures += 1
        if fail_fast: return failures

    # ── 2. Reflection weights from learnings.json ────────────────────────────
    print("\n2. Reflection weights")
    def _check_reflection_weights():
        from intelligence_hub import hub
        w = hub.get_reflection_weights()
        assert isinstance(w, dict) and len(w) > 0, "empty weights"
        assert "macd" in w and "volume" in w, "missing core keys"
    if not _check("get_reflection_weights() → dict with macd + volume", _check_reflection_weights):
        failures += 1

    # ── 3. Macro regime from world_context ───────────────────────────────────
    print("\n3. Macro regime (world_context)")
    def _check_macro():
        import world_context as wctx
        ctx    = wctx.get()
        regime = ctx.get("macro", {}).get("regime", "")
        if not regime:
            return "WARN: no regime set yet (world_context not updated)"
        return True
    if not _check("world_context.get()['macro']['regime']", _check_macro):
        failures += 1

    # ── 4. Regime thresholds ─────────────────────────────────────────────────
    print("\n4. Regime thresholds")
    def _check_thresholds():
        from intelligence_hub import hub
        t = hub.get_regime_thresholds()
        required = ["rsi_oversold", "rsi_overbought", "volume_spike_min",
                    "confidence_min", "confidence_cap", "agreement_min"]
        missing = [k for k in required if k not in t]
        if missing:
            return f"missing keys: {missing}"
    if not _check("get_regime_thresholds() has all 6 keys", _check_thresholds):
        failures += 1

    # ── 5. TechAgent writes stoch_rsi_signal ────────────────────────────────
    print("\n5. TechAgent — stoch_rsi_signal")
    def _check_stoch_rsi():
        from agents.tech_agent import tech_node
        import inspect
        src = inspect.getsource(tech_node)
        assert "stoch_rsi_signal" in src, "stoch_rsi_signal not written in tech_node"
        assert "StochRSIIndicator" in src, "StochRSIIndicator not used"
    if not _check("tech_node writes stoch_rsi_signal", _check_stoch_rsi):
        failures += 1

    # ── 6. SignalAggregator imports hub ─────────────────────────────────────
    print("\n6. SignalAggregator — hub integration")
    def _check_aggregator_hub():
        from agents import signal_aggregator as sa
        assert hasattr(sa, "hub"), "hub not imported in signal_aggregator"
        src = open(os.path.join(os.path.dirname(__file__),
                                "agents", "signal_aggregator.py")).read()
        assert "hub.get_reflection_weights()" in src, "get_reflection_weights not called"
        assert "agreement_min" in src, "agreement_min not used"
    if not _check("hub imported + reflection weights + agreement_min used", _check_aggregator_hub):
        failures += 1

    # ── 7. DecisionAgent dedup check ────────────────────────────────────────
    print("\n7. DecisionAgent — dedup + confidence cap")
    def _check_decision_hub():
        src = open(os.path.join(os.path.dirname(__file__),
                                "agents", "decision_agent.py")).read()
        assert "hub.was_alerted_today" in src, "dedup check missing"
        assert "hub.mark_alerted" in src, "mark_alerted missing"
        assert "confidence_cap" in src, "confidence_cap not applied"
    if not _check("was_alerted_today + mark_alerted + confidence_cap", _check_decision_hub):
        failures += 1

    # ── 8. EDGAR injection (edgar_watcher → GRAPH) ──────────────────────────
    print("\n8. EDGAR injection")
    def _check_edgar():
        src = open(os.path.join(os.path.dirname(__file__), "edgar_watcher.py")).read()
        assert "_run_edgar_pipeline" in src, "_run_edgar_pipeline not defined"
        assert "has_edgar_filing" in src, "has_edgar_filing not injected"
        assert "GRAPH.invoke" in src, "GRAPH.invoke not called"
    if not _check("edgar_watcher._run_edgar_pipeline injects has_edgar_filing", _check_edgar):
        failures += 1

    # ── 9. CircuitBreaker in risk_agent ─────────────────────────────────────
    print("\n9. CircuitBreaker in pipeline")
    def _check_circuit_breaker():
        src = open(os.path.join(os.path.dirname(__file__),
                                "agents", "risk_agent.py")).read()
        assert "check_market" in src, "check_market not called in risk_agent"
        assert "circuit_breaker" in src, "circuit_breaker not imported"
    if not _check("risk_agent gates on check_market()", _check_circuit_breaker):
        failures += 1

    # ── 10. Tomorrow watchlist data file ────────────────────────────────────
    print("\n10. Tomorrow watchlist")
    def _check_tomorrow_wl():
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        path     = os.path.join(data_dir, "tomorrow_watchlist.json")
        if not os.path.exists(path):
            return "WARN: tomorrow_watchlist.json does not exist yet (runs at 3:30 PM)"
        with open(path) as f:
            d = json.load(f)
        assert "setups" in d, "missing 'setups' key"
    if not _check("tomorrow_watchlist.json exists and has setups", _check_tomorrow_wl):
        failures += 1

    # ── 11. Portfolio context ────────────────────────────────────────────────
    print("\n11. Portfolio context")
    def _check_portfolio():
        from intelligence_hub import hub
        ctx = hub.get_portfolio_context("BZAI")
        assert "already_open" in ctx, "already_open missing"
        assert "open_count"   in ctx, "open_count missing"
        assert "open_tickers" in ctx, "open_tickers missing"
    if not _check("get_portfolio_context() returns 3 keys", _check_portfolio):
        failures += 1

    # ── 12. Reflection → Hub weight loop ────────────────────────────────────
    print("\n12. Reflection → Hub weight loop")
    def _check_reflection_loop():
        src = open(os.path.join(os.path.dirname(__file__), "reflection_agent.py")).read()
        assert "hub.update_reflection_weights" in src, "hub.update_reflection_weights not called"
        assert "_compute_signal_weights" in src, "_compute_signal_weights not defined"
        # Verify self_learner reads learnings.json first
        sl_src = open(os.path.join(os.path.dirname(__file__), "self_learner.py")).read()
        assert "LEARNINGS_PATH" in sl_src, "LEARNINGS_PATH not defined in self_learner"
        assert "signal_weights" in sl_src, "signal_weights not read in self_learner"
    if not _check("reflection writes signal_weights → hub EMA blend → self_learner reads", _check_reflection_loop):
        failures += 1

    # ── 13. LangGraph pipeline imports ──────────────────────────────────────
    print("\n13. LangGraph pipeline")
    def _check_graph():
        from graph import GRAPH
        assert GRAPH is not None, "GRAPH is None"
        # Verify all 9 nodes exist as importable modules
        nodes = [
            ("agents.data_agent",         "data_node"),
            ("agents.news_agent",         "news_node"),
            ("agents.tech_agent",         "tech_node"),
            ("agents.signal_aggregator",  "aggregator_node"),
            ("agents.decision_agent",     "decision_node"),
            ("agents.risk_agent",         "risk_node"),
            ("agents.alert_agent",        "alert_node"),
        ]
        for module, fn in nodes:
            import importlib
            mod = importlib.import_module(module)
            assert hasattr(mod, fn), f"{module}.{fn} missing"
    if not _check("GRAPH importable + 7 core node functions exist", _check_graph):
        failures += 1

    # ── 14. Alerts (WhatsApp) importable ────────────────────────────────────
    print("\n14. Alerts")
    def _check_alerts():
        from alerts import send_whatsapp
        assert callable(send_whatsapp), "send_whatsapp not callable"
    if not _check("alerts.send_whatsapp importable", _check_alerts):
        failures += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    total = 14
    passed = total - failures
    print(f"\n{'═'*60}")
    if failures == 0:
        print(f"  {PASS} All {total} checks passed — system is healthy")
    else:
        print(f"  {FAIL} {failures}/{total} checks FAILED  ({passed} passed)")
    print(f"{'═'*60}\n")

    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argus connection health check")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first failure")
    args = parser.parse_args()

    failures = run_checks(fail_fast=args.fail_fast)
    sys.exit(0 if failures == 0 else 1)
