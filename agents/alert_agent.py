"""
alert_agent.py — LangGraph node: fires SMS via Twilio + push via Pushover using alerts.py.
Respects paper_trading flag (logs only, no real alerts).
Only fires when should_alert is True.
"""

from alerts import send_signal_alert


def alert_node(state: dict) -> dict:
    signal        = state.get("signal", "HOLD")
    confidence    = state.get("confidence", 0)
    should_alert  = state.get("should_alert", False)
    paper_trading = state.get("paper_trading", False)
    ticker        = state["ticker"]
    price         = state.get("current_price", 0.0)

    if not should_alert:
        emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
        print(f"{emoji} [AlertAgent] {signal} conf={confidence}/100 — no alert needed")
        return {**state, "alert_sent": False}

    if paper_trading:
        print(
            f"📋 [AlertAgent] PAPER MODE — {signal} on {ticker} @ ${price:.4f}  "
            f"conf={confidence}/100  (alerts suppressed)"
        )
        return {**state, "alert_sent": False}

    try:
        emoji = "🟢" if signal == "BUY" else "🔴"
        print(
            f"{emoji} [AlertAgent] Firing {signal} alert  "
            f"{ticker} @ ${price:.4f}  conf={confidence}/100..."
        )
        sent = send_signal_alert(state)
        if sent:
            print("✅ [AlertAgent] Delivered via SMS + Push")
        else:
            print("⚠️  [AlertAgent] Delivery failed (check Twilio/Pushover config)")
        return {**state, "alert_sent": sent}

    except Exception as e:
        print(f"❌ [AlertAgent] Error: {e}")
        return {**state, "alert_sent": False}
