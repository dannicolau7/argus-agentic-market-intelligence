"""
graph.py — builds and compiles the full LangGraph pipeline.
Flow: fetch_data → analyze_news → analyze_tech → decide → alert
"""

from datetime import datetime
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.data_agent     import data_node
from agents.news_agent     import news_node
from agents.tech_agent     import tech_node
from agents.decision_agent import decision_node
from agents.alert_agent    import alert_node


# ── Shared state schema ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Config / input
    ticker:        str
    timestamp:     str
    paper_trading: bool

    # Data node
    current_price: float
    prev_close:    float
    volume:        float
    avg_volume:    float
    bars:          list
    snapshot:      dict
    raw_news:      list

    # News node
    news_sentiment:  str
    sentiment_score: float
    news_summary:    str

    # Tech node
    rsi:               float
    macd:              dict
    bollinger:         dict
    atr:               float
    support:           float
    resistance:        float
    volume_spike:      bool
    volume_spike_ratio: float

    # Decision node
    signal:       str
    confidence:   float
    entry_zone:   str
    targets:      list
    stop_loss:    float
    reasoning:    str
    should_alert: bool

    # Alert node
    alert_sent: bool


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("fetch_data",   data_node)
    g.add_node("analyze_news", news_node)
    g.add_node("analyze_tech", tech_node)
    g.add_node("decide",       decision_node)
    g.add_node("alert",        alert_node)

    g.set_entry_point("fetch_data")
    g.add_edge("fetch_data",   "analyze_news")
    g.add_edge("analyze_news", "analyze_tech")
    g.add_edge("analyze_tech", "decide")
    g.add_edge("decide",       "alert")
    g.add_edge("alert",        END)

    return g.compile()


# ── Compiled singleton ─────────────────────────────────────────────────────────

GRAPH = build_graph()


def make_initial_state(ticker: str, paper_trading: bool = False) -> AgentState:
    return AgentState(
        ticker=ticker,
        timestamp=datetime.now().isoformat(),
        paper_trading=paper_trading,
        current_price=0.0,
        prev_close=0.0,
        volume=0.0,
        avg_volume=0.0,
        bars=[],
        snapshot={},
        raw_news=[],
        news_sentiment="NEUTRAL",
        sentiment_score=50.0,
        news_summary="",
        rsi=50.0,
        macd={"macd": 0.0, "signal": 0.0, "histogram": 0.0},
        bollinger={"upper": 0.0, "middle": 0.0, "lower": 0.0, "bandwidth": 0.0},
        atr=0.0,
        support=0.0,
        resistance=0.0,
        volume_spike=False,
        volume_spike_ratio=1.0,
        signal="HOLD",
        confidence=0.0,
        entry_zone="",
        targets=[],
        stop_loss=0.0,
        reasoning="",
        should_alert=False,
        alert_sent=False,
    )
