"""
evals/signal_evaluators.py — LangSmith evaluator functions for the 4-layer
scoring system used by stock-ai-agent.

Each evaluator takes (run, example) and returns:
  key     : metric name shown in LangSmith UI
  score   : 1 (pass) or 0 (fail)
  comment : explanation

Evaluators:
  confidence_evaluator      — confidence >= 65 on BUY/SELL signals
  setup_type_evaluator      — setup_type is a known valid value
  score_threshold_evaluator — total_score >= 68 on non-HOLD signals
  signal_format_evaluator   — signal is exactly BUY, SELL, or HOLD
"""

VALID_SETUP_TYPES = {
    "gap_and_go",
    "breakout",
    "oversold_bounce",
    "first_pullback",
    "general",
}

SCORE_THRESHOLD = 68
CONFIDENCE_THRESHOLD = 65


def _output(run) -> dict:
    """Extract the function output dict from a LangSmith run."""
    out = run.outputs or {}
    if "output" in out and isinstance(out["output"], dict):
        return out["output"]
    return out


# ── evaluators ─────────────────────────────────────────────────────────────────

def confidence_evaluator(run, example) -> dict:
    """
    BUY and SELL signals must have confidence >= 65 (the alert threshold).
    HOLD signals are exempt — they can have any confidence value.
    """
    out    = _output(run)
    signal = str(out.get("signal", "")).upper()

    if signal not in ("BUY", "SELL"):
        return {
            "key":     "confidence_threshold",
            "score":   1,
            "comment": f"n/a for {signal}",
        }

    try:
        conf = int(out.get("confidence", 0))
    except (TypeError, ValueError):
        return {
            "key":     "confidence_threshold",
            "score":   0,
            "comment": "confidence is not a valid integer",
        }

    ok = conf >= CONFIDENCE_THRESHOLD
    return {
        "key":     "confidence_threshold",
        "score":   1 if ok else 0,
        "comment": (
            f"{signal} confidence={conf} >= {CONFIDENCE_THRESHOLD} ✅"
            if ok else
            f"{signal} confidence={conf} < {CONFIDENCE_THRESHOLD} — would not fire alert ❌"
        ),
    }


def setup_type_evaluator(run, example) -> dict:
    """
    setup_type must be one of the 5 known values produced by detect_and_score().
    An unrecognised value means something upstream broke or changed.
    """
    out        = _output(run)
    setup_type = str(out.get("setup_type", "")).strip().lower()

    ok = setup_type in VALID_SETUP_TYPES
    return {
        "key":     "setup_type_valid",
        "score":   1 if ok else 0,
        "comment": (
            f"setup_type='{setup_type}' ✅"
            if ok else
            f"setup_type='{setup_type}' not in {sorted(VALID_SETUP_TYPES)} ❌"
        ),
    }


def score_threshold_evaluator(run, example) -> dict:
    """
    Non-HOLD signals must have total_score >= 68 (the scanner minimum to qualify
    for a Claude call). A BUY/SELL with score < 68 means the gate was bypassed.
    HOLD signals are exempt.
    """
    out    = _output(run)
    signal = str(out.get("signal", "")).upper()

    if signal == "HOLD":
        return {
            "key":     "score_threshold",
            "score":   1,
            "comment": "n/a for HOLD",
        }

    try:
        total_score = int(out.get("total_score", out.get("score", 0)))
    except (TypeError, ValueError):
        return {
            "key":     "score_threshold",
            "score":   0,
            "comment": "total_score missing or non-integer",
        }

    ok = total_score >= SCORE_THRESHOLD
    return {
        "key":     "score_threshold",
        "score":   1 if ok else 0,
        "comment": (
            f"{signal} total_score={total_score} >= {SCORE_THRESHOLD} ✅"
            if ok else
            f"{signal} total_score={total_score} < {SCORE_THRESHOLD} — gate breach ❌"
        ),
    }


def signal_format_evaluator(run, example) -> dict:
    """
    Signal must be exactly 'BUY', 'SELL', or 'HOLD' (uppercase, no whitespace).
    Any other value means Claude returned a malformed response.
    """
    out    = _output(run)
    signal = out.get("signal", "")

    ok = signal in ("BUY", "SELL", "HOLD")
    return {
        "key":     "signal_format",
        "score":   1 if ok else 0,
        "comment": (
            f"signal='{signal}' ✅"
            if ok else
            f"signal='{signal}' — expected BUY, SELL, or HOLD ❌"
        ),
    }


# ── registry ───────────────────────────────────────────────────────────────────

ALL_EVALUATORS = [
    confidence_evaluator,
    setup_type_evaluator,
    score_threshold_evaluator,
    signal_format_evaluator,
]
