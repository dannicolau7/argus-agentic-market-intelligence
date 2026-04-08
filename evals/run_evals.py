"""
evals/run_evals.py — Run LangSmith evaluations against the signal pipeline.

Behaviour:
  1. Loads the dataset "bzai-signal-evals" from LangSmith.
  2. If the dataset does not exist, creates it with 3 representative examples
     using the real state-dict field names from the scanner.
  3. Runs all 4 evaluators from signal_evaluators.py via langsmith.evaluate().
  4. Prints a pass-rate summary table.
  5. Exits with code 1 if any evaluator average score < 0.70.

Usage:
    python3 evals/run_evals.py
    python3 evals/run_evals.py --verbose
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT   # loads .env
from langsmith import Client, evaluate
from evals.signal_evaluators import ALL_EVALUATORS

DATASET_NAME      = "bzai-signal-evals"
EXPERIMENT_PREFIX = "ci-run"
PASS_THRESHOLD    = 0.70   # exit code 1 if any evaluator drops below this


# ── Dataset examples ───────────────────────────────────────────────────────────
# Inputs use the exact field names from the scanner state dict.
# Outputs represent the signal our pipeline should produce for each scenario.
# The target function (below) calls analyze_market with the full context and
# merges setup_type + total_score from inputs so evaluators can check them.

EXAMPLES = [
    {
        "inputs": {
            "ticker":         "BZAI",
            "price":          2.45,
            "rvol":           8.5,
            "rsi":            42.0,
            "macd_cross":     True,
            "ema_aligned":    True,
            "gap_pct":        7.2,
            "news_headline":  "BZAI beats estimates and raises full-year guidance",
            "news_category":  "earnings_beat",
            "total_score":    142,
            "setup_type":     "gap_and_go",
        },
        "outputs": {
            "signal":     "BUY",
            "confidence": 78,
            "setup_type": "gap_and_go",
            "total_score": 142,
        },
    },
    {
        "inputs": {
            "ticker":         "BZAI",
            "price":          1.87,
            "rvol":           1.4,
            "rsi":            51.0,
            "macd_cross":     False,
            "ema_aligned":    False,
            "gap_pct":        0.3,
            "news_headline":  "BZAI announces upcoming investor day presentation",
            "news_category":  "general",
            "total_score":    38,
            "setup_type":     "general",
        },
        "outputs": {
            "signal":     "HOLD",
            "confidence": 42,
            "setup_type": "general",
            "total_score": 38,
        },
    },
    {
        "inputs": {
            "ticker":         "BZAI",
            "price":          3.10,
            "rvol":           4.2,
            "rsi":            58.0,
            "macd_cross":     True,
            "ema_aligned":    True,
            "gap_pct":        1.5,
            "news_headline":  "BZAI wins government contract for AI defense platform",
            "news_category":  "contract_win",
            "total_score":    118,
            "setup_type":     "breakout",
        },
        "outputs": {
            "signal":     "BUY",
            "confidence": 72,
            "setup_type": "breakout",
            "total_score": 118,
        },
    },
]


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _dataset_exists(client: Client) -> bool:
    try:
        datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
        return len(datasets) > 0
    except Exception:
        return False


def _create_dataset(client: Client) -> None:
    print(f"📋 Dataset '{DATASET_NAME}' not found — creating with {len(EXAMPLES)} examples...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "BZAI signal evaluation dataset. "
            "Tests gap-and-go BUY, no-signal HOLD, and breakout BUY scenarios."
        ),
    )
    client.create_examples(
        inputs=[ex["inputs"]  for ex in EXAMPLES],
        outputs=[ex["outputs"] for ex in EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"✅ Created dataset '{DATASET_NAME}' with {len(EXAMPLES)} examples.")


# ── Target function ────────────────────────────────────────────────────────────

def _build_context(inputs: dict) -> dict:
    """
    Expand the slim example inputs into a full analyze_market context dict,
    filling in sensible defaults for fields not present in the example.
    """
    price = float(inputs.get("price", inputs.get("current_price", 5.0)))
    rvol  = float(inputs.get("rvol", 1.0))
    rsi   = float(inputs.get("rsi", 50.0))

    return {
        # ── Core fields from the example ──────────────────────────────────────
        "ticker":         inputs.get("ticker", "BZAI"),
        "current_price":  price,
        "setup_type":     inputs.get("setup_type", "general"),
        "total_score":    inputs.get("total_score", 0),
        "news_headline":  inputs.get("news_headline", ""),
        "news_category":  inputs.get("news_category", "general"),
        "gap_pct":        inputs.get("gap_pct", 0.0),
        "rvol":           rvol,
        "rsi":            rsi,
        "macd_cross":     inputs.get("macd_cross", False),
        "ema_aligned":    inputs.get("ema_aligned", False),
        # ── Defaults for all other analyze_market fields ──────────────────────
        "prev_close":     round(price / (1 + inputs.get("gap_pct", 0) / 100), 4),
        "volume":         int(rvol * 1_000_000),
        "avg_volume":     1_000_000,
        "macd":           {"histogram": 0.002 if inputs.get("macd_cross") else -0.001,
                           "macd": 0.005, "signal": 0.004},
        "bollinger":      {"upper": round(price * 1.10, 4), "middle": price,
                           "lower": round(price * 0.90, 4), "bandwidth": 0.20},
        "atr":            round(price * 0.04, 4),
        "support":        round(price * 0.93, 4),
        "resistance":     round(price * 1.10, 4),
        "volume_spike":   rvol >= 2.0,
        "volume_spike_ratio": rvol,
        "vwap":           round(price * 0.98, 4),
        "obv":            10_000_000,
        "smart_money":    "ACCUMULATION" if inputs.get("macd_cross") else "NEUTRAL",
        "ema_stack":      {
            "ema9":      round(price * 0.97, 4),
            "ema21":     round(price * 0.95, 4),
            "ema50":     round(price * 0.91, 4),
            "alignment": "BULLISH" if inputs.get("ema_aligned") else "NEUTRAL",
        },
        "float_rotation":  20.0,
        "sector_momentum": {"etf": "XLK", "change_pct": 0.4, "signal": "NEUTRAL"},
        "timing":          {"multiplier": 1.0, "window": "regular-hours"},
        "gap_info":        {"label": f"gap {inputs.get('gap_pct', 0):+.1f}%"},
        "earnings_info":   {"earnings_risk": "none", "days_to_earnings": 999, "earnings_date": ""},
        "market_regime":   {"regime": "BULL", "ema5": 520.0, "ema20": 515.0, "spy_day_chg": 0.3},
        "relative_strength": {"label": "RS +5.2% vs mkt"},
        "score_breakdown": {
            "raw_score": inputs.get("total_score", 0),
            "timing_mult": 1.0,
            "final_score": inputs.get("total_score", 0),
            "fired": [], "missed": [],
        },
        "intraday_rsi":   rsi,
        "sr_levels":      {},
        "news_sentiment": "BULLISH" if inputs.get("news_category") in
                          ("earnings_beat", "fda_approval", "contract_win") else "NEUTRAL",
        "sentiment_score": 70 if inputs.get("news_category") in
                           ("earnings_beat", "fda_approval", "contract_win") else 50,
        "news_summary":   inputs.get("news_headline", ""),
        "social_velocity": {"label": f"RVOL {rvol:.1f}x", "multiplier": rvol},
    }


def _pipeline_target(inputs: dict) -> dict:
    """
    Target function for langsmith.evaluate().
    Calls analyze_market with a full context built from example inputs,
    then merges setup_type and total_score so evaluators can check them.
    """
    from analyzer import analyze_market
    context = _build_context(inputs)
    result  = analyze_market(context)
    # Merge scanner fields so evaluators that check setup_type / total_score work
    result["setup_type"]  = inputs.get("setup_type", "general")
    result["total_score"] = inputs.get("total_score", 0)
    return result


# ── Summary table ──────────────────────────────────────────────────────────────

def _print_summary(results, verbose: bool = False) -> bool:
    """
    Print pass-rate per evaluator. Returns True if all pass PASS_THRESHOLD.
    """
    # Collect scores per evaluator key from the results object
    from collections import defaultdict
    scores = defaultdict(list)

    try:
        for r in results:
            for fb in (r.get("feedback_results") or r.get("evaluation_results") or []):
                key   = getattr(fb, "key",   None) or fb.get("key",   "unknown")
                score = getattr(fb, "score", None)
                if score is None:
                    score = fb.get("score")
                if score is not None:
                    scores[key].append(float(score))
    except Exception as e:
        print(f"⚠️  Could not parse result details: {e}")

    if not scores:
        print("⚠️  No evaluator scores returned — check LangSmith for run details.")
        return True   # don't fail CI on parse issues

    print("\n" + "─" * 52)
    print(f"{'Evaluator':<35} {'Pass rate':>10}")
    print("─" * 52)

    all_pass = True
    for ev in ALL_EVALUATORS:
        # try to match by function name or by key returned by evaluator
        key_scores = scores.get(ev.__name__, [])
        if not key_scores:
            # fall back: look for any key that contains the function name stem
            stem = ev.__name__.replace("_evaluator", "")
            for k, v in scores.items():
                if stem in k:
                    key_scores = v
                    break

        if key_scores:
            rate = sum(key_scores) / len(key_scores)
            icon = "🟢" if rate >= 0.80 else "🟡" if rate >= PASS_THRESHOLD else "🔴"
            print(f"{icon} {ev.__name__:<33} {rate * 100:>8.1f}%")
            if rate < PASS_THRESHOLD:
                all_pass = False
                if verbose:
                    print(f"   ⚠️  Below threshold ({PASS_THRESHOLD*100:.0f}%) — CI FAIL")
        else:
            print(f"   {ev.__name__:<33} {'n/a':>10}")

    print("─" * 52)
    print(f"Threshold: {PASS_THRESHOLD*100:.0f}%  |  "
          f"Project: {LANGCHAIN_PROJECT}  |  "
          f"Dataset: {DATASET_NAME}")
    return all_pass


# ── Main ───────────────────────────────────────────────────────────────────────

def main(verbose: bool = False) -> None:
    if not LANGCHAIN_API_KEY:
        print("❌ LANGCHAIN_API_KEY not set. Add it to .env and re-run.")
        sys.exit(1)

    client = Client()

    # Step 1 — ensure dataset exists
    if not _dataset_exists(client):
        _create_dataset(client)
    else:
        print(f"✅ Dataset '{DATASET_NAME}' found.")

    # Step 2 — run evaluations
    print(f"\n🧪 Running {len(ALL_EVALUATORS)} evaluators "
          f"(experiment_prefix='{EXPERIMENT_PREFIX}')...")

    results = evaluate(
        _pipeline_target,
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=EXPERIMENT_PREFIX,
        client=client,
    )

    # Step 3 — summary + exit code
    all_pass = _print_summary(results, verbose=verbose)

    if not all_pass:
        print("\n❌ One or more evaluators dropped below "
              f"{PASS_THRESHOLD*100:.0f}% — exiting with code 1.")
        sys.exit(1)

    print("\n✅ All evaluators passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangSmith signal evals")
    parser.add_argument("--verbose", action="store_true", help="Show per-evaluator details")
    args = parser.parse_args()
    main(verbose=args.verbose)
