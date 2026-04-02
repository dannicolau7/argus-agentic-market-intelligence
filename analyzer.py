"""
analyzer.py — calls Claude with full market context, returns structured signal.
Output: signal (BUY/SELL/HOLD), confidence (0-100), entry_zone, targets, stop_loss, reasoning
"""

import json
import anthropic
from config import ANTHROPIC_API_KEY

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def analyze_market(context: dict) -> dict:
    """
    Analyzes full market context via Claude and returns a structured trading signal.
    """
    ticker           = context.get("ticker", "UNKNOWN")
    price            = context.get("current_price", 0.0)
    prev_close       = context.get("prev_close", price)
    volume           = context.get("volume", 0)
    avg_volume       = context.get("avg_volume", 0)
    rsi              = context.get("rsi", 50.0)
    macd             = context.get("macd", {})
    bollinger        = context.get("bollinger", {})
    atr              = context.get("atr", 0.0)
    support          = context.get("support", 0.0)
    resistance       = context.get("resistance", 0.0)
    volume_spike     = context.get("volume_spike", False)
    vol_ratio        = context.get("volume_spike_ratio", 1.0)
    news_sentiment   = context.get("news_sentiment", "NEUTRAL")
    sentiment_score  = context.get("sentiment_score", 50)
    news_summary     = context.get("news_summary", "")

    day_change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

    bb_upper = bollinger.get("upper", 0)
    bb_mid   = bollinger.get("middle", 0)
    bb_lower = bollinger.get("lower", 0)
    bb_bw    = bollinger.get("bandwidth", 0)

    if bb_upper and price:
        if price > bb_upper:
            bb_pos = "ABOVE_UPPER (overbought)"
        elif price < bb_lower:
            bb_pos = "BELOW_LOWER (oversold)"
        elif price > bb_mid:
            bb_pos = "UPPER_HALF"
        else:
            bb_pos = "LOWER_HALF"
    else:
        bb_pos = "UNKNOWN"

    prompt = f"""You are an expert quantitative stock trader. Analyze {ticker} and provide a trading recommendation.

=== PRICE ACTION ===
Price:          ${price:.4f}
Prev Close:     ${prev_close:.4f}
Day Change:     {day_change_pct:+.2f}%

=== TECHNICAL INDICATORS ===
RSI(14):        {rsi:.1f}{'  ⚠️ OVERBOUGHT' if rsi > 70 else '  ⚠️ OVERSOLD' if rsi < 30 else ''}
MACD Line:      {macd.get('macd', 0):.6f}
MACD Signal:    {macd.get('signal', 0):.6f}
MACD Histogram: {macd.get('histogram', 0):+.6f}{'  (BULLISH)' if macd.get('histogram', 0) > 0 else '  (BEARISH)'}
BB Upper:       ${bb_upper:.4f}
BB Middle:      ${bb_mid:.4f}
BB Lower:       ${bb_lower:.4f}
BB Position:    {bb_pos}
BB Bandwidth:   {bb_bw:.4f}
ATR(14):        {atr:.4f}

=== KEY LEVELS ===
Support:        ${support:.4f}
Resistance:     ${resistance:.4f}

=== VOLUME ===
Current Volume: {volume:,.0f}
Avg Vol (20d):  {avg_volume:,.0f}
Volume Spike:   {'YES (' + str(round(vol_ratio, 1)) + 'x avg) 🔥' if volume_spike else 'NO'}

=== NEWS & SENTIMENT ===
Sentiment:      {news_sentiment}  (score {sentiment_score}/100)
Summary:        {news_summary}

=== INSTRUCTIONS ===
Consider all signals holistically. Look for confluence across technicals, volume, and sentiment.
A confidence score reflects how many signals align:
- 80-100: Very strong confluence
- 65-79: Clear directional bias
- 50-64: Mixed signals → output HOLD
- 0-49: Opposing signals → output HOLD

CRITICAL: If confidence < 65, you MUST set signal to "HOLD".

Respond ONLY with this exact JSON (no markdown fences, no extra text):
{{
  "signal": "BUY" or "SELL" or "HOLD",
  "confidence": <integer 0-100>,
  "entry_zone": "<price range, e.g. '$1.20 - $1.25'>",
  "targets": [<float T1>, <float T2>, <float T3>],
  "stop_loss": <float>,
  "reasoning": "<2-3 sentences citing the specific indicators that drove this decision>"
}}"""

    try:
        client   = _get_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            parts = text.split("```")
            text  = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        signal     = str(result.get("signal", "HOLD")).upper()
        confidence = max(0, min(100, int(result.get("confidence", 0))))

        return {
            "signal":     signal,
            "confidence": confidence,
            "entry_zone": str(result.get("entry_zone", f"${price:.4f}")),
            "targets":    [float(t) for t in result.get("targets", [])],
            "stop_loss":  float(result.get("stop_loss", round(price * 0.95, 4))),
            "reasoning":  str(result.get("reasoning", "")),
        }

    except json.JSONDecodeError as e:
        print(f"⚠️  [Analyzer] JSON parse error: {e}")
        return _fallback(price)
    except Exception as e:
        print(f"❌ [Analyzer] Claude API error: {e}")
        return _fallback(price)


def _fallback(price: float) -> dict:
    return {
        "signal":     "HOLD",
        "confidence": 0,
        "entry_zone": f"${price:.4f}",
        "targets":    [round(price * 1.05, 4), round(price * 1.10, 4), round(price * 1.20, 4)],
        "stop_loss":  round(price * 0.95, 4),
        "reasoning":  "Analysis unavailable — defaulting to HOLD.",
    }
