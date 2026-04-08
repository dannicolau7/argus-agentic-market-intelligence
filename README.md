# Stock AI Agent

> AI-powered stock monitoring system that combines three discovery engines, layered technical scoring, news-driven catalyst detection, and Claude AI to generate real-time BUY / SELL / HOLD signals with WhatsApp alerts.

**Version:** v2.0 &nbsp;|&nbsp; **Status:** Active Development

---

## 1. Project Overview

### What it does

Stock AI Agent runs three independent discovery systems simultaneously to ensure no opportunity is missed — whether you already know the ticker or not.

#### Discovery Engine 1 — Watchlist Monitor (every 5 minutes, market hours)

Continuously monitors your saved tickers every 5 minutes during market hours (9:30 AM – 4:00 PM EST). For each cycle it:

1. Fetches real-time price and OHLCV bars (yfinance + Polygon)
2. Scores social sentiment (StockTwits bull/bear ratio + StockTwits velocity)
3. Calculates RSI, MACD, EMA stack, volume, ATR, support and resistance
4. Classifies the latest news headline into a catalyst category
5. Computes relative strength vs SPY and QQQ across 1d / 5d / 20d
6. Detects the best-matching setup pattern (gap-and-go, breakout, pullback, oversold bounce)
7. Builds a 4-layer score: context + setup + execution − risk penalties
8. Passes everything to Claude AI which decides BUY / SELL / HOLD with confidence
9. Fires a WhatsApp alert via Twilio when confidence ≥ 65

#### Discovery Engine 2 — Morning Market Scanner (7:45 AM EST daily)

Scans all ~6,000 US stocks every morning before the open to find the single best opportunity of the day. No watchlist needed — it finds stocks you've never heard of.

1. Applies Gate 0: dollar volume ≥ $100k (eliminates illiquid garbage)
2. Runs the full 4-layer scoring pipeline on every survivor
3. Ranks by total score and selects the top candidate
4. Sends a pre-market WhatsApp summary with entry zone, targets, and stop
5. Logs pick to `best_picks_log.csv` and tracks forward accuracy at +1d and +3d

#### Discovery Engine 3 — 24/7 News Watcher (every 5 minutes, around the clock)

Monitors news for **all** US stocks — not just your watchlist — by polling the Polygon news API every 5 minutes, day and night. When a new article appears on any ticker:

1. Quick gate: price $0.50–$50, average volume ≥ 50k (prevents garbage)
2. 4-hour cooldown per ticker (no alert storms)
3. Classifies the headline (FDA approval, earnings beat, dilution offering, etc.)
4. If bullish catalyst: runs the full pipeline and sends a WhatsApp alert

### Who it's for

- Retail traders who want AI-assisted signal generation on small/mid cap stocks
- Developers learning LangGraph multi-agent patterns with real financial data
- Anyone who wants a 24/7 news-driven discovery system without paying for premium data

---

## 2. Tech Stack

| Layer | Tool | Purpose | Cost |
|---|---|---|---|
| Agent orchestration | LangGraph + LangChain | Multi-agent pipeline, state management | Free |
| AI reasoning | Claude API (claude-sonnet-4-6) | Signal interpretation, confidence scoring | Pay-per-use |
| Historical data | Polygon.io | Daily OHLCV bars, news, ticker details | Free tier |
| Real-time data | yfinance | Live price, intraday 5-min candles, benchmarks | Free |
| Social sentiment | StockTwits API | Bull/bear ratio + message velocity | Free, no auth |
| News classification | features/news_classifier.py | Phrase-level catalyst detection | Free (local) |
| Relative strength | features/relative_strength.py | Multi-horizon RS vs SPY + QQQ | Free (local) |
| Setup detection | setups/ (4 modules) | Gap-and-go, breakout, pullback, bounce | Free (local) |
| Self-learning | self_learner.py | Tracks signal win rates, adjusts weights | Free (local) |
| WhatsApp alerts | Twilio WhatsApp API | BUY/SELL signal delivery | Free sandbox |
| Push notifications | Pushover | iOS/Android push notifications | $5 one-time |
| Dashboard server | FastAPI + uvicorn | REST API + HTML serving | Free |
| Chart UI | TradingView Lightweight Charts | Candlestick chart, RSI panel, markers | Free |

---

## 3. Project Structure

```
stock-ai-agent/
│
├── main.py                  # Entry point — three async loops, FastAPI server, market hours
├── config.py                # Loads all API keys from .env via python-dotenv
├── graph.py                 # Builds and compiles the LangGraph pipeline
├── analyzer.py              # Calls Claude API with full market context → structured signal
├── polygon_feed.py          # Polygon.io + yfinance data layer (bars, price, news)
├── alerts.py                # Twilio WhatsApp + Pushover senders
├── logger.py                # Appends every BUY/SELL signal to signals_log.csv
├── market_scanner.py        # Morning scanner — scans ~6k stocks, picks best of day
├── news_watcher.py          # 24/7 news-driven stock discovery loop
├── scheduler.py             # Daily event scheduler (10 timed events)
├── self_learner.py          # Signal win-rate tracker — learns from past picks
├── backtest.py              # Rule-based backtester
├── watchlist_manager.py     # Persist tickers to watchlist.json between sessions
├── watchlist.json           # Saved ticker watchlist
├── quick_check.py           # Instant price + news snapshot for any ticker(s)
├── requirements.txt         # Python dependencies
├── .env                     # API keys (gitignored)
├── .env.example             # Template for .env setup
│
├── agents/
│   ├── data_agent.py        # Node 1 — fetches price, bars, volume, news
│   ├── news_agent.py        # Node 2 — StockTwits sentiment + news classification
│   ├── tech_agent.py        # Node 3 — RSI, MACD, EMA stack, ATR, volume
│   ├── decision_agent.py    # Node 4 — calls analyzer.py, gates on confidence >= 65
│   └── alert_agent.py       # Node 5 — fires WhatsApp + push, respects paper mode
│
├── features/
│   ├── __init__.py
│   ├── relative_strength.py # Multi-horizon RS vs SPY + QQQ (1d / 5d / 20d composite)
│   └── news_classifier.py   # Phrase-level headline categorisation + score adjustments
│
├── setups/
│   ├── __init__.py          # detect_and_score() — scores all matching setups, returns best
│   ├── gap_and_go.py        # Gap ≥ 3% + RVOL ≥ 3x + RSI ≤ 67
│   ├── breakout.py          # Price ≥ 20-day high + RVOL ≥ 2x + RSI 50–67
│   ├── first_pullback.py    # EMA9 > EMA21 > EMA50, price near EMA9, RSI 38–55
│   └── oversold_bounce.py   # RSI ≤ 35 + RVOL ≥ 2x + price near 20-bar support
│
└── dashboard/
    └── index.html           # Live trading terminal — candlesticks, RSI, signal log, news
```

### Agent Pipeline Flow

```
fetch_data → analyze_news → analyze_tech → decide → alert
    │              │              │            │         │
 price/bars   StockTwits     RSI/MACD/EMA  Claude AI  WhatsApp
              RS vs SPY/QQQ  setup detect   4-layer    +Pushover
              news classify  execution      score
```

---

## 4. Signal Sources & Quality

| Signal | Source | Cost | Status | Reliability |
|---|---|---|---|---|
| RSI (14) | Calculated from daily bars | Free | ✅ | ⭐⭐⭐⭐ |
| MACD (12/26/9) | Calculated from daily bars | Free | ✅ | ⭐⭐⭐⭐ |
| EMA Stack (9/21/50) | Calculated from daily bars | Free | ✅ | ⭐⭐⭐⭐ |
| Volume / RVOL | yfinance / Polygon | Free | ✅ | ⭐⭐⭐⭐⭐ |
| ATR | Calculated from OHLCV | Free | ✅ | ⭐⭐⭐⭐ |
| Support / Resistance | 20-bar high/low | Free | ✅ | ⭐⭐⭐⭐ |
| Relative Strength (1d/5d/20d) | vs SPY + QQQ via yfinance | Free | ✅ | ⭐⭐⭐⭐⭐ |
| Social sentiment | StockTwits bull/bear + velocity | Free | ✅ | ⭐⭐⭐ |
| News classification | Polygon headlines (local NLP) | Free | ✅ | ⭐⭐⭐⭐ |
| Setup pattern | gap_and_go / breakout / pullback / bounce | Free | ✅ | ⭐⭐⭐⭐ |
| Self-learned weights | CSV signal win-rate tracker | Free | ✅ | ⭐⭐⭐ |
| Options flow | Unusual Whales | $50/mo | ❌ Not yet | ⭐⭐⭐⭐⭐ |
| Dark pool prints | Unusual Whales | $50/mo | ❌ Not yet | ⭐⭐⭐⭐⭐ |
| Insider trading | SEC EDGAR | Free | ❌ Not yet | ⭐⭐⭐⭐⭐ |

---

## 5. Scoring System

Scores are built in four independent layers. There is no fixed maximum — a perfect setup can score 150+, a weak one 30.

### Layer 1 — Context Score (market fit)

| Signal | Condition | Points |
|---|---|---|
| Relative strength | RS composite ≥ +12% vs SPY+QQQ | +30 |
| Relative strength | RS composite ≥ +7% | +22 |
| Relative strength | RS composite ≥ +3% | +12 |
| Dollar volume | ≥ $5M/day | +15 |
| Dollar volume | ≥ $1M/day | +8 |
| Bullish catalyst | FDA approval, earnings beat, contract win | +10 to +25 |
| Small cap | Market cap ≤ $500M | +10 |
| Price | < $10 (momentum-friendly) | +5 |
| Relative strength | RS composite ≤ -7% (lagging market) | −20 |

### Layer 2 — Setup Score (pattern quality)

All qualifying setups are scored; the **highest-scoring** setup wins (not the first match).

| Setup | Hard Filters | Max Score |
|---|---|---|
| Gap and Go | Gap ≥ 3%, RVOL ≥ 3x, RSI ≤ 67 | ~120 |
| Oversold Bounce | RSI ≤ 35, RVOL ≥ 2x, within 5% of 20-bar support | ~110 |
| Breakout | Price ≥ 20-day high, RVOL ≥ 2x, RSI 50–67 | ~115 |
| First Pullback | EMA9 > EMA21 > EMA50, near EMA9, RSI 38–55 | ~90 |
| General | No pattern — RVOL and news bonus only | ~50 |

### Layer 3 — Execution Score (technical confirmation only)

| Signal | Condition | Points |
|---|---|---|
| MACD | Bullish crossover (hist crosses 0 from below) | +20 |
| MACD | Histogram positive | +8 |
| EMA cross | EMA9 crosses above EMA21 | +15 |
| EMA cross | EMA9 > EMA21 (aligned bullish) | +6 |
| Fib support | Price within 2% of 38.2% or 61.8% fib | +12 |
| Price support | Within 3% of 20-bar low | +10 |

### Layer 4 — Risk Penalty

| Signal | Condition | Penalty |
|---|---|---|
| Bearish catalyst | Dilution, FDA rejection, earnings miss | −15 to −25 |
| Overbought | RSI ≥ 75 | −20 |
| Overbought | RSI ≥ 68 | −10 |

### Alert threshold

```
Total score ≥ 68  AND  Claude confidence ≥ 65  →  WhatsApp alert fires
Otherwise  →  HOLD (no alert sent)
```

Score breakdown is shown in every WhatsApp alert:

```
Score: 142 (ctx=45 setup=67 exec=50 risk=-20)
```

---

## 6. News Classification

Headlines from the Polygon news API are classified into 10 categories using phrase-level matching (not single words — to prevent false positives like "cloud offering" triggering dilution).

| Category | Example Phrases | Score Adj |
|---|---|---|
| fda_approval | "FDA approves", "FDA clearance", "breakthrough designation" | +25 |
| earnings_beat | "beats estimates", "raises guidance", "record revenue" | +25 |
| contract_win | "wins contract", "awarded contract", "defense contract" | +20 |
| partnership | "strategic partnership", "joint venture", "licensing agreement" | +15 |
| upgrade | "upgrades to buy", "price target raised", "initiates buy" | +10 |
| general | No match | 0 |
| downgrade | "downgrades to sell", "price target cut" | −10 |
| earnings_miss | "misses estimates", "lowered guidance", "swings to loss" | −20 |
| offering_dilution | "secondary offering", "private placement", "prospectus supplement" | −25 |
| fda_rejection | "FDA rejects", "complete response letter", "refuse to file" | −25 |

---

## 7. API Keys Required

| Variable | Where to Get It | Cost |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Pay-per-use |
| `POLYGON_API_KEY` | [polygon.io/dashboard](https://polygon.io/dashboard) | Free tier |
| `TWILIO_ACCOUNT_SID` | [twilio.com/console](https://twilio.com/console) | Free sandbox |
| `TWILIO_AUTH_TOKEN` | [twilio.com/console](https://twilio.com/console) | Free sandbox |
| `TWILIO_FROM_NUMBER` | Twilio console → WhatsApp Sandbox | `whatsapp:+14155238886` |
| `TWILIO_TO_NUMBER` | Your WhatsApp number | e.g. `whatsapp:+40...` |
| `PUSHOVER_APP_TOKEN` | [pushover.net](https://pushover.net) → Your Apps | $5 one-time (optional) |
| `PUSHOVER_USER_KEY` | [pushover.net](https://pushover.net) → Settings | $5 one-time (optional) |

### `.env` setup

```bash
cp .env.example .env
nano .env
```

```env
ANTHROPIC_API_KEY=sk-ant-...
POLYGON_API_KEY=...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=whatsapp:+14155238886
TWILIO_TO_NUMBER=whatsapp:+40xxxxxxxxx
PUSHOVER_APP_TOKEN=...        # optional
PUSHOVER_USER_KEY=...         # optional
MONITOR_INTERVAL=300
```

---

## 8. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify setup

```bash
# Test API keys load correctly
python3 config.py

# Test Polygon + yfinance connection
python3 polygon_feed.py

# Quick price and news snapshot
python3 quick_check.py BZAI AAPL NVDA
```

### Run the agent

```bash
# Paper trading — full pipeline, no real WhatsApp alerts sent
python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN --paper

# Live monitoring — fires real WhatsApp alerts
python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN

# Monitor tickers from watchlist.json (no --ticker flag needed)
python3 main.py --paper

# Custom scan interval (every 2 minutes)
python3 main.py --ticker BZAI --interval 120

# Custom dashboard port
python3 main.py --ticker BZAI --port 8080
```

`main.py` starts all three engines automatically:
- Watchlist monitoring loop (every `MONITOR_INTERVAL` seconds, market hours)
- Daily scheduler (7:45 AM scanner, 4:30 PM report, etc.)
- 24/7 news watcher (every 5 minutes, all clocks)

### Watchlist management

```bash
python3 main.py --add NVDA        # add ticker to watchlist.json
python3 main.py --remove NVDA     # remove ticker from watchlist.json
python3 main.py --list            # print current watchlist
```

### Run the morning scanner manually

```bash
# Scan all ~6,000 US stocks and show the best pick (paper — no alert)
python3 market_scanner.py --paper

# Live run — sends WhatsApp if a strong pick is found
python3 market_scanner.py
```

### Backtesting

```bash
python3 backtest.py --ticker BZAI --days 30
python3 backtest.py --ticker AAPL --days 90 --forward 10
```

### Dashboard

Once `main.py` is running, open your browser:

```
http://localhost:8000
```

Tabs: **Signal** · **History** · **News** · **Watchlist**

---

## 9. Alert Format

Every BUY or SELL signal is sent as a WhatsApp message:

```
🟢 BUY — BZAI  [gap_and_go]
Price:      $1.7900
Entry Zone: $1.75 – $1.85
Targets:    $1.95 / $2.10 / $2.35
Stop Loss:  $1.62
RSI:        34.2  |  RVOL: 4.8x
Score: 142 (ctx=45 setup=67 exec=50 risk=-20)
Signals: gap +8.3%, RVOL 4.8x 🔥, RS +9.2% vs mkt, earnings beat 🟢

RSI oversold with gap-and-go setup on earnings beat catalyst.
RVOL 4.8x confirms institutional entry near the $1.65 support.
```

```
🔴 SELL — BZAI  [breakout_fail]
Price:      $1.9900
Entry Zone: $1.95 – $2.00
Targets:    $1.78 / $1.65 / $1.50
Stop Loss:  $2.14
RSI:        74.1  |  RVOL: 2.1x
Score: 78 (ctx=20 setup=38 exec=25 risk=-5)
Signals: RSI 74 overbought, share offering 🔴

RSI overbought on dilution catalyst. MACD bearish crossover with
price rejecting at resistance. Stop above recent high $2.14.
```

HOLD signals are **never** sent — alerts only fire when confidence ≥ 65.

---

## 10. Daily Scheduler

The scheduler runs 10 timed events each trading day:

| Time (EST) | Event |
|---|---|
| 7:45 AM | Morning market scanner — best-of-day pick sent via WhatsApp |
| 8:00 AM | Pre-market news scan — any high-conviction catalyst alerts |
| 9:00 AM | Market open prep — watchlist summary |
| 9:30 AM | Market open — monitoring loop starts |
| 12:00 PM | Midday check — any new setups emerging |
| 2:00 PM | Power hour prep |
| 3:30 PM | Final 30 min alert if strong signal |
| 4:00 PM | Market close — monitoring loop pauses |
| 4:30 PM | Daily performance report via WhatsApp |
| 5:00 PM | Pick accuracy update — 1d forward returns logged |

---

## 11. Self-Learner

`self_learner.py` reads `best_picks_log.csv` after the market closes and updates signal weights based on which signals were present in winning vs losing picks.

**Tracked signals:**

| Signal | CSV column | Notes |
|---|---|---|
| Volume spike | `rvol` | RVOL ≥ 3x tagged as volume signal |
| RSI bounce | `rsi` | RSI ≤ 35 at entry |
| RSI momentum | `rsi` | RSI 40–55 in trend |
| MACD cross | `macd_cross` | 1 = bullish crossover at entry |
| EMA stack | `ema_cross` | 1 = EMA9 > EMA21 > EMA50 |
| Gap | `gap_pct` | Gap ≥ 3% |
| Bullish news | `news_category` | Any BULLISH_CATEGORIES hit |

Weights are loaded at scanner startup and adjust the scoring multipliers on each signal over time.

---

## 12. Data Sources Explained

### Polygon.io (free tier)

- **Provides:** Adjusted daily OHLCV bars (up to 2 years), previous-day close, company details, news headlines
- **Rate limit:** 5 API calls/minute
- **Used for:** Historical bars (RSI/MACD/EMA), news feed, ticker metadata, all-stock news polling (news watcher)

### yfinance (free, no key)

- **Provides:** Real-time last price, intraday bars (1m/5m/15m/1h), SPY/QQQ benchmark closes
- **Used for:** Live price, dashboard candles, RS computation, quick gate in news watcher
- **Limitation:** Unofficial API, occasional rate-limiting; Polygon is the fallback for price

### StockTwits (free, no key)

- **Endpoint:** `https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json`
- **Provides:** Bull/bear ratio (% of messages tagged bullish vs bearish), message count velocity
- **Scoring:** Bull ratio ≥ 65% → +15 pts; velocity ≥ 2x → additional +10 pts
- **Limitation:** No message text — ratio and count only; thin coverage on very small caps

---

## 13. Roadmap

### v1.0 — Shipped ✅

- [x] Multi-agent LangGraph pipeline (data → news → tech → decision → alert)
- [x] RSI, MACD, Bollinger Bands, volume spike, support/resistance
- [x] Claude AI confidence scoring with 65-threshold gate
- [x] WhatsApp alerts via Twilio
- [x] Push notifications via Pushover
- [x] FastAPI dashboard with TradingView candlestick chart
- [x] Paper trading mode
- [x] Signal logger → `signals_log.csv`
- [x] Watchlist persistence (`watchlist.json`)
- [x] Market hours awareness (9:30 AM – 4:00 PM EST)
- [x] Daily report at 4:30 PM EST

### v2.0 — Current ✅

- [x] Multi-ticker monitoring loop
- [x] Morning market scanner (7:45 AM, ~6,000 stocks)
- [x] 24/7 news watcher — discovers unknown stocks from news events
- [x] StockTwits sentiment (replaces Reddit)
- [x] 4-layer scoring: context + setup + execution − risk penalty
- [x] 4 named setup patterns with individual hard filters and scoring
- [x] Best-match setup detection (all qualifying setups scored, highest wins)
- [x] Relative strength vs SPY + QQQ (1d / 5d / 20d weighted composite)
- [x] News classification into 10 catalyst categories (phrase-level)
- [x] Self-learner — tracks 7 signal win rates across CSV history
- [x] Forward return accuracy (actual historical close, not spot price)
- [x] Dollar volume Gate 0 (eliminates illiquid stocks before scoring)
- [x] Score breakdown in every WhatsApp alert
- [x] Daily scheduler with 10 timed events

### v3.0 — Planned 📋

- [ ] Global rate limiter shared across all three discovery engines
- [ ] SEC EDGAR insider trading signal (free)
- [ ] Options flow via Unusual Whales ($50/mo)
- [ ] Dark pool prints via Unusual Whales ($50/mo)
- [ ] Sector ETF RS (third benchmark in composite)
- [ ] Parallel node execution in LangGraph pipeline
- [ ] Browser-based watchlist editor in dashboard
- [ ] Email digest option (SendGrid)

---

## 14. Important Disclaimers

> ⚠️ **This software is for educational and research purposes only.**

- **Not financial advice.** Nothing in this project constitutes investment advice, a recommendation to buy or sell any security, or a solicitation of any kind.
- **Paper trade first.** Always run in `--paper` mode for a minimum of 30 days before considering any live use.
- **Past performance does not guarantee future results.** A high backtest win rate on historical data does not mean the strategy will perform the same going forward.
- **Never risk money you cannot afford to lose.** Algorithmic trading systems can and do produce losing trades, streaks of losses, and complete failures in certain market conditions.
- **The AI makes mistakes.** Claude is a language model, not a licensed financial analyst. Its reasoning can be incorrect, incomplete, or confidently wrong.
- **You are responsible for your own trading decisions.** The authors of this software accept no liability for financial losses incurred through its use.

---

## 15. Author

**Dan Nicolau**
Senior QA Engineer → AI QA Architect

- GitHub: [github.com/dannicolau7](https://github.com/dannicolau7)

---

*Built with Claude Sonnet 4.6, LangGraph, and a lot of coffee.*
