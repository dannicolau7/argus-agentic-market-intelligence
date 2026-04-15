"""
edgar_watcher.py — SEC EDGAR 8-K real-time filing watcher.

Polls the EDGAR current-events Atom feed every 60 seconds. Each new 8-K
filing (earnings, FDA decisions, contract wins, partnerships, etc.) is filed
with the SEC within minutes of the press release — before any news aggregator
(including Polygon) picks it up.

Flow:
  1. Load CIK → ticker map from https://www.sec.gov/files/company_tickers.json
  2. Every 60 s: fetch EDGAR 8-K Atom feed, extract new filings
  3. For each new filing: CIK → ticker, quick gate (price/volume sanity),
     cooldown check
  4. Send preliminary WhatsApp: "📋 TICKER filed 8-K with SEC — analyzing..."
  5. Run full pipeline with news_triggered=True (confidence threshold = 55)

Shares _alerted_at cooldown with news_watcher so the same ticker is never
double-alerted across Polygon, Yahoo Finance, and EDGAR sources.
"""

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from alerts import send_whatsapp
from news_watcher import _quick_gate, _already_alerted, _mark_alerted, _run_pipeline

EST            = ZoneInfo("America/New_York")
EDGAR_INTERVAL = 60   # seconds between polls

EDGAR_FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include"
    "&count=40&search_text=&output=atom"
)
EDGAR_HEADERS  = {"User-Agent": "argus contact@example.com"}
ATOM_NS        = "http://www.w3.org/2005/Atom"

_edgar_seen: set = set()   # accession numbers already processed


# ── CIK → ticker map ──────────────────────────────────────────────────────────

def _load_cik_map() -> dict:
    """
    Download the SEC company tickers JSON once at startup.
    Returns {cik_str_no_leading_zeros: ticker_symbol}.
    """
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS,
            timeout=15,
        )
        data = r.json()
        # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        return {str(v["cik_str"]): v["ticker"].upper() for v in data.values()}
    except Exception as e:
        print(f"⚠️  [EDGAR] Could not load CIK map: {e}")
        return {}


# ── Feed parsing ───────────────────────────────────────────────────────────────

def _fetch_8k_feed() -> list:
    """
    Fetch and parse the EDGAR 8-K Atom feed.
    Returns list of {accession, cik, company_name, filed_at_ts}.
    """
    try:
        r = requests.get(EDGAR_FEED_URL, headers=EDGAR_HEADERS, timeout=15)
        if r.status_code != 200:
            return []

        root    = ET.fromstring(r.text)
        entries = root.findall(f"{{{ATOM_NS}}}entry")
        results = []

        for entry in entries:
            # Accession number from <id>
            id_text   = (entry.findtext(f"{{{ATOM_NS}}}id") or "").strip()
            accession = id_text.split("accession-number=")[-1] if "accession-number=" in id_text else id_text

            if not accession or accession in _edgar_seen:
                continue

            # Company name + CIK from <title> (format: "8-K - Company Name (0000320193) (Filer)")
            title = (entry.findtext(f"{{{ATOM_NS}}}title") or "").strip()
            m     = re.search(r'\((\d+)\)\s*\(Filer\)', title)
            cik   = m.group(1).lstrip("0") if m else None
            if not cik:
                continue

            company_name = re.sub(r'\s*\(\d+\).*$', '', title).replace("8-K - ", "").strip()

            # Filing timestamp from <updated>
            updated_str = (entry.findtext(f"{{{ATOM_NS}}}updated") or "").strip()
            try:
                filed_dt  = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                filed_ts  = filed_dt.timestamp()
            except Exception:
                filed_ts  = time.time()

            results.append({
                "accession":    accession,
                "cik":          cik,
                "company_name": company_name,
                "filed_at_ts":  filed_ts,
            })

        return results

    except Exception as e:
        print(f"⚠️  [EDGAR] Feed fetch error: {e}")
        return []


# ── Main loop ──────────────────────────────────────────────────────────────────

async def edgar_watcher_loop(run_pipeline_fn=None, paper: bool = False):
    """
    Async loop started in main.py lifespan.

    run_pipeline_fn is unused (kept for API symmetry) — we call
    news_watcher._run_pipeline directly so we share its _alerted_at state.
    """
    mode = "PAPER" if paper else "LIVE"
    print(f"📋 [EDGAR] Started — polling every {EDGAR_INTERVAL}s | {mode}")

    loop = asyncio.get_running_loop()

    # Load CIK map at startup
    print("📋 [EDGAR] Loading CIK → ticker map from SEC...")
    cik_map = await loop.run_in_executor(None, _load_cik_map)
    print(f"📋 [EDGAR] CIK map loaded: {len(cik_map):,} companies")

    # Seed: mark current filings as seen without processing
    print("📋 [EDGAR] Seeding current 8-K filings...")
    seed = await loop.run_in_executor(None, _fetch_8k_feed)
    for f in seed:
        _edgar_seen.add(f["accession"])
    print(f"📋 [EDGAR] Seeded {len(_edgar_seen)} existing filings. Ready.")

    while True:
        try:
            await asyncio.sleep(EDGAR_INTERVAL)

            filings = await loop.run_in_executor(None, _fetch_8k_feed)
            new_filings = [f for f in filings if f["accession"] not in _edgar_seen]

            if not new_filings:
                continue

            print(f"\n📋 [EDGAR] {len(new_filings)} new 8-K filing(s)")

            for filing in new_filings:
                _edgar_seen.add(filing["accession"])

                cik          = filing["cik"]
                company_name = filing["company_name"]
                ticker       = cik_map.get(cik)

                if not ticker:
                    print(f"   ❓ [EDGAR] No ticker for CIK {cik} ({company_name}) — skipping")
                    continue

                # Skip if filed > 30 minutes ago (startup catch-up window)
                age_s = time.time() - filing["filed_at_ts"]
                if age_s > 1800:
                    continue

                # Quick gate: price $0.50–$50, min volume
                if not _quick_gate(ticker):
                    print(f"   ❌ [EDGAR] {ticker} failed price/volume gate")
                    continue

                # Cooldown: don't re-alert for same ticker within 4 hours
                if _already_alerted(ticker):
                    print(f"   ⏸️  [EDGAR] {ticker} in cooldown — skipping")
                    continue

                now_s = datetime.now(tz=EST).strftime("%H:%M")
                print(f"   📋 [EDGAR] {ticker} ({company_name}) filed 8-K at {now_s}")

                # Preliminary WhatsApp
                pre_msg = (
                    f"📋 8-K FILING DETECTED [{now_s}]\n"
                    f"{ticker} ({company_name})\n"
                    f"Filed with SEC — analyzing now..."
                )
                if not paper:
                    try:
                        send_whatsapp(pre_msg)
                    except Exception as e:
                        print(f"⚠️  [EDGAR] WhatsApp failed: {e}")
                else:
                    print(f"   📋 [PAPER] Would send:\n{pre_msg}")

                # Mark alerted BEFORE pipeline so concurrent sources don't double-alert
                _mark_alerted(ticker)

                # Full pipeline with news_triggered=True
                try:
                    result = await loop.run_in_executor(
                        None, _run_pipeline, ticker, paper, True
                    )
                    signal = result.get("signal", "HOLD")
                    conf   = result.get("confidence", 0)
                    emoji  = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
                    print(f"   {emoji} [EDGAR] {ticker} → {signal} {conf}/100")
                except Exception as e:
                    print(f"   ❌ [EDGAR] Pipeline error for {ticker}: {e}")

        except asyncio.CancelledError:
            print("📋 [EDGAR] Stopped.")
            break
        except Exception as e:
            print(f"❌ [EDGAR] Loop error: {e}")
            await asyncio.sleep(60)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching latest 8-K filings from SEC EDGAR...")
    filings = _fetch_8k_feed()

    print("Loading CIK map...")
    cik_map = _load_cik_map()

    print(f"\nLatest {min(5, len(filings))} 8-K filings:")
    for f in filings[:5]:
        ticker = cik_map.get(f["cik"], "???")
        age_m  = (time.time() - f["filed_at_ts"]) / 60
        print(f"  {ticker:6s}  {f['company_name'][:40]:40s}  {age_m:.0f} min ago")
