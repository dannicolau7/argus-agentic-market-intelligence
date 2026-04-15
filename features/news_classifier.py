"""
features/news_classifier.py — Classify a news headline into a catalyst category.

Design principles:
  1. All keywords are phrase-level (not single words) to avoid false positives.
     "offering" alone matches "cloud offering" — use "share offering" instead.
     "beat" alone matches "dead beat" — use "beat estimates" instead.
  2. Categories are checked in priority order; first match wins.
  3. Dilution patterns are checked first to avoid misclassifying them
     as partnerships or contract wins.
  4. Headlines are normalised (lowercased, collapse whitespace) before matching.

Categories and their score adjustments:
  fda_approval       +25   Strong bullish (biotech)
  earnings_beat      +25   Strong bullish
  contract_win       +20   Bullish catalyst
  partnership        +15   Moderately bullish
  upgrade            +10   Analyst upgrade
  general              0   No clear catalyst
  downgrade          -10   Analyst downgrade
  earnings_miss      -20   Strong bearish
  offering_dilution  -25   Share sale / ATM — often tanks stock immediately
  fda_rejection      -25   Strong bearish (biotech)
"""

import re


# ── Keyword lists ──────────────────────────────────────────────────────────────
# Each entry is a phrase (multi-word preferred) to minimise false positives.
# PRIORITY ORDER: dilution first (critical to catch before partnership check).

CATEGORIES: list[tuple[str, list[str]]] = [
    ("offering_dilution", [
        "share offering", "stock offering", "secondary offering",
        "registered direct", "private placement", "atm offering",
        "at-the-market offering", "prospectus supplement",
        "underwritten public offering", "direct offering",
        "sells shares", "share sale", "common stock offering",
        "pricing of", "prices offering", "upsized offering",
        "announces pricing", "priced its", "million shares",
        "shelf registration", "warrant exercise",
    ]),
    ("fda_rejection", [
        "fda rejects", "fda refuses", "fda declines",
        "complete response letter", "crl issued",
        "refuse to file", "not approvable", "fda rejection",
    ]),
    ("earnings_miss", [
        "misses estimates", "missed estimates", "earnings miss",
        "below estimates", "below expectations", "below consensus",
        "disappoints investors", "fell short of",
        "lowered guidance", "cuts outlook", "reduced outlook",
        "swings to loss", "wider loss",
    ]),
    ("downgrade", [
        "downgrades to sell", "downgrades to underperform",
        "downgrades to neutral", "cut to sell",
        "cut to underperform", "price target cut",
        "lowers price target", "reduces target",
    ]),
    ("fda_approval", [
        "fda approves", "fda approved", "fda approval",
        "fda clears", "fda clearance", "fda grants",
        "fda accepts", "breakthrough designation",
        "nda approved", "bla approved", "510(k) cleared",
        "receives fda", "granted fda",
    ]),
    ("earnings_beat", [
        "beats estimates", "beat estimates", "beats expectations",
        "beat expectations", "beats consensus", "beat consensus",
        "earnings beat", "tops estimates", "tops expectations",
        "record revenue", "record earnings", "record profit",
        "raises guidance", "raises outlook", "raises full-year",
        "profit rose", "revenue surged", "revenue jumped",
    ]),
    ("contract_win", [
        "wins contract", "awarded contract", "government contract",
        "defense contract", "selected as", "chosen as provider",
        "secures contract", "signs agreement with",
        "exclusive agreement", "supply agreement with",
    ]),
    ("upgrade", [
        "upgrades to buy", "upgrades to outperform",
        "raised to buy", "raised to outperform",
        "initiates buy", "initiates with buy",
        "price target raised", "raises price target",
        "increases target", "starts outperform",
    ]),
    ("partnership", [
        "strategic partnership", "joint venture",
        "collaboration agreement", "co-development agreement",
        "licensing agreement", "strategic alliance",
        "research collaboration", "commercialization agreement",
    ]),
]

CATEGORY_SCORES: dict[str, int] = {
    "fda_approval":      25,
    "earnings_beat":     25,
    "contract_win":      20,
    "partnership":       15,
    "upgrade":           10,
    "general":            0,
    "downgrade":        -10,
    "earnings_miss":    -20,
    "offering_dilution": -25,
    "fda_rejection":    -25,
}

BULLISH_CATEGORIES: set[str] = {
    "fda_approval", "earnings_beat", "contract_win", "upgrade", "partnership",
}
BEARISH_CATEGORIES: set[str] = {
    "fda_rejection", "earnings_miss", "offering_dilution", "downgrade",
}

_LABELS: dict[str, str] = {
    "earnings_beat":     "earnings beat 🟢",
    "fda_approval":      "FDA approval 🟢",
    "contract_win":      "contract win 🟢",
    "partnership":       "partnership 🟢",
    "upgrade":           "analyst upgrade 🟡",
    "general":           "general news",
    "downgrade":         "analyst downgrade 🔴",
    "earnings_miss":     "earnings miss 🔴",
    "offering_dilution": "share offering 🔴",
    "fda_rejection":     "FDA rejection 🔴",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def _normalise(headline: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for matching."""
    h = headline.lower()
    h = re.sub(r"[''`]", "", h)            # remove apostrophes
    h = re.sub(r"\s+", " ", h).strip()
    return h


def classify(headline: str) -> str:
    """Return the category for a headline string. Defaults to 'general'."""
    if not headline:
        return "general"
    norm = _normalise(headline)
    for category, phrases in CATEGORIES:
        if any(phrase in norm for phrase in phrases):
            return category
    return "general"


def score(category: str) -> int:
    """Points to add/subtract from the scanner score."""
    return CATEGORY_SCORES.get(category, 0)


def label(category: str) -> str:
    """Short human-readable label for prompts and WhatsApp messages."""
    return _LABELS.get(category, category)
