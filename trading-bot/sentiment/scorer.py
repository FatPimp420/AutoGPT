"""Keyword-based sentiment scorer.

Scores a piece of text on a [-1, +1] scale using weighted bullish/bearish
word lists. No external models required — works fully offline.
Swap get_score() internals for a real model (VADER, FinBERT, etc.) when ready.
"""
import re
from typing import List

from .base import Headline, SentimentScore

# --- lexicon -----------------------------------------------------------------

BULLISH_WORDS = {
    # strong positive
    "surge": 2.0, "soar": 2.0, "rally": 1.5, "breakout": 1.5, "bullrun": 2.0,
    "moon": 1.5, "pump": 1.2, "ath": 2.0, "adoption": 1.5, "partnership": 1.2,
    "approve": 1.5, "approval": 1.5, "etf": 1.5, "institutional": 1.2,
    "upgrade": 1.2, "buy": 1.0, "long": 1.0, "profit": 1.2, "gain": 1.0,
    "growth": 1.2, "positive": 1.0, "bullish": 2.0, "recover": 1.2,
    "rebound": 1.2, "outperform": 1.5, "mainstream": 1.0, "launch": 1.0,
    "record": 1.2, "high": 0.8, "rise": 1.0, "increase": 0.8, "up": 0.5,
    "exceed": 1.0, "beat": 1.0, "strong": 1.0, "demand": 1.0, "accumulate": 1.2,
}

BEARISH_WORDS = {
    # strong negative
    "crash": 2.0, "plunge": 2.0, "dump": 1.5, "collapse": 2.0, "hack": 2.0,
    "exploit": 1.8, "ban": 2.0, "regulation": 1.0, "lawsuit": 1.5, "fraud": 2.0,
    "scam": 2.0, "ponzi": 2.0, "bubble": 1.5, "bearish": 2.0, "sell": 1.0,
    "short": 1.0, "loss": 1.2, "drop": 1.2, "fall": 1.0, "decline": 1.2,
    "slump": 1.5, "tumble": 1.5, "fear": 1.2, "uncertainty": 1.0, "doubt": 1.0,
    "warning": 1.2, "risk": 0.8, "concern": 0.8, "problem": 1.0, "issue": 0.8,
    "sell-off": 1.8, "selloff": 1.8, "downtrend": 1.5, "bear": 1.2,
    "liquidation": 1.8, "fine": 1.2, "penalty": 1.2, "suspend": 1.5, "low": 0.8,
}

NEGATIONS = {"not", "no", "never", "without", "neither", "nor", "barely", "hardly"}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def score_text(text: str) -> float:
    """Return a raw sentiment score for a single text string.

    Handles simple negation: a negation word within 2 tokens of a sentiment
    word flips its sign.
    """
    tokens = _tokenize(text)
    total_weight = 0.0
    raw_score = 0.0

    for i, token in enumerate(tokens):
        weight = BULLISH_WORDS.get(token) or BEARISH_WORDS.get(token)
        if weight is None:
            continue

        sign = 1.0 if token in BULLISH_WORDS else -1.0
        # check preceding two tokens for negation
        window = tokens[max(0, i - 2): i]
        if any(t in NEGATIONS for t in window):
            sign *= -1.0

        raw_score += sign * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0
    # normalise to [-1, +1]
    return max(-1.0, min(1.0, raw_score / total_weight))


class KeywordScorer:
    """Aggregate sentiment scores across multiple headlines."""

    def score_headlines(self, symbol: str, headlines: List[Headline]) -> SentimentScore:
        if not headlines:
            return SentimentScore(
                symbol=symbol, score=0.0, confidence=0.0, headline_count=0
            )

        weighted_score = 0.0
        total_weight = 0.0
        scored_count = 0

        for h in headlines:
            s = score_text(h.text)
            w = h.weight
            weighted_score += s * w
            total_weight += w
            if s != 0.0:
                scored_count += 1

        score = weighted_score / total_weight if total_weight > 0 else 0.0
        # confidence: fraction of headlines that contained scored keywords,
        # discounted by number of headlines (more = more confident)
        coverage = scored_count / len(headlines)
        depth = min(1.0, len(headlines) / 10)  # saturates at 10 headlines
        confidence = coverage * depth

        return SentimentScore(
            symbol=symbol,
            score=round(score, 4),
            confidence=round(confidence, 4),
            headline_count=len(headlines),
            headlines=headlines,
        )
