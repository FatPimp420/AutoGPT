"""CryptoPanic sentiment provider.

Free tier: unlimited requests (with key).
Register at https://cryptopanic.com/developers/api/

Set env var: CRYPTOPANIC_API_KEY=your_token

API docs: https://cryptopanic.com/developers/api/
"""
import os
from typing import Dict, List, Optional

import requests

from sentiment.base import Headline, SentimentProvider, SentimentScore
from sentiment.scorer import KeywordScorer

_BASE = "https://cryptopanic.com/api/v1"
_TIMEOUT = 10

# Map trading pair → CryptoPanic currency code
_SYMBOL_TO_CURRENCY: Dict[str, str] = {
    "BTC/USDT": "BTC",
    "ETH/USDT": "ETH",
    "SOL/USDT": "SOL",
    "BNB/USDT": "BNB",
    "XRP/USDT": "XRP",
}


class CryptoPanicProvider(SentimentProvider):
    """Fetches crypto-specific news with community vote scores."""

    def __init__(self, api_key: Optional[str] = None, max_posts: int = 20):
        self._key = api_key or os.getenv("CRYPTOPANIC_API_KEY", "")
        self._max_posts = max_posts
        self._scorer = KeywordScorer()

    def _currency(self, symbol: str) -> Optional[str]:
        return _SYMBOL_TO_CURRENCY.get(symbol)

    def get_headlines(self, symbol: str) -> List[Headline]:
        currency = self._currency(symbol)
        if not currency or not self._key:
            return []

        try:
            resp = requests.get(
                f"{_BASE}/posts/",
                params={
                    "auth_token": self._key,
                    "currencies": currency,
                    "kind": "news",
                    "public": "true",
                },
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return []

        headlines = []
        for post in (data.get("results") or [])[: self._max_posts]:
            title = post.get("title", "").strip()
            if not title:
                continue

            # CryptoPanic vote scores as source credibility weight
            votes = post.get("votes") or {}
            liked = votes.get("liked") or 0
            disliked = votes.get("disliked") or 0
            total_votes = liked + disliked

            # Weight by engagement — more votes = more credible signal
            base_weight = 1.0
            if total_votes > 50:
                base_weight = 1.4
            elif total_votes > 10:
                base_weight = 1.2

            # If CryptoPanic gives us a direct sentiment label, boost weight
            kind = post.get("kind", "")
            if kind == "bullish":
                title = f"[BULLISH] {title}"
                base_weight *= 1.3
            elif kind == "bearish":
                title = f"[BEARISH] {title}"
                base_weight *= 1.3

            # Panic score → negative signal
            panic_count = (post.get("votes") or {}).get("disliked", 0)
            if panic_count > 20:
                title += " — high panic sentiment"

            headlines.append(Headline(
                text=title,
                source="cryptopanic",
                weight=round(base_weight, 2),
            ))

        return headlines

    def get_score(self, symbol: str) -> SentimentScore:
        headlines = self.get_headlines(symbol)
        if not headlines:
            return SentimentScore(symbol=symbol, score=0.0, confidence=0.0,
                                  headline_count=0)

        base_score = self._scorer.score_headlines(symbol, headlines)

        # Also fetch overall sentiment stats from CryptoPanic if available
        currency = self._currency(symbol)
        try:
            resp = requests.get(
                f"{_BASE}/posts/",
                params={
                    "auth_token": self._key,
                    "currencies": currency,
                    "filter": "bullish",
                    "public": "true",
                },
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            bullish_count = resp.json().get("count") or 0

            resp2 = requests.get(
                f"{_BASE}/posts/",
                params={
                    "auth_token": self._key,
                    "currencies": currency,
                    "filter": "bearish",
                    "public": "true",
                },
                timeout=_TIMEOUT,
            )
            resp2.raise_for_status()
            bearish_count = resp2.json().get("count") or 0

            total = bullish_count + bearish_count
            if total > 0:
                vote_score = (bullish_count - bearish_count) / total
                blended = 0.5 * vote_score + 0.5 * base_score.score
                return SentimentScore(
                    symbol=symbol,
                    score=round(blended, 4),
                    confidence=round(min(1.0, base_score.confidence + 0.2), 4),
                    headline_count=base_score.headline_count,
                    headlines=base_score.headlines,
                )
        except requests.RequestException:
            pass

        return base_score
