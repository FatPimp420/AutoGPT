"""CoinGecko sentiment provider.

Uses the free public API — no key required.
Endpoints used:
  /coins/{id}  → sentiment_votes_up/down_percentage, community data
  /coins/{id}/categories (via search) for trending context

Rate limit: 30 req/min on free tier. We make 1 call per symbol.
"""
import time
from typing import Dict, List, Optional

import requests

from sentiment.base import Headline, SentimentProvider, SentimentScore
from sentiment.scorer import KeywordScorer

_BASE = "https://api.coingecko.com/api/v3"

# Map trading pair → CoinGecko coin id
_SYMBOL_TO_ID: Dict[str, str] = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "SOL/USDT": "solana",
    "BNB/USDT": "binancecoin",
    "XRP/USDT": "ripple",
}

_TIMEOUT = 10  # seconds


def _symbol_to_id(symbol: str) -> Optional[str]:
    return _SYMBOL_TO_ID.get(symbol)


class CoinGeckoProvider(SentimentProvider):
    """No API key required. Pulls community sentiment votes + description text."""

    def __init__(self, api_key: Optional[str] = None, calls_per_minute: int = 25):
        self._api_key = api_key  # optional Pro key for higher rate limits
        self._min_interval = 60.0 / calls_per_minute
        self._last_call: float = 0.0
        self._scorer = KeywordScorer()

    def _get(self, path: str, params: dict = {}) -> dict:
        # Respect rate limit
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        headers = {}
        if self._api_key:
            headers["x-cg-pro-api-key"] = self._api_key
        resp = requests.get(f"{_BASE}{path}", params=params, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        self._last_call = time.monotonic()
        return resp.json()

    def get_headlines(self, symbol: str) -> List[Headline]:
        coin_id = _symbol_to_id(symbol)
        if not coin_id:
            return []

        data = self._get(
            f"/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "false",
                "community_data": "true",
                "developer_data": "false",
                "sparkline": "false",
            },
        )

        headlines = []

        # Sentiment votes → synthetic headline
        up = data.get("sentiment_votes_up_percentage") or 0.0
        down = data.get("sentiment_votes_down_percentage") or 0.0
        coin_name = data.get("name", symbol.split("/")[0])

        if up > 60:
            headlines.append(Headline(
                text=f"{coin_name} community sentiment strongly bullish — {up:.0f}% positive votes",
                source="coingecko_votes",
                weight=1.5,
            ))
        elif down > 60:
            headlines.append(Headline(
                text=f"{coin_name} community sentiment bearish — {down:.0f}% negative votes",
                source="coingecko_votes",
                weight=1.5,
            ))
        else:
            headlines.append(Headline(
                text=f"{coin_name} community sentiment neutral — {up:.0f}% positive",
                source="coingecko_votes",
                weight=0.8,
            ))

        # Description snippets (first 3 sentences) for keyword scoring
        desc = (data.get("description") or {}).get("en", "")
        if desc:
            sentences = [s.strip() for s in desc.split(".") if len(s.strip()) > 20][:3]
            for s in sentences:
                headlines.append(Headline(text=s, source="coingecko_desc", weight=0.5))

        # Community stats → additional signals
        community = data.get("community_data") or {}
        reddit_subs = community.get("reddit_subscribers") or 0
        twitter_followers = community.get("twitter_followers") or 0
        if reddit_subs > 1_000_000:
            headlines.append(Headline(
                text=f"{coin_name} has strong community with {reddit_subs:,} Reddit subscribers",
                source="coingecko_community",
                weight=0.6,
            ))

        return headlines

    def get_score(self, symbol: str) -> SentimentScore:
        try:
            headlines = self.get_headlines(symbol)
            score = self._scorer.score_headlines(symbol, headlines)

            # Override with direct vote ratio if available (more reliable)
            coin_id = _symbol_to_id(symbol)
            if coin_id:
                data = self._get(
                    f"/coins/{coin_id}",
                    params={"localization": "false", "tickers": "false",
                            "market_data": "false", "community_data": "false"},
                )
                up = data.get("sentiment_votes_up_percentage") or 0.0
                if up > 0:
                    # Convert vote % to [-1, +1]: 50% → 0, 100% → +1, 0% → -1
                    vote_score = (up - 50.0) / 50.0
                    # Blend: 60% vote score + 40% keyword score
                    blended = 0.6 * vote_score + 0.4 * score.score
                    return SentimentScore(
                        symbol=symbol,
                        score=round(blended, 4),
                        confidence=round(min(1.0, score.confidence + 0.3), 4),
                        headline_count=score.headline_count,
                        headlines=score.headlines,
                    )
            return score
        except requests.RequestException as e:
            return SentimentScore(symbol=symbol, score=0.0, confidence=0.0,
                                  headline_count=0)
