"""Mock sentiment provider for offline testing and development.

Generates deterministic, realistic-looking headlines per symbol using a seeded
RNG. The seed is derived from symbol + date so the same day always yields the
same headlines, but different days yield different results.

Swap this for a real provider (RedditProvider, NewsAPIProvider, etc.) by
subclassing SentimentProvider and passing it to SignalBlender.
"""
import hashlib
import random
from datetime import date
from typing import List

from .base import Headline, SentimentProvider, SentimentScore
from .scorer import KeywordScorer

# Template headlines with {coin} placeholder and a polarity tag
_BULLISH_TEMPLATES = [
    "{coin} surges to new monthly high amid institutional demand",
    "{coin} rally continues as ETF approval speculation grows",
    "{coin} adoption hits record levels in Q2 report",
    "Major partnership announced: {coin} integration goes live",
    "{coin} breaks out of key resistance — analysts bullish",
    "Institutional investors accumulate {coin} at current levels",
    "{coin} outperforms peers as market sentiment improves",
    "Upgrade to {coin} network boosts transaction throughput",
    "{coin} buy signal flashes on weekly chart",
    "Positive regulatory clarity expected for {coin}: sources",
]

_BEARISH_TEMPLATES = [
    "{coin} drops sharply on rising sell pressure",
    "Analysts warn of {coin} bubble as RSI enters overbought zone",
    "{coin} faces uncertainty after regulatory crackdown",
    "Hack reported on {coin}-linked protocol — funds at risk",
    "{coin} falls as broader crypto fear grips market",
    "Short sellers increase bets against {coin}",
    "{coin} sell-off accelerates — support levels breached",
    "Lawsuit filed against {coin} foundation over token sales",
    "{coin} liquidations spike as market dumps overnight",
    "Warning: {coin} on-chain metrics signal bearish divergence",
]

_NEUTRAL_TEMPLATES = [
    "{coin} price stable as market awaits Fed decision",
    "{coin} trading sideways — no clear direction",
    "Developers release {coin} v2.1 with minor improvements",
    "{coin} volume flat despite broader market movement",
    "Analysis: {coin} at pivotal level after recent consolidation",
]

_COIN_NAMES = {
    "BTC/USDT": "Bitcoin",
    "ETH/USDT": "Ethereum",
    "SOL/USDT": "Solana",
    "BNB/USDT": "BNB",
    "XRP/USDT": "XRP",
}


def _seed_for(symbol: str, today: date) -> int:
    key = f"{symbol}-{today.isoformat()}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


class MockSentimentProvider(SentimentProvider):
    """Returns seeded deterministic headlines — no network required."""

    def __init__(self, headlines_per_symbol: int = 8, seed_date: date | None = None):
        self.headlines_per_symbol = headlines_per_symbol
        self._seed_date = seed_date or date.today()
        self._scorer = KeywordScorer()

    def get_headlines(self, symbol: str) -> List[Headline]:
        rng = random.Random(_seed_for(symbol, self._seed_date))
        coin = _COIN_NAMES.get(symbol, symbol.split("/")[0])
        n = self.headlines_per_symbol

        # weighted distribution: 40% bullish, 35% bearish, 25% neutral
        pool = (
            rng.sample(_BULLISH_TEMPLATES, min(4, len(_BULLISH_TEMPLATES)))
            + rng.sample(_BEARISH_TEMPLATES, min(3, len(_BEARISH_TEMPLATES)))
            + rng.sample(_NEUTRAL_TEMPLATES, min(2, len(_NEUTRAL_TEMPLATES)))
        )
        chosen = rng.sample(pool, min(n, len(pool)))

        return [
            Headline(
                text=tmpl.format(coin=coin),
                source="mock",
                weight=round(rng.uniform(0.8, 1.2), 2),
            )
            for tmpl in chosen
        ]

    def get_score(self, symbol: str) -> SentimentScore:
        headlines = self.get_headlines(symbol)
        return self._scorer.score_headlines(symbol, headlines)
