"""Unit tests for the sentiment module."""
import pytest
from sentiment.base import Headline, SentimentScore
from sentiment.scorer import KeywordScorer, score_text
from sentiment.mock import MockSentimentProvider
from sentiment.blender import SignalBlender


# ── score_text ───────────────────────────────────────────────────────────────

def test_score_text_bullish():
    assert score_text("Bitcoin surges to new ATH amid institutional adoption") > 0.3


def test_score_text_bearish():
    assert score_text("Bitcoin crashes amid fraud lawsuit and regulatory ban") < -0.3


def test_score_text_neutral():
    score = score_text("Bitcoin trading sideways with no clear direction today")
    assert -0.3 <= score <= 0.3


def test_score_text_negation_flips_sign():
    positive = score_text("Bitcoin rally continues")
    negated = score_text("Bitcoin rally does not continue, no gain expected")
    assert negated < positive


def test_score_text_empty():
    assert score_text("") == 0.0


# ── KeywordScorer ────────────────────────────────────────────────────────────

def test_scorer_empty_headlines():
    scorer = KeywordScorer()
    result = scorer.score_headlines("BTC/USDT", [])
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.headline_count == 0


def test_scorer_bullish_headlines():
    scorer = KeywordScorer()
    headlines = [
        Headline("Bitcoin surges to record high"),
        Headline("Institutional adoption drives rally"),
        Headline("ETF approval boosts Bitcoin"),
    ]
    result = scorer.score_headlines("BTC/USDT", headlines)
    assert result.score > 0
    assert result.label == "bullish"
    assert result.headline_count == 3


def test_scorer_bearish_headlines():
    scorer = KeywordScorer()
    headlines = [
        Headline("Bitcoin crashes on regulatory ban"),
        Headline("Hack reported, funds lost in exploit"),
    ]
    result = scorer.score_headlines("BTC/USDT", headlines)
    assert result.score < 0
    assert result.label == "bearish"


def test_scorer_confidence_scales_with_coverage():
    scorer = KeywordScorer()
    few = scorer.score_headlines("ETH/USDT", [Headline("Ethereum surges")])
    many = scorer.score_headlines(
        "ETH/USDT",
        [Headline("Ethereum surges and rallies to new high")] * 10,
    )
    assert many.confidence >= few.confidence


# ── MockSentimentProvider ────────────────────────────────────────────────────

def test_mock_provider_deterministic():
    p = MockSentimentProvider()
    s1 = p.get_score("BTC/USDT")
    s2 = p.get_score("BTC/USDT")
    assert s1.score == s2.score


def test_mock_provider_different_symbols_differ():
    p = MockSentimentProvider()
    btc = p.get_score("BTC/USDT")
    eth = p.get_score("ETH/USDT")
    # Scores may coincidentally match, but headlines must differ
    btc_texts = {h.text for h in p.get_headlines("BTC/USDT")}
    eth_texts = {h.text for h in p.get_headlines("ETH/USDT")}
    assert btc_texts != eth_texts


def test_mock_provider_headline_count():
    p = MockSentimentProvider(headlines_per_symbol=6)
    headlines = p.get_headlines("SOL/USDT")
    assert len(headlines) == 6


def test_mock_provider_all_default_symbols():
    p = MockSentimentProvider()
    for sym in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]:
        score = p.get_score(sym)
        assert -1.0 <= score.score <= 1.0
        assert 0.0 <= score.confidence <= 1.0


# ── SignalBlender ────────────────────────────────────────────────────────────

class _StubProvider:
    """Returns a fixed score for testing."""
    def __init__(self, score: float, confidence: float = 0.8):
        self._score = score
        self._confidence = confidence

    def get_score(self, symbol: str) -> SentimentScore:
        from datetime import date
        return SentimentScore(
            symbol=symbol,
            score=self._score,
            confidence=self._confidence,
            headline_count=5,
        )


def test_blender_no_signal_returns_none():
    blender = SignalBlender(provider=_StubProvider(0.5))
    result = blender.blend("BTC/USDT", None)
    assert result.final_signal is None
    assert result.action == "neutral"


def test_blender_low_confidence_passes_through():
    blender = SignalBlender(provider=_StubProvider(score=-0.9, confidence=0.1))
    result = blender.blend("BTC/USDT", "buy")
    assert result.final_signal == "buy"
    assert result.action == "passed"


def test_blender_contradicted_blocks_signal():
    # Strong bearish sentiment contradicts a buy signal
    blender = SignalBlender(provider=_StubProvider(score=-0.6, confidence=0.9))
    result = blender.blend("BTC/USDT", "buy")
    assert result.final_signal is None
    assert result.action == "contradicted"


def test_blender_boosted_confirms_aligned_signal():
    # Strong bullish sentiment boosts a buy signal
    blender = SignalBlender(provider=_StubProvider(score=0.7, confidence=0.9))
    result = blender.blend("BTC/USDT", "buy")
    assert result.final_signal == "buy"
    assert result.action == "boosted"


def test_blender_confirmed_when_weakly_aligned():
    blender = SignalBlender(provider=_StubProvider(score=0.15, confidence=0.5))
    result = blender.blend("BTC/USDT", "sell")
    # Weak bearish score doesn't align strongly with sell — confirmed or passed
    assert result.final_signal in ("sell", None)


def test_blender_sell_contradicted_by_bullish():
    blender = SignalBlender(provider=_StubProvider(score=0.6, confidence=0.9))
    result = blender.blend("BTC/USDT", "sell")
    assert result.final_signal is None
    assert result.action == "contradicted"


def test_blender_to_dict_keys():
    blender = SignalBlender(provider=_StubProvider(0.4))
    result = blender.blend("ETH/USDT", "buy")
    d = result.to_dict()
    assert "original_signal" in d
    assert "final_signal" in d
    assert "sentiment" in d
    assert "blend_action" in d
