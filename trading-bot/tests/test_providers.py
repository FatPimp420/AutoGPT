"""Tests for CoinGecko, CryptoPanic, and Reddit providers using mocked HTTP."""
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from sentiment.providers.coingecko import CoinGeckoProvider
from sentiment.providers.cryptopanic import CryptoPanicProvider
from sentiment.providers.reddit import RedditProvider
from sentiment.providers.factory import get_provider, MultiProvider
from sentiment.base import SentimentScore


# ── CoinGeckoProvider ────────────────────────────────────────────────────────

_COINGECKO_RESPONSE = {
    "name": "Bitcoin",
    "sentiment_votes_up_percentage": 72.5,
    "sentiment_votes_down_percentage": 27.5,
    "description": {
        "en": (
            "Bitcoin is the first decentralized cryptocurrency. "
            "It uses peer-to-peer technology to operate. "
            "Bitcoin has seen strong institutional adoption recently."
        )
    },
    "community_data": {
        "reddit_subscribers": 4_500_000,
        "twitter_followers": 6_000_000,
    },
}


@patch("sentiment.providers.coingecko.requests.get")
def test_coingecko_get_headlines_returns_list(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _COINGECKO_RESPONSE
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    provider = CoinGeckoProvider()
    headlines = provider.get_headlines("BTC/USDT")

    assert len(headlines) >= 1
    sources = {h.source for h in headlines}
    assert "coingecko_votes" in sources


@patch("sentiment.providers.coingecko.requests.get")
def test_coingecko_bullish_vote_score_positive(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _COINGECKO_RESPONSE
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    provider = CoinGeckoProvider()
    score = provider.get_score("BTC/USDT")

    # 72.5% up → blended score should be positive
    assert score.score > 0
    assert score.symbol == "BTC/USDT"
    assert 0.0 <= score.confidence <= 1.0


@patch("sentiment.providers.coingecko.requests.get")
def test_coingecko_bearish_vote_score_negative(mock_get):
    bearish_resp = {**_COINGECKO_RESPONSE,
                    "sentiment_votes_up_percentage": 28.0,
                    "sentiment_votes_down_percentage": 72.0}
    mock_resp = MagicMock()
    mock_resp.json.return_value = bearish_resp
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    provider = CoinGeckoProvider()
    score = provider.get_score("ETH/USDT")
    assert score.score < 0


@patch("sentiment.providers.coingecko.requests.get")
def test_coingecko_unknown_symbol_returns_zero(mock_get):
    provider = CoinGeckoProvider()
    headlines = provider.get_headlines("UNKNOWN/USDT")
    assert headlines == []


@patch("sentiment.providers.coingecko.requests.get")
def test_coingecko_network_error_returns_zero_score(mock_get):
    import requests as req
    mock_get.side_effect = req.RequestException("timeout")

    provider = CoinGeckoProvider()
    score = provider.get_score("BTC/USDT")
    assert score.score == 0.0
    assert score.confidence == 0.0


# ── CryptoPanicProvider ──────────────────────────────────────────────────────

_CRYPTOPANIC_POSTS = {
    "count": 42,
    "results": [
        {
            "title": "Bitcoin surges past key resistance level",
            "votes": {"liked": 120, "disliked": 15, "important": 30, "saved": 8},
            "kind": "news",
        },
        {
            "title": "ETF approval hopes boost crypto market rally",
            "votes": {"liked": 85, "disliked": 5, "important": 20, "saved": 3},
            "kind": "bullish",
        },
        {
            "title": "Regulatory concerns weigh on Bitcoin price",
            "votes": {"liked": 10, "disliked": 45, "important": 5, "saved": 1},
            "kind": "news",
        },
    ],
}

_CRYPTOPANIC_BULLISH = {"count": 30}
_CRYPTOPANIC_BEARISH = {"count": 12}


@patch("sentiment.providers.cryptopanic.requests.get")
def test_cryptopanic_get_headlines(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _CRYPTOPANIC_POSTS
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    provider = CryptoPanicProvider(api_key="test_key")
    headlines = provider.get_headlines("BTC/USDT")

    assert len(headlines) == 3
    assert all(h.source == "cryptopanic" for h in headlines)


@patch("sentiment.providers.cryptopanic.requests.get")
def test_cryptopanic_bullish_posts_score_positive(mock_get):
    responses = [_CRYPTOPANIC_POSTS, _CRYPTOPANIC_BULLISH, _CRYPTOPANIC_BEARISH]
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.side_effect = responses
    mock_get.return_value = mock_resp

    provider = CryptoPanicProvider(api_key="test_key")
    score = provider.get_score("BTC/USDT")

    assert score.score > 0
    assert score.headline_count == 3


@patch("sentiment.providers.cryptopanic.requests.get")
def test_cryptopanic_no_key_returns_empty(mock_get):
    provider = CryptoPanicProvider(api_key="")
    headlines = provider.get_headlines("BTC/USDT")
    assert headlines == []
    mock_get.assert_not_called()


@patch("sentiment.providers.cryptopanic.requests.get")
def test_cryptopanic_boosted_weight_for_labeled_posts(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _CRYPTOPANIC_POSTS
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    provider = CryptoPanicProvider(api_key="test_key")
    headlines = provider.get_headlines("BTC/USDT")

    # The "bullish" kind post should have higher weight than plain "news"
    bullish_h = next(h for h in headlines if "[BULLISH]" in h.text)
    news_h = next(h for h in headlines if "[BULLISH]" not in h.text and "[BEARISH]" not in h.text)
    assert bullish_h.weight > news_h.weight


# ── RedditProvider ───────────────────────────────────────────────────────────

def _make_mock_post(title: str, upvote_ratio: float = 0.85, stickied: bool = False):
    post = MagicMock()
    post.title = title
    post.upvote_ratio = upvote_ratio
    post.stickied = stickied
    comment = MagicMock()
    comment.body = f"Great news about {title[:20]}"
    post.comments = MagicMock()
    post.comments.replace_more.return_value = None
    post.comments.__iter__ = MagicMock(return_value=iter([comment]))
    return post


@patch("sentiment.providers.reddit.RedditProvider._get_reddit")
def test_reddit_get_headlines(mock_get_reddit):
    posts = [
        _make_mock_post("Bitcoin rally continues as adoption grows"),
        _make_mock_post("BTC breaks resistance — analysts bullish"),
        _make_mock_post("Mod stickied post", stickied=True),
    ]
    mock_sub = MagicMock()
    mock_sub.hot.return_value = iter(posts)
    mock_reddit = MagicMock()
    mock_reddit.subreddit.return_value = mock_sub
    mock_get_reddit.return_value = mock_reddit

    provider = RedditProvider(client_id="cid", client_secret="csec")
    headlines = provider.get_headlines("BTC/USDT")

    # Stickied post skipped; 2 posts × (title + 1 comment) = 4 headlines
    assert len(headlines) == 4
    post_titles = [h.text for h in headlines if h.source == "reddit/r/Bitcoin"]
    assert any("rally" in t.lower() for t in post_titles)
    assert any("breaks" in t.lower() for t in post_titles)


@patch("sentiment.providers.reddit.RedditProvider._get_reddit")
def test_reddit_score_positive_posts(mock_get_reddit):
    posts = [
        _make_mock_post("Bitcoin surges to new record high amid adoption", upvote_ratio=0.95),
        _make_mock_post("Institutional rally boosts crypto market", upvote_ratio=0.90),
    ]
    mock_sub = MagicMock()
    mock_sub.hot.return_value = iter(posts)
    mock_reddit = MagicMock()
    mock_reddit.subreddit.return_value = mock_sub
    mock_get_reddit.return_value = mock_reddit

    provider = RedditProvider(client_id="cid", client_secret="csec")
    score = provider.get_score("BTC/USDT")

    assert score.score > 0
    assert score.headline_count > 0


def test_reddit_no_credentials_returns_empty():
    provider = RedditProvider(client_id="", client_secret="")
    headlines = provider.get_headlines("BTC/USDT")
    assert headlines == []


# ── Provider factory ─────────────────────────────────────────────────────────

def test_factory_mock_default(monkeypatch):
    monkeypatch.delenv("SENTIMENT_PROVIDER", raising=False)
    provider = get_provider()
    from sentiment.mock import MockSentimentProvider
    assert isinstance(provider, MockSentimentProvider)


def test_factory_mock_explicit():
    provider = get_provider("mock")
    from sentiment.mock import MockSentimentProvider
    assert isinstance(provider, MockSentimentProvider)


def test_factory_coingecko():
    provider = get_provider("coingecko")
    assert isinstance(provider, CoinGeckoProvider)


def test_factory_cryptopanic():
    provider = get_provider("cryptopanic")
    assert isinstance(provider, CryptoPanicProvider)


def test_factory_reddit():
    provider = get_provider("reddit")
    assert isinstance(provider, RedditProvider)


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown SENTIMENT_PROVIDER"):
        get_provider("telegram")


def test_factory_multi_no_keys_falls_back_to_coingecko(monkeypatch):
    monkeypatch.delenv("CRYPTOPANIC_API_KEY", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    provider = get_provider("multi")
    assert isinstance(provider, MultiProvider)


def test_factory_env_var(monkeypatch):
    monkeypatch.setenv("SENTIMENT_PROVIDER", "coingecko")
    provider = get_provider()
    assert isinstance(provider, CoinGeckoProvider)
    monkeypatch.delenv("SENTIMENT_PROVIDER")
