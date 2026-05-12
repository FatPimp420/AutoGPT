"""Reddit sentiment provider via PRAW.

Register an app at https://www.reddit.com/prefs/apps (select "script").
Set env vars:
  REDDIT_CLIENT_ID=your_client_id
  REDDIT_CLIENT_SECRET=your_client_secret
  REDDIT_USER_AGENT=trading-bot/1.0 (optional, defaults to a sensible value)

Rate limit: 60 req/min on free tier.
"""
import os
from typing import Dict, List, Optional

from sentiment.base import Headline, SentimentProvider, SentimentScore
from sentiment.scorer import KeywordScorer

# Primary subreddits per symbol — ordered by signal quality
_SYMBOL_TO_SUBS: Dict[str, List[str]] = {
    "BTC/USDT": ["Bitcoin", "CryptoCurrency", "btc"],
    "ETH/USDT": ["ethereum", "CryptoCurrency", "ethtrader"],
    "SOL/USDT": ["solana", "CryptoCurrency"],
    "BNB/USDT": ["binance", "CryptoCurrency"],
    "XRP/USDT": ["Ripple", "CryptoCurrency", "XRP"],
}

_POST_LIMIT = 25   # hot posts per subreddit
_COMMENT_LIMIT = 5  # top comments per post


class RedditProvider(SentimentProvider):
    """Scans hot posts + top comments in crypto subreddits."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        posts_per_sub: int = 15,
    ):
        self._client_id = client_id or os.getenv("REDDIT_CLIENT_ID", "")
        self._client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "")
        self._user_agent = (
            user_agent
            or os.getenv("REDDIT_USER_AGENT", "trading-bot/1.0 by AutoGPT-Ruflo")
        )
        self._posts_per_sub = posts_per_sub
        self._scorer = KeywordScorer()
        self._reddit = None  # lazy init

    def _get_reddit(self):
        if self._reddit is None:
            import praw
            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
        return self._reddit

    def get_headlines(self, symbol: str) -> List[Headline]:
        subs = _SYMBOL_TO_SUBS.get(symbol, ["CryptoCurrency"])
        if not self._client_id or not self._client_secret:
            return []

        reddit = self._get_reddit()
        headlines = []

        for sub_name in subs[:2]:  # cap at 2 subs to stay within rate limits
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.hot(limit=self._posts_per_sub):
                    # Skip stickied mod posts
                    if post.stickied:
                        continue

                    # Upvote ratio as weight: 0.95 ratio → weight 1.3
                    weight = round(0.5 + post.upvote_ratio, 2)

                    headlines.append(Headline(
                        text=post.title,
                        source=f"reddit/r/{sub_name}",
                        weight=weight,
                    ))

                    # Top comments add texture
                    post.comments.replace_more(limit=0)
                    for comment in list(post.comments)[:_COMMENT_LIMIT]:
                        if hasattr(comment, "body") and len(comment.body) > 20:
                            headlines.append(Headline(
                                text=comment.body[:200],
                                source=f"reddit/r/{sub_name}/comment",
                                weight=round(weight * 0.7, 2),
                            ))
            except Exception:
                continue  # skip inaccessible subreddits silently

        return headlines

    def get_score(self, symbol: str) -> SentimentScore:
        headlines = self.get_headlines(symbol)
        if not headlines:
            return SentimentScore(symbol=symbol, score=0.0, confidence=0.0,
                                  headline_count=0)
        return self._scorer.score_headlines(symbol, headlines)
