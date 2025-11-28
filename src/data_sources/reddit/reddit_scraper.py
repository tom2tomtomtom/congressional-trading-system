"""
Reddit Intelligence Scraper
Tracks retail investor sentiment on r/wallstreetbets and correlates with congressional trading
"""

import praw
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class RedditScraper:
    """
    Scrapes Reddit for stock ticker mentions and sentiment
    Focuses on r/wallstreetbets, r/stocks, r/investing
    """

    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit API client

        Get credentials from: https://www.reddit.com/prefs/apps
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

        # Subreddits to monitor
        self.subreddits = [
            'wallstreetbets',
            'stocks',
            'investing',
            'StockMarket',
            'options'
        ]

        # Common stock ticker pattern
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')

        # Words that indicate sentiment
        self.bullish_words = {
            'moon', 'rocket', 'buy', 'calls', 'bullish', 'long',
            'tendies', 'gains', 'pump', 'rally', 'breakout', 'squeeze'
        }

        self.bearish_words = {
            'crash', 'puts', 'short', 'bearish', 'dump', 'tank',
            'fall', 'drop', 'sell', 'collapse', 'plummet'
        }

    def scrape_subreddit(
        self,
        subreddit_name: str,
        limit: int = 100,
        time_filter: str = 'day'
    ) -> List[Dict]:
        """
        Scrape posts from a subreddit

        Args:
            subreddit_name: Name of subreddit (e.g., 'wallstreetbets')
            limit: Number of posts to fetch
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'

        Returns:
            List of post dictionaries
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []

        for post in subreddit.hot(limit=limit):
            # Extract ticker mentions
            text = f"{post.title} {post.selftext}"
            tickers = self.extract_tickers(text)

            # Calculate sentiment
            sentiment = self.calculate_sentiment(text)

            post_data = {
                'post_id': post.id,
                'subreddit': subreddit_name,
                'title': post.title,
                'content': post.selftext[:1000],  # Truncate long posts
                'author': str(post.author) if post.author else '[deleted]',
                'posted_at': datetime.fromtimestamp(post.created_utc),
                'score': post.score,  # Upvotes - Downvotes
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'url': f"https://reddit.com{post.permalink}",
                'mentioned_tickers': list(tickers),
                'sentiment_score': sentiment,
                'is_dd_post': self.is_due_diligence(post)
            }

            posts.append(post_data)

        logger.info(f"Scraped {len(posts)} posts from r/{subreddit_name}")
        return posts

    def extract_tickers(self, text: str) -> set:
        """
        Extract stock ticker symbols from text

        Uses pattern matching + filtering to reduce false positives
        """
        # Find all potential tickers (1-5 uppercase letters)
        potential_tickers = self.ticker_pattern.findall(text.upper())

        # Filter out common false positives
        false_positives = {
            'I', 'A', 'THE', 'TO', 'FOR', 'AND', 'OR', 'BUT',
            'DD', 'YOLO', 'FD', 'WSB', 'IMO', 'IMHO', 'CEO',
            'CFO', 'IPO', 'ETF', 'NYSE', 'SEC', 'FDA', 'PE',
            'PS', 'EPS', 'ATH', 'ATL', 'ER', 'IV', 'OTM', 'ITM'
        }

        # Filter tickers
        tickers = {
            ticker for ticker in potential_tickers
            if ticker not in false_positives
            and len(ticker) >= 1 and len(ticker) <= 5
        }

        return tickers

    def calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score from -1 (bearish) to +1 (bullish)

        Simple word-counting approach (can be replaced with ML model)
        """
        text_lower = text.lower()

        bullish_count = sum(1 for word in self.bullish_words if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_words if word in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        # Normalize to -1 to +1
        sentiment = (bullish_count - bearish_count) / total
        return round(sentiment, 3)

    def is_due_diligence(self, post) -> bool:
        """Check if post is labeled as Due Diligence (DD)"""
        flair = post.link_flair_text
        title = post.title.lower()

        return (
            (flair and 'DD' in flair.upper()) or
            'due diligence' in title or
            'dd:' in title or
            '[dd]' in title
        )

    def aggregate_ticker_sentiment(
        self,
        posts: List[Dict],
        min_mentions: int = 3
    ) -> Dict[str, Dict]:
        """
        Aggregate sentiment per ticker across all posts

        Args:
            posts: List of post dictionaries
            min_mentions: Minimum mentions to include ticker

        Returns:
            Dict of {ticker: {mentions, avg_sentiment, total_score, hype_level}}
        """
        ticker_data = {}

        for post in posts:
            for ticker in post['mentioned_tickers']:
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        'mentions': 0,
                        'sentiment_sum': 0,
                        'score_sum': 0,
                        'posts': []
                    }

                ticker_data[ticker]['mentions'] += 1
                ticker_data[ticker]['sentiment_sum'] += post['sentiment_score']
                ticker_data[ticker]['score_sum'] += post['score']
                ticker_data[ticker]['posts'].append(post['post_id'])

        # Calculate aggregates and filter by min_mentions
        aggregated = {}
        for ticker, data in ticker_data.items():
            if data['mentions'] >= min_mentions:
                avg_sentiment = data['sentiment_sum'] / data['mentions']

                aggregated[ticker] = {
                    'mentions': data['mentions'],
                    'avg_sentiment': round(avg_sentiment, 3),
                    'total_score': data['score_sum'],
                    'hype_level': self.calculate_hype_level(
                        data['mentions'],
                        data['score_sum']
                    ),
                    'top_posts': data['posts'][:5]  # Top 5 posts
                }

        return aggregated

    def calculate_hype_level(self, mentions: int, total_score: int) -> str:
        """
        Classify hype level: low, medium, high, extreme

        Based on mention count and total upvotes
        """
        # Combined hype metric
        hype_score = mentions * 10 + total_score

        if hype_score < 100:
            return 'low'
        elif hype_score < 500:
            return 'medium'
        elif hype_score < 2000:
            return 'high'
        else:
            return 'extreme'

    def scrape_all_subreddits(
        self,
        limit_per_sub: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Scrape all configured subreddits

        Returns:
            Dict of {subreddit_name: [posts]}
        """
        all_posts = {}

        for subreddit in self.subreddits:
            try:
                posts = self.scrape_subreddit(subreddit, limit=limit_per_sub)
                all_posts[subreddit] = posts
            except Exception as e:
                logger.error(f"Failed to scrape r/{subreddit}: {e}")
                all_posts[subreddit] = []

        return all_posts

    def get_trending_tickers(
        self,
        hours: int = 24,
        min_mentions: int = 5
    ) -> List[Dict]:
        """
        Get trending tickers across all subreddits in the last N hours

        Returns:
            List of tickers sorted by hype score
        """
        # Scrape recent posts
        all_posts = []
        for subreddit in self.subreddits:
            posts = self.scrape_subreddit(subreddit, limit=200, time_filter='day')

            # Filter by time window
            cutoff = datetime.now() - timedelta(hours=hours)
            recent_posts = [p for p in posts if p['posted_at'] >= cutoff]
            all_posts.extend(recent_posts)

        # Aggregate sentiment
        ticker_sentiment = self.aggregate_ticker_sentiment(all_posts, min_mentions)

        # Convert to list and sort by hype
        hype_order = {'extreme': 4, 'high': 3, 'medium': 2, 'low': 1}
        trending = [
            {'ticker': ticker, **data}
            for ticker, data in ticker_sentiment.items()
        ]
        trending.sort(
            key=lambda x: (hype_order[x['hype_level']], x['mentions']),
            reverse=True
        )

        return trending


class RedditCongressCorrelator:
    """
    Correlates Reddit sentiment with congressional trading
    Detects predatory behavior and information advantage
    """

    def __init__(self, reddit_scraper: RedditScraper, db_session):
        self.scraper = reddit_scraper
        self.db = db_session

    def detect_predatory_trading(
        self,
        member_id: int,
        days: int = 30
    ) -> List[Dict]:
        """
        Detect when member trades against Reddit sentiment

        Pattern: Reddit bullish → Member sells (dumping on retail)
                 Reddit bearish → Member buys (contrarian advantage)
        """
        from src.models.trading import Trade

        # Get member's recent trades
        trades = self.db.query(Trade).filter(
            Trade.member_id == member_id,
            Trade.transaction_date >= datetime.now() - timedelta(days=days)
        ).all()

        predatory_patterns = []

        for trade in trades:
            # Get Reddit sentiment at time of trade
            sentiment = self.get_reddit_sentiment_at_date(
                trade.symbol,
                trade.transaction_date
            )

            if sentiment is None:
                continue

            # Check for predatory behavior
            is_predatory = False
            explanation = ""

            if trade.transaction_type == 'sale' and sentiment['avg_sentiment'] > 0.5:
                is_predatory = True
                explanation = (
                    f"Sold {trade.symbol} while Reddit was highly bullish "
                    f"(sentiment: {sentiment['avg_sentiment']:.2f}, "
                    f"hype: {sentiment['hype_level']}) - Possible dump on retail"
                )

            elif trade.transaction_type == 'purchase' and sentiment['avg_sentiment'] < -0.5:
                is_predatory = True
                explanation = (
                    f"Bought {trade.symbol} while Reddit was highly bearish "
                    f"(sentiment: {sentiment['avg_sentiment']:.2f}) - "
                    f"Information advantage over retail"
                )

            if is_predatory:
                predatory_patterns.append({
                    'trade_id': trade.id,
                    'symbol': trade.symbol,
                    'transaction_type': trade.transaction_type,
                    'amount': float(trade.amount_mid or 0),
                    'transaction_date': trade.transaction_date,
                    'reddit_sentiment': sentiment['avg_sentiment'],
                    'reddit_hype_level': sentiment['hype_level'],
                    'reddit_mentions': sentiment['mentions'],
                    'explanation': explanation,
                    'predatory_score': self.calculate_predatory_score(trade, sentiment)
                })

        return predatory_patterns

    def get_reddit_sentiment_at_date(
        self,
        ticker: str,
        date: datetime
    ) -> Optional[Dict]:
        """
        Get Reddit sentiment for ticker around a specific date
        (±3 days window)
        """
        # This would query your RedditTickerSentiment table
        # For now, returning mock data structure

        # TODO: Implement database query
        # from src.models.social_media import RedditTickerSentiment
        # sentiment = self.db.query(RedditTickerSentiment).filter(...)

        return None  # Placeholder

    def calculate_predatory_score(self, trade, sentiment: Dict) -> float:
        """
        Calculate predatory behavior score (0-100)

        Factors:
        - Sentiment divergence (30 points)
        - Trade size (20 points)
        - Reddit hype level (25 points)
        - Timing precision (25 points)
        """
        score = 0.0

        # Sentiment divergence
        if trade.transaction_type == 'sale':
            score += min(sentiment['avg_sentiment'] * 30, 30)
        else:
            score += min(abs(sentiment['avg_sentiment']) * 30, 30)

        # Trade size (larger trades more suspicious)
        if trade.amount_mid:
            if trade.amount_mid > 500000:
                score += 20
            elif trade.amount_mid > 100000:
                score += 15
            elif trade.amount_mid > 50000:
                score += 10

        # Hype level
        hype_scores = {'low': 5, 'medium': 10, 'high': 20, 'extreme': 25}
        score += hype_scores.get(sentiment['hype_level'], 0)

        # Timing (if traded exactly at peak hype)
        # TODO: Implement peak detection

        return min(score, 100)


# Example usage
if __name__ == "__main__":
    # Initialize scraper
    scraper = RedditScraper(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_SECRET",
        user_agent="CongressionalTrading/1.0"
    )

    # Get trending tickers
    trending = scraper.get_trending_tickers(hours=24, min_mentions=10)

    print("📈 Trending on Reddit (last 24h):\n")
    for item in trending[:10]:
        print(f"{item['ticker']:6} | "
              f"Mentions: {item['mentions']:3} | "
              f"Sentiment: {item['avg_sentiment']:+.2f} | "
              f"Hype: {item['hype_level']:8}")

    # Scrape specific subreddit
    wsb_posts = scraper.scrape_subreddit('wallstreetbets', limit=50)
    print(f"\n🎰 Found {len(wsb_posts)} posts from r/wallstreetbets")

    # Aggregate ticker sentiment
    ticker_sentiment = scraper.aggregate_ticker_sentiment(wsb_posts)
    print(f"\n📊 Ticker sentiment for {len(ticker_sentiment)} symbols")
