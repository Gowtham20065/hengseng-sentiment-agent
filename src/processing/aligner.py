"""Time-series alignment module ensuring zero lookahead bias between sentiment signals and price bars."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class TimeSeriesAligner:
    """Aligns asynchronous multilingual social posts and news articles with hourly OHLCV bars.

    Zero Lookahead-Bias Guarantee:
    For hourly bar B_t starting at t_start:
    Only posts/news published strictly in (t_prev_bar_start, t_start] are aggregated.
    """

    REQUIRED_SENTIMENT_COLUMNS = [
        "social_mentions",
        "social_weighted_sentiment",
        "social_bull_ratio",
        "social_bear_ratio",
        "news_sentiment",
        "news_social_divergence",
        "sentiment_momentum",
        "mention_zscore",
    ]

    def align(
        self,
        price_df: pd.DataFrame,
        scored_posts: List[Dict[str, Any]],
        scored_news: List[Dict[str, Any]],
        news_decay_hours: int = 24,
    ) -> pd.DataFrame:
        """Align scored sentiment data with price bars without future information leakage.

        Parameters:
        - price_df: Standardized OHLCV DataFrame with UTC timestamp index/column.
        - scored_posts: List of posts containing 'timestamp', 'sentiment_score', 'direction', 'engagement_weight'.
        - scored_news: List of news containing 'published_date', 'sentiment_score'.
        - news_decay_hours: Lookback window in hours for prevailing news sentiment prior to t_start.
        """
        if price_df.empty:
            return pd.DataFrame()

        df = price_df.copy()
        # Reset index if already named 'timestamp' to avoid ambiguity
        if df.index.name == "timestamp":
            df.reset_index(drop=True, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values(by="timestamp", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Parse posts into structured DataFrame
        post_records = []
        for p in scored_posts:
            ts = p.get("timestamp")
            if ts:
                try:
                    dt = pd.to_datetime(ts, utc=True)
                    post_records.append({
                        "timestamp": dt,
                        "sentiment_score": float(p.get("sentiment_score", 0.0)),
                        "direction": str(p.get("direction", "NEUTRAL")).upper(),
                        "weight": float(p.get("engagement_weight", 1.0)),
                    })
                except Exception:
                    pass

        posts_df = pd.DataFrame(post_records) if post_records else pd.DataFrame(
            columns=["timestamp", "sentiment_score", "direction", "weight"]
        )

        # Parse news into structured DataFrame
        news_records = []
        for n in scored_news:
            ts = n.get("published_date") or n.get("timestamp")
            if ts:
                try:
                    dt = pd.to_datetime(ts, utc=True)
                    news_records.append({
                        "timestamp": dt,
                        "sentiment_score": float(n.get("sentiment_score", 0.0)),
                    })
                except Exception:
                    pass

        news_df = pd.DataFrame(news_records) if news_records else pd.DataFrame(
            columns=["timestamp", "sentiment_score"]
        )

        bar_timestamps = df["timestamp"].tolist()
        num_bars = len(bar_timestamps)

        social_mentions = np.zeros(num_bars, dtype=int)
        social_weighted_sentiment = np.zeros(num_bars, dtype=float)
        social_bull_ratio = np.zeros(num_bars, dtype=float)
        social_bear_ratio = np.zeros(num_bars, dtype=float)
        news_sentiment = np.zeros(num_bars, dtype=float)

        last_known_news_sent = 0.0

        for i, t_start in enumerate(bar_timestamps):
            if i == 0:
                # For first bar, lookback 1 hour
                t_prev_bar_start = t_start - timedelta(hours=1)
            else:
                t_prev_bar_start = bar_timestamps[i - 1]

            # 1. Social Posts in (t_prev_bar_start, t_start]
            if not posts_df.empty:
                post_mask = (posts_df["timestamp"] > t_prev_bar_start) & (posts_df["timestamp"] <= t_start)
                window_posts = posts_df[post_mask]
                count = len(window_posts)
                social_mentions[i] = count

                if count > 0:
                    weights = window_posts["weight"].values
                    scores = window_posts["sentiment_score"].values
                    total_weight = np.sum(weights)
                    if total_weight > 0:
                        social_weighted_sentiment[i] = round(float(np.sum(scores * weights) / total_weight), 4)
                    else:
                        social_weighted_sentiment[i] = round(float(np.mean(scores)), 4)

                    bull_count = np.sum(window_posts["direction"].values == "BULLISH")
                    bear_count = np.sum(window_posts["direction"].values == "BEARISH")
                    social_bull_ratio[i] = round(float(bull_count / count), 4)
                    social_bear_ratio[i] = round(float(bear_count / count), 4)
                else:
                    social_weighted_sentiment[i] = 0.0
                    social_bull_ratio[i] = 0.0
                    social_bear_ratio[i] = 0.0
            else:
                social_mentions[i] = 0
                social_weighted_sentiment[i] = 0.0
                social_bull_ratio[i] = 0.0
                social_bear_ratio[i] = 0.0

            # 2. News Sentiment strictly prior to t_start (within lookback window)
            if not news_df.empty:
                # Immediate bar news
                news_immediate = (news_df["timestamp"] > t_prev_bar_start) & (news_df["timestamp"] <= t_start)
                win_news = news_df[news_immediate]

                if not win_news.empty:
                    last_known_news_sent = float(win_news["sentiment_score"].mean())
                    news_sentiment[i] = round(last_known_news_sent, 4)
                else:
                    # Prevailing news sentiment in last news_decay_hours prior to t_start
                    lookback_cutoff = t_start - timedelta(hours=news_decay_hours)
                    prev_news_mask = (news_df["timestamp"] > lookback_cutoff) & (news_df["timestamp"] <= t_start)
                    recent_news = news_df[prev_news_mask]
                    if not recent_news.empty:
                        last_known_news_sent = float(recent_news["sentiment_score"].mean())
                        news_sentiment[i] = round(last_known_news_sent, 4)
                    else:
                        news_sentiment[i] = round(last_known_news_sent, 4)
            else:
                news_sentiment[i] = 0.0

        # Assign feature columns
        df["social_mentions"] = social_mentions
        df["social_weighted_sentiment"] = social_weighted_sentiment
        df["social_bull_ratio"] = social_bull_ratio
        df["social_bear_ratio"] = social_bear_ratio
        df["news_sentiment"] = news_sentiment

        # Institutional vs Retail Divergence metric
        df["news_social_divergence"] = (df["news_sentiment"] - df["social_weighted_sentiment"]).round(4)

        # Sentiment Momentum: change in social_weighted_sentiment over previous 3 bars
        df["sentiment_momentum"] = df["social_weighted_sentiment"].diff(3).fillna(0.0).round(4)

        # Mention Z-Score: rolling Z-score of social mention volume
        roll_mean = df["social_mentions"].rolling(window=10, min_periods=1).mean()
        roll_std = df["social_mentions"].rolling(window=10, min_periods=1).std().fillna(0.0)
        df["mention_zscore"] = np.where(
            roll_std > 1e-5,
            (df["social_mentions"] - roll_mean) / (roll_std + 1e-5),
            0.0
        ).round(4)

        # Set index to UTC timestamp
        df.set_index("timestamp", drop=False, inplace=True)
        df.index.name = "timestamp"

        return df
