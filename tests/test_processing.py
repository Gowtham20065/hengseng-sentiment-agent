"""Unit tests for Multilingual Sentiment Engine and TimeSeriesAligner."""

from datetime import datetime, timezone
import pandas as pd
import pytest

from src.processing.sentiment_engine import MultilingualSentimentEngine
from src.processing.aligner import TimeSeriesAligner


class TestMultilingualSentimentEngine:
    def test_cantonese_bullish_scoring(self):
        engine = MultilingualSentimentEngine()
        text = "0700呢期回購超預期，大戶掃貨穩如泰山，睇好破頂加倉！"
        res = engine.score_text(text, ticker="0700.HK", views=1000, likes=50)

        assert res["sentiment_score"] > 0.2
        assert res["direction"] == "BULLISH"
        assert res["confidence"] > 0.5
        assert res["retail_position"] in ["LONG", "SIDELINED"]
        assert res["engagement_weight"] > 0
        assert res["weighted_sentiment"] > 0

    def test_cantonese_bearish_bagholding_scoring(self):
        engine = MultilingualSentimentEngine()
        text = "高位接左大批蟹貨，今日大跌跌穿支持位，爆倉割肉救命！"
        res = engine.score_text(text, ticker="0700.HK", views=500, likes=10)

        assert res["sentiment_score"] < -0.2
        assert res["direction"] == "BEARISH"
        assert res["retail_position"] == "BAGHOLDING"
        assert res["weighted_sentiment"] < 0

    def test_korean_bullish_scoring(self):
        engine = MultilingualSentimentEngine()
        text = "삼성전자 HBM3E 수주 대박 떡상 가즈아! 상한가 직행 풀매수 탑승 완료!"
        res = engine.score_text(text, ticker="005930.KS", views=2000, likes=80)

        assert res["sentiment_score"] > 0.2
        assert res["direction"] == "BULLISH"
        assert res["retail_position"] == "LONG"
        assert res["weighted_sentiment"] > 0

    def test_korean_bearish_bagholding_scoring(self):
        engine = MultilingualSentimentEngine()
        text = "고점에 물린 개미들 곡소리 난다... 떡락에 하한가 맞고 손절 탈출합니다 존버 실패"
        res = engine.score_text(text, ticker="005930.KS", views=1200, likes=35)

        assert res["sentiment_score"] < -0.2
        assert res["direction"] == "BEARISH"
        assert res["retail_position"] == "BAGHOLDING"

    def test_engagement_weighting_sanity(self):
        engine = MultilingualSentimentEngine()

        # Base / Low engagement post
        w_low = engine.calculate_engagement_weight(
            views=50,
            likes=2,
            reposts=0,
            author_followers=20,
        )

        # High engagement viral post
        w_high = engine.calculate_engagement_weight(
            views=15000,
            likes=350,
            reposts=80,
            author_followers=5000,
        )

        assert w_low > 0
        assert w_high > w_low
        # Verify monotonic increase
        w_more_likes = engine.calculate_engagement_weight(views=50, likes=20, reposts=0, author_followers=20)
        assert w_more_likes > w_low


class TestTimeSeriesAligner:
    def test_zero_lookahead_alignment(self):
        aligner = TimeSeriesAligner()

        # Create 3 hourly bars: 02:00 UTC, 03:00 UTC, 04:00 UTC
        timestamps = [
            pd.Timestamp("2026-11-09 02:00:00", tz="UTC"),
            pd.Timestamp("2026-11-09 03:00:00", tz="UTC"),
            pd.Timestamp("2026-11-09 04:00:00", tz="UTC"),
        ]
        price_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": [400.0, 402.0, 401.5],
            "high": [403.0, 404.0, 402.5],
            "low": [399.5, 401.0, 400.0],
            "close": [402.0, 401.5, 401.0],
            "volume": [100000, 150000, 120000],
            "ticker": ["0700.HK", "0700.HK", "0700.HK"],
        })

        # Posts timestamps:
        # Post 1: 01:45 UTC (in window (01:00, 02:00] for Bar 0)
        # Post 2: 02:30 UTC (in window (02:00, 03:00] for Bar 1)
        # Post 3: 04:30 UTC (after Bar 2 - MUST NOT appear in any bar)
        posts = [
            {
                "timestamp": "2026-11-09T01:45:00Z",
                "sentiment_score": 0.8,
                "direction": "BULLISH",
                "engagement_weight": 10.0,
            },
            {
                "timestamp": "2026-11-09T02:30:00Z",
                "sentiment_score": -0.5,
                "direction": "BEARISH",
                "engagement_weight": 5.0,
            },
            {
                "timestamp": "2026-11-09T04:30:00Z",
                "sentiment_score": 0.9,
                "direction": "BULLISH",
                "engagement_weight": 20.0,
            },
        ]

        news = [
            {
                "published_date": "2026-11-09T01:15:00Z",
                "sentiment_score": 0.6,
            }
        ]

        enriched = aligner.align(price_df, posts, news)

        assert not enriched.empty
        assert len(enriched) == 3

        # All expected feature columns must be present
        for col in [
            "social_mentions",
            "social_weighted_sentiment",
            "social_bull_ratio",
            "social_bear_ratio",
            "news_sentiment",
            "news_social_divergence",
            "sentiment_momentum",
            "mention_zscore",
        ]:
            assert col in enriched.columns

        # Verify strict window aggregation:
        # Bar 0 (02:00 UTC) gets Post 1
        assert enriched.iloc[0]["social_mentions"] == 1
        assert enriched.iloc[0]["social_weighted_sentiment"] == pytest.approx(0.8, abs=1e-3)
        assert enriched.iloc[0]["social_bull_ratio"] == 1.0

        # Bar 1 (03:00 UTC) gets Post 2
        assert enriched.iloc[1]["social_mentions"] == 1
        assert enriched.iloc[1]["social_weighted_sentiment"] == pytest.approx(-0.5, abs=1e-3)
        assert enriched.iloc[1]["social_bear_ratio"] == 1.0

        # Bar 2 (04:00 UTC) gets no posts (Post 3 is after 04:00 UTC)
        assert enriched.iloc[2]["social_mentions"] == 0
        assert enriched.iloc[2]["social_weighted_sentiment"] == 0.0

        # News sentiment check (published prior to Bar 0, carried forward)
        assert enriched.iloc[0]["news_sentiment"] == pytest.approx(0.6, abs=1e-3)

        # Divergence check for Bar 0: news (0.6) - social (0.8) = -0.2
        assert enriched.iloc[0]["news_social_divergence"] == pytest.approx(-0.2, abs=1e-3)

