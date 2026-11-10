"""Unit tests for data ingestion modules (Market data, Exa news, Apify social)."""

import os
from pathlib import Path
import pandas as pd
import pytest

from src.ingestion.market_data import MarketDataFetcher
from src.ingestion.exa_collector import ExaNewsCollector
from src.ingestion.apify_collector import ApifySocialCollector


@pytest.fixture
def temp_cache_dir(tmp_path):
    cache_d = tmp_path / "raw"
    cache_d.mkdir(parents=True, exist_ok=True)
    return str(cache_d)


class TestMarketDataFetcher:
    def test_fetch_bars_hkex(self, temp_cache_dir):
        fetcher = MarketDataFetcher(cache_dir=temp_cache_dir)
        ticker = "0700.HK"
        df = fetcher.fetch_bars(ticker=ticker, days=5, interval="1h")

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

        # Standardized column checks
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]
        assert list(df.columns) == expected_cols

        # Enforce UTC alignment
        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

        # Ticker column correctness
        assert (df["ticker"] == ticker).all()

        # Cache file creation check
        cache_file = Path(temp_cache_dir) / f"price_{ticker}.csv"
        assert cache_file.exists()

    def test_fetch_bars_krx(self, temp_cache_dir):
        fetcher = MarketDataFetcher(cache_dir=temp_cache_dir)
        ticker = "005930.KS"
        df = fetcher.fetch_bars(ticker=ticker, days=5, interval="1h")

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "timestamp" in df.columns
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert (df["open"] > 0).all()
        assert (df["ticker"] == ticker).all()


class TestExaNewsCollector:
    def test_news_structure_and_fallback(self, temp_cache_dir, monkeypatch):
        # Force mock/synthetic fallback to test guaranteed structural contract
        monkeypatch.setenv("USE_MOCK_DATA", "true")
        collector = ExaNewsCollector(cache_dir=temp_cache_dir)

        ticker = "9988.HK"
        articles = collector.fetch_news(ticker=ticker, asset_name="Alibaba Group", days=7)

        assert isinstance(articles, list)
        assert len(articles) > 0

        required_keys = {"id", "ticker", "published_date", "title", "text", "url", "score"}
        for art in articles:
            assert required_keys.issubset(art.keys())
            assert art["ticker"] == ticker
            assert isinstance(art["title"], str) and len(art["title"]) > 0
            assert isinstance(art["score"], (float, int))
            assert art["published_date"].endswith("Z") or "+00:00" in art["published_date"]

        # Cache file verification
        cache_file = Path(temp_cache_dir) / f"news_{ticker}.json"
        assert cache_file.exists()


class TestApifySocialCollector:
    def test_social_posts_hkex_mock_and_structure(self, temp_cache_dir, monkeypatch):
        monkeypatch.setenv("USE_MOCK_DATA", "true")
        collector = ApifySocialCollector(cache_dir=temp_cache_dir)

        ticker = "0700.HK"
        posts = collector.fetch_social_posts(ticker=ticker, keywords=["騰訊", "鵝廠"], days=7)

        assert isinstance(posts, list)
        assert len(posts) > 0

        required_keys = {
            "post_id",
            "ticker",
            "source",
            "timestamp",
            "text",
            "likes",
            "reposts",
            "views",
            "author_followers",
        }
        for post in posts:
            assert required_keys.issubset(post.keys())
            assert post["ticker"] == ticker
            assert post["source"] in ["LIHKG", "Futu Community"]
            assert post["likes"] >= 0
            assert post["views"] >= post["likes"]
            assert post["timestamp"].endswith("Z") or "+00:00" in post["timestamp"]

        # Cache file verification
        cache_file = Path(temp_cache_dir) / f"social_{ticker}.json"
        assert cache_file.exists()

    def test_social_posts_krx_mock_and_structure(self, temp_cache_dir, monkeypatch):
        monkeypatch.setenv("USE_MOCK_DATA", "true")
        collector = ApifySocialCollector(cache_dir=temp_cache_dir)

        ticker = "005930.KS"
        posts = collector.fetch_social_posts(ticker=ticker, keywords=["삼성전자", "삼전"], days=7)

        assert len(posts) > 0
        for post in posts:
            assert post["ticker"] == ticker
            assert post["source"] in ["Naver Finance Board", "Telegram KR Stocks"]
            assert len(post["text"]) > 0
            # Verify text contains Korean characters
            assert any("\uac00" <= ch <= "\ud7a3" for ch in post["text"])
