"""Unit tests for TechnicalAnalyzer, Support/Resistance Engine, and Ticker Profiler."""

import numpy as np
import pandas as pd
import pytest

from src.processing.technicals import TechnicalAnalyzer


@pytest.fixture
def sample_price_df():
    """Create realistic synthetic hourly price dataframe."""
    np.random.seed(42)
    n = 30
    timestamps = pd.date_range("2026-11-01 02:00:00", periods=n, freq="h", tz="UTC")
    base_price = 400.0
    returns = np.random.normal(0.0005, 0.008, size=n)
    prices = base_price * np.cumprod(1.0 + returns)

    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.004, size=n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.004, size=n)))
    opens = prices * (1.0 + np.random.normal(0, 0.002, size=n))
    volumes = np.random.randint(50000, 1500000, size=n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens.round(2),
        "high": highs.round(2),
        "low": lows.round(2),
        "close": prices.round(2),
        "volume": volumes,
        "ticker": ["0700.HK"] * n,
        "social_mentions": np.random.randint(0, 8, size=n),
        "social_weighted_sentiment": np.random.uniform(-0.5, 0.8, size=n).round(4),
        "social_bull_ratio": np.random.uniform(0.2, 0.8, size=n).round(2),
        "social_bear_ratio": np.random.uniform(0.1, 0.5, size=n).round(2),
        "news_sentiment": np.random.uniform(-0.3, 0.6, size=n).round(4),
        "news_social_divergence": np.random.uniform(-0.4, 0.4, size=n).round(4),
        "sentiment_momentum": np.random.uniform(-0.2, 0.2, size=n).round(4),
        "mention_zscore": np.random.uniform(-1.0, 2.5, size=n).round(2),
    })
    df.set_index("timestamp", drop=False, inplace=True)
    return df


class TestTechnicalAnalyzer:
    def test_indicator_calculations(self, sample_price_df):
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(sample_price_df)

        for col in ["rsi_14", "macd", "macd_signal", "macd_hist", "atr_14", "volume_ratio"]:
            assert col in df.columns

        # Sanity checks
        assert (df["rsi_14"] >= 0.0).all() and (df["rsi_14"] <= 100.0).all()
        assert (df["atr_14"] > 0.0).all()
        assert (df["volume_ratio"] >= 0.0).all()

    def test_support_resistance_ordering(self, sample_price_df):
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(sample_price_df)
        curr_price = float(df["close"].iloc[-1])

        supports, resistances = analyzer.compute_support_resistance(df)

        s1 = supports["S1"]
        s2 = supports["S2"]
        r1 = resistances["R1"]
        r2 = resistances["R2"]

        # Strict quantitative ordering constraint:
        # S2 < S1 < current_price < R1 < R2
        assert s2 < s1, f"Expected S2 ({s2}) < S1 ({s1})"
        assert s1 < curr_price, f"Expected S1 ({s1}) < current_price ({curr_price})"
        assert curr_price < r1, f"Expected current_price ({curr_price}) < R1 ({r1})"
        assert r1 < r2, f"Expected R1 ({r1}) < R2 ({r2})"

    def test_profile_generation_structure(self, sample_price_df):
        analyzer = TechnicalAnalyzer()
        profile = analyzer.extract_ticker_profile("0700.HK", sample_price_df)

        required_keys = [
            "ticker",
            "asset_name",
            "current_price",
            "sentiment_score",
            "sentiment_label",
            "direction",
            "support_levels",
            "resistance_levels",
            "position_recommendation",
            "crowd_vs_smart_money",
            "market_microstructure_notes",
        ]
        for key in required_keys:
            assert key in profile, f"Missing key in profile: {key}"

        assert profile["ticker"] == "0700.HK"
        assert profile["asset_name"] == "Tencent Holdings"
        assert profile["sentiment_label"] in [
            "EXTREMELY BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "EXTREMELY BEARISH"
        ]
        assert profile["direction"] in ["BULLISH", "BEARISH", "NEUTRAL"]

        pos = profile["position_recommendation"]
        for p_key in ["action", "entry_price", "stop_loss", "take_profit", "risk_reward_ratio"]:
            assert p_key in pos
        assert pos["action"] in ["LONG", "SHORT", "FLAT / AVOID"]
        assert pos["risk_reward_ratio"] > 0

        csm = profile["crowd_vs_smart_money"]
        for c_key in ["retail_sentiment", "institutional_sentiment", "divergence", "alert"]:
            assert c_key in csm

        # HKEX CBBC microstructure note check
        assert "CBBC" in profile["market_microstructure_notes"]

    def test_krx_profile_microstructure(self, sample_price_df):
        analyzer = TechnicalAnalyzer()
        krx_df = sample_price_df.copy()
        krx_df["close"] = krx_df["close"] * 200  # Scale to KRW levels
        krx_df["ticker"] = "005930.KS"

        profile = analyzer.extract_ticker_profile("005930.KS", krx_df)
        assert profile["ticker"] == "005930.KS"
        assert profile["asset_name"] == "Samsung Electronics"
        assert "Naver" in profile["market_microstructure_notes"] or "RETAIL" in profile["market_microstructure_notes"]

