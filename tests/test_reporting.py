"""Unit tests for reporting, Plotly dashboard visualizer, and report generator."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.reporting.visualizer import QuantVisualizer
from src.reporting.report_generator import ReportGenerator


@pytest.fixture
def sample_profile_and_df(tmp_path):
    timestamps = pd.date_range("2026-11-01 02:00:00", periods=20, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.linspace(400, 410, 20),
        "high": np.linspace(402, 412, 20),
        "low": np.linspace(398, 408, 20),
        "close": np.linspace(401, 411, 20),
        "volume": np.random.randint(10000, 50000, 20),
        "social_weighted_sentiment": np.linspace(-0.2, 0.6, 20),
        "news_sentiment": np.linspace(0.1, 0.4, 20),
        "news_social_divergence": np.linspace(0.3, -0.2, 20),
        "macd": np.linspace(-0.5, 0.5, 20),
        "macd_signal": np.linspace(-0.4, 0.4, 20),
        "macd_hist": np.linspace(-0.1, 0.1, 20),
    })

    profile = {
        "ticker": "0700.HK",
        "asset_name": "Tencent Holdings",
        "current_price": 411.0,
        "sentiment_score": 0.60,
        "sentiment_label": "BULLISH",
        "direction": "BULLISH",
        "support_levels": {"S1": 405.0, "S2": 400.0, "confidence": 0.88},
        "resistance_levels": {"R1": 415.0, "R2": 420.0, "confidence": 0.88},
        "position_recommendation": {
            "action": "LONG",
            "entry_price": 411.0,
            "stop_loss": 403.0,
            "take_profit": 415.0,
            "risk_reward_ratio": 1.5,
        },
        "crowd_vs_smart_money": {
            "retail_sentiment": 0.60,
            "institutional_sentiment": 0.40,
            "divergence": -0.20,
            "alert": "NEUTRAL ALIGNMENT",
        },
        "market_microstructure_notes": "CBBC Neutral Zone near S1/R1.",
    }

    validation = {
        "ticker": "0700.HK",
        "out_of_sample_bars": 10,
        "directional_accuracy": 60.0,
        "information_coefficient": 0.25,
        "mae": 2.50,
        "rmse": 3.10,
        "strategy_cumulative_return": 4.50,
        "benchmark_cumulative_return": 2.10,
        "excess_return": 2.40,
        "sharpe_ratio": 1.65,
        "max_drawdown": 1.80,
    }

    return df, profile, validation


class TestReporting:
    def test_visualizer_creates_html_dashboard(self, sample_profile_and_df, tmp_path):
        df, profile, validation = sample_profile_and_df
        viz = QuantVisualizer(output_dir=str(tmp_path))

        out_file = viz.create_ticker_dashboard(
            ticker="0700.HK",
            enriched_df=df,
            profile=profile,
            validation_dossier=validation,
        )

        assert Path(out_file).exists()
        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "plotly" in content.lower()
        assert "0700.HK" in content
        assert "Tencent Holdings" in content

    def test_report_generator_markdown_output(self, sample_profile_and_df, tmp_path):
        _, profile, validation = sample_profile_and_df
        gen = ReportGenerator(output_dir=str(tmp_path))

        md_file = gen.generate_markdown_report(
            profiles=[profile],
            validations=[validation],
        )

        assert Path(md_file).exists()
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()

        assert "# Hang Seng & KRX Sentiment Price Prediction Agent" in text
        assert "10-Asset Master Quantitative Scorecard" in text
        assert "0700.HK" in text
        assert "Tencent Holdings" in text
        assert "CBBC" in text
        assert "Mathematical Formulations" in text

