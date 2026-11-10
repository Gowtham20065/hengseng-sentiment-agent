"""Unit tests for PriceDirectionPredictor and WalkForwardValidator."""

import numpy as np
import pandas as pd
import pytest

from src.models.predictor import PriceDirectionPredictor
from src.models.validation import WalkForwardValidator


@pytest.fixture
def sample_enriched_df():
    """Create synthetic enriched market and sentiment dataframe across 30 hourly bars."""
    np.random.seed(42)
    n = 30
    timestamps = pd.date_range("2026-11-01 02:00:00", periods=n, freq="h", tz="UTC")
    base_price = 405.0

    returns = np.random.normal(0.0004, 0.007, size=n)
    prices = base_price * np.cumprod(1.0 + returns)
    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.003, size=n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.003, size=n)))
    opens = prices * (1.0 + np.random.normal(0, 0.001, size=n))
    volumes = np.random.randint(80000, 1200000, size=n)

    # Lead-lag sentiment correlated with returns
    sentiment = np.roll(returns * 25.0, -1) + np.random.normal(0, 0.15, size=n)
    sentiment = np.clip(sentiment, -0.85, 0.85)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens.round(2),
        "high": highs.round(2),
        "low": lows.round(2),
        "close": prices.round(2),
        "volume": volumes,
        "ticker": ["0700.HK"] * n,
        "social_mentions": np.random.randint(1, 12, size=n),
        "social_weighted_sentiment": sentiment.round(4),
        "social_bull_ratio": np.where(sentiment > 0, 0.75, 0.25),
        "social_bear_ratio": np.where(sentiment < 0, 0.65, 0.15),
        "news_sentiment": (sentiment * 0.75).round(4),
        "news_social_divergence": np.random.uniform(-0.25, 0.25, size=n).round(4),
        "sentiment_momentum": pd.Series(sentiment).diff(3).fillna(0.0).round(4),
        "mention_zscore": np.random.uniform(-1.0, 2.2, size=n).round(2),
    })
    df.set_index("timestamp", drop=False, inplace=True)
    return df


class TestPriceDirectionPredictor:
    def test_feature_matrix_and_target_construction(self, sample_enriched_df):
        predictor = PriceDirectionPredictor()
        X, y_reg, y_clf, full_df = predictor.prepare_features_and_targets(sample_enriched_df)

        assert not X.empty
        assert len(X) == len(sample_enriched_df) - 1
        assert len(y_reg) == len(X)
        assert len(y_clf) == len(X)

        # Verify all feature columns are present
        for col in predictor.FEATURE_COLS:
            assert col in X.columns, f"Missing feature column {col}"

        # Target verification
        close_vals = full_df["close"].values
        expected_ret_0 = (close_vals[1] - close_vals[0]) / close_vals[0]
        assert y_reg.iloc[0] == pytest.approx(expected_ret_0, abs=1e-4)
        assert y_clf.iloc[0] == (1 if expected_ret_0 > 0 else 0)

    def test_fit_and_predict_n_plus_1(self, sample_enriched_df):
        predictor = PriceDirectionPredictor()
        X, y_reg, y_clf, full_df = predictor.prepare_features_and_targets(sample_enriched_df)

        predictor.fit(X, y_reg, y_clf)
        assert predictor.is_fitted
        assert len(predictor.feature_importances_) > 0

        current_feat = X.iloc[-1]
        current_price = float(full_df["close"].iloc[-1])

        pred = predictor.predict_n_plus_1(current_feat, current_price)

        required_keys = [
            "predicted_return_pct",
            "predicted_price_n1",
            "predicted_direction",
            "win_probability",
            "feature_importances",
        ]
        for key in required_keys:
            assert key in pred, f"Missing key {key} in predictor output"

        assert isinstance(pred["predicted_return_pct"], float)
        assert pred["predicted_price_n1"] > 0
        assert pred["predicted_direction"] in ["UP", "DOWN"]
        assert 0.0 <= pred["win_probability"] <= 1.0
        assert len(pred["feature_importances"]) > 0


class TestWalkForwardValidator:
    def test_walk_forward_execution_and_metrics(self, sample_enriched_df):
        validator = WalkForwardValidator(initial_train_pct=0.50, step_pct=0.15)
        dossier = validator.validate_ticker("0700.HK", sample_enriched_df)

        required_keys = [
            "ticker",
            "total_bars",
            "train_window_bars",
            "out_of_sample_bars",
            "directional_accuracy",
            "information_coefficient",
            "mae",
            "rmse",
            "strategy_cumulative_return",
            "benchmark_cumulative_return",
            "excess_return",
            "sharpe_ratio",
            "max_drawdown",
            "out_of_sample_predictions",
        ]
        for key in required_keys:
            assert key in dossier, f"Missing key {key} in validation dossier"

        assert dossier["ticker"] == "0700.HK"
        assert dossier["out_of_sample_bars"] > 0
        # Hit rate bound check
        assert 0.0 <= dossier["directional_accuracy"] <= 100.0
        # Error bound check
        assert dossier["mae"] >= 0.0
        assert dossier["rmse"] >= dossier["mae"] or pytest.approx(dossier["rmse"], abs=0.1) == dossier["mae"]
        assert dossier["max_drawdown"] >= 0.0

        # Out of sample dataframe verification
        oos_df = dossier["out_of_sample_predictions"]
        assert isinstance(oos_df, pd.DataFrame)
        assert len(oos_df) == dossier["out_of_sample_bars"]
        assert "predicted_price" in oos_df.columns
        assert "realized_price" in oos_df.columns
