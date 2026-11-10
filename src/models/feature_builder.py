"""Feature builder for technical, sentiment, and interaction signals with forward targets."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from src.processing.technicals import TechnicalAnalyzer


class FeatureBuilder:
    """Computes technical indicators, sentiment interactions, and forward prediction targets."""

    DEFAULT_FEATURE_COLS = [
        # Price and volatility features
        "return_1h",
        "return_3h",
        "volatility_6h",
        "volume_zscore",
        "hl_spread",
        "rsi_14",
        "macd_hist",
        "atr_14",
        # Sentiment features
        "social_mentions",
        "social_weighted_sentiment",
        "social_bull_ratio",
        "social_bear_ratio",
        "news_sentiment",
        "news_social_divergence",
        "sentiment_momentum",
        "mention_zscore",
        # Interaction features
        "sentiment_x_volume",
        "divergence_x_momentum",
    ]

    def __init__(self):
        self.tech_analyzer = TechnicalAnalyzer()

    def build_features(
        self,
        aligned_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Enrich aligned market + sentiment DataFrame with technical features and forward targets."""
        if aligned_df.empty:
            return pd.DataFrame()

        # Compute technical indicators (RSI, MACD, ATR)
        df = self.tech_analyzer.compute_indicators(aligned_df)
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", drop=False, inplace=True)
        elif df.index.name == "timestamp":
            df["timestamp"] = pd.to_datetime(df.index, utc=True)

        df.sort_values(by="timestamp", ascending=True, inplace=True)

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # 1. Technical features (Backward-looking, no lookahead)
        df["return_1h"] = close.pct_change(1).fillna(0.0).round(6)
        df["return_3h"] = close.pct_change(3).fillna(0.0).round(6)
        df["volatility_6h"] = df["return_1h"].rolling(window=6, min_periods=1).std().fillna(0.0).round(6)

        vol_mean = volume.rolling(window=10, min_periods=1).mean()
        vol_std = volume.rolling(window=10, min_periods=1).std().fillna(0.0)
        df["volume_zscore"] = np.where(vol_std > 1e-4, (volume - vol_mean) / (vol_std + 1e-4), 0.0).round(4)
        df["hl_spread"] = np.where(close > 0, (high - low) / close, 0.0).round(6)

        # 2. Interaction Features
        soc_sentiment = df.get("social_weighted_sentiment", 0.0).astype(float)
        news_divergence = df.get("news_social_divergence", 0.0).astype(float)
        sent_momentum = df.get("sentiment_momentum", 0.0).astype(float)

        # Interaction 1: Sentiment amplified by unusual trading volume
        df["sentiment_x_volume"] = (soc_sentiment * df["volume_zscore"]).round(4)
        # Interaction 2: Divergence amplified by shifting retail sentiment momentum
        df["divergence_x_momentum"] = (news_divergence * sent_momentum).round(4)

        # 3. Forward Targets (Strictly future bars for supervision)
        df["forward_return_1h"] = close.pct_change(1).shift(-1).round(6)
        df["forward_return_3h"] = close.pct_change(3).shift(-3).round(6)
        df["target_direction"] = np.where(df["forward_return_1h"] > 0, 1, 0)

        return df

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        drop_target_nans: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Extract X (features), y_dir (direction target), y_ret (return target)."""
        cols = feature_cols or self.DEFAULT_FEATURE_COLS
        valid_cols = [c for c in cols if c in df.columns]

        data = df.copy()
        if drop_target_nans:
            data = data.dropna(subset=["forward_return_1h"])

        X = data[valid_cols].fillna(0.0)
        y_dir = data["target_direction"].astype(int)
        y_ret = data["forward_return_1h"].astype(float)

        return X, y_dir, y_ret

