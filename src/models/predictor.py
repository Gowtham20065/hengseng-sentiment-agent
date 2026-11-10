"""Predictive ML Model for next-bar (n+1) price return and direction forecasting."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.processing.technicals import TechnicalAnalyzer


class PriceDirectionPredictor:
    """Predicts asset price and direction at bar n+1 using fused technical and sentiment signals.

    Dual Model Strategy:
    - Production: Regularized LightGBM (LGBMRegressor & LGBMClassifier)
    - Baseline: Standardized Ridge & Logistic Regression
    """

    FEATURE_COLS = [
        # Technical indicators
        "rsi_14",
        "macd",
        "macd_hist",
        "atr_14",
        "volume_ratio",
        "lag_return_1h",
        "lag_return_3h",
        "rolling_volatility_5h",
        # Quant sentiment signals
        "social_weighted_sentiment",
        "social_mentions",
        "social_bull_ratio",
        "social_bear_ratio",
        "news_sentiment",
        "news_social_divergence",
        "sentiment_momentum",
        "mention_zscore",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.tech_analyzer = TechnicalAnalyzer()
        self.scaler = StandardScaler()

        # Models
        self.lgb_reg = None
        self.lgb_clf = None
        self.baseline_reg = Ridge(alpha=1.0, random_state=random_state)
        self.baseline_clf = LogisticRegression(C=1.0, max_iter=250, random_state=random_state)

        self.feature_importances_: Dict[str, float] = {}
        self.is_fitted = False

    def prepare_features_and_targets(
        self, enriched_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """Generate technical & sentiment feature matrix and forward n+1 targets."""
        if enriched_df.empty:
            return pd.DataFrame(), pd.Series(), pd.Series(), pd.DataFrame()

        # Compute technical indicators (RSI, MACD, ATR, volume_ratio)
        df = self.tech_analyzer.compute_indicators(enriched_df)
        close = df["close"].astype(float)

        # Lag and volatility features
        df["lag_return_1h"] = close.pct_change(1).fillna(0.0).round(6)
        df["lag_return_3h"] = close.pct_change(3).fillna(0.0).round(6)
        df["rolling_volatility_5h"] = (
            df["lag_return_1h"].rolling(window=5, min_periods=1).std().fillna(0.0).round(6)
        )

        # Ensure all sentiment features exist
        for col in [
            "social_weighted_sentiment",
            "social_mentions",
            "social_bull_ratio",
            "social_bear_ratio",
            "news_sentiment",
            "news_social_divergence",
            "sentiment_momentum",
            "mention_zscore",
        ]:
            if col not in df.columns:
                df[col] = 0.0

        # Target Formulation: Strictly n+1 forward return and direction
        # Return_{n+1} = (Close_{n+1} - Close_n) / Close_n
        df["target_return_n1"] = close.pct_change(1).shift(-1).round(6)
        df["target_direction_n1"] = np.where(df["target_return_n1"] > 0, 1, 0)

        # Drop the last row with NaN target for training
        trainable = df.dropna(subset=["target_return_n1"]).copy()

        X = trainable[self.FEATURE_COLS].fillna(0.0)
        y_reg = trainable["target_return_n1"].astype(float)
        y_clf = trainable["target_direction_n1"].astype(int)

        return X, y_reg, y_clf, df

    def fit(self, X: pd.DataFrame, y_reg: pd.Series, y_clf: pd.Series) -> "PriceDirectionPredictor":
        """Train regularized LightGBM models and standardized baseline models."""
        if len(X) < 4:
            raise ValueError(f"Insufficient training samples: {len(X)}")

        X_cols = [c for c in self.FEATURE_COLS if c in X.columns]
        X_mat = X[X_cols].values

        # Scale features for baseline models
        X_scaled = self.scaler.fit_transform(X_mat)
        self.baseline_reg.fit(X_scaled, y_reg)
        try:
            self.baseline_clf.fit(X_scaled, y_clf)
        except Exception:
            pass

        # Production LightGBM with regularization to prevent overfitting on 10-day lookback
        min_child = max(2, int(len(X) * 0.08))
        self.lgb_reg = lgb.LGBMRegressor(
            n_estimators=35,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=min_child,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=self.random_state,
            verbose=-1,
        )
        self.lgb_reg.fit(X[X_cols], y_reg)

        self.lgb_clf = lgb.LGBMClassifier(
            n_estimators=35,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=min_child,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=self.random_state,
            verbose=-1,
        )
        self.lgb_clf.fit(X[X_cols], y_clf)

        # Aggregate feature importances
        importances = self.lgb_clf.feature_importances_
        if np.sum(importances) == 0:
            importances = self.lgb_reg.feature_importances_
        total_imp = float(np.sum(importances)) or 1.0

        self.feature_importances_ = {
            col: round(float(imp / total_imp), 4)
            for col, imp in zip(X_cols, importances)
        }

        self.is_fitted = True
        return self

    def predict_n_plus_1(
        self, current_bar_features: pd.Series, current_price: float
    ) -> Dict[str, Any]:
        """Predict price and probability direction at bar n+1.

        Returns:
        - predicted_return_pct: float (%)
        - predicted_price_n1: float
        - predicted_direction: "UP" or "DOWN"
        - win_probability: float [0.0, 1.0]
        - feature_importances: dict of top features
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor is not fitted. Call fit() before predict_n_plus_1.")

        feat_dict = {col: float(current_bar_features.get(col, 0.0)) for col in self.FEATURE_COLS}
        feat_df = pd.DataFrame([feat_dict])

        # LightGBM Regressor prediction
        pred_ret_lgb = float(self.lgb_reg.predict(feat_df)[0])
        # Baseline Ridge prediction
        scaled_feat = self.scaler.transform(feat_df.values)
        pred_ret_base = float(self.baseline_reg.predict(scaled_feat)[0])

        # Ensembled expected return (70% LightGBM + 30% Ridge)
        ensemble_ret = (0.70 * pred_ret_lgb) + (0.30 * pred_ret_base)
        # Clip extreme returns on 1h bar (max +/- 3.5%)
        ensemble_ret = max(-0.035, min(0.035, ensemble_ret))

        # Direction probability
        try:
            prob_up_lgb = float(self.lgb_clf.predict_proba(feat_df)[0, 1])
        except Exception:
            prob_up_lgb = 0.50

        try:
            prob_up_base = float(self.baseline_clf.predict_proba(scaled_feat)[0, 1])
        except Exception:
            prob_up_base = 0.50

        win_prob = (0.70 * prob_up_lgb) + (0.30 * prob_up_base)
        win_prob = max(0.01, min(0.99, win_prob))

        # Predicted target price: P_{n+1} = Close_n * (1 + predicted_return)
        decimals = 0 if current_price > 1000 else 2
        pred_price_n1 = round(current_price * (1.0 + ensemble_ret), decimals)
        pred_direction = "UP" if win_prob >= 0.50 or ensemble_ret > 0 else "DOWN"

        return {
            "predicted_return_pct": round(ensemble_ret * 100.0, 3),
            "predicted_price_n1": pred_price_n1,
            "predicted_direction": pred_direction,
            "win_probability": round(win_prob, 3),
            "feature_importances": dict(
                sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:8]
            ),
        }
