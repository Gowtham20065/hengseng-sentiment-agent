"""Rigorous Out-of-Sample Walk-Forward Validation Engine for Time-Series Prediction."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.predictor import PriceDirectionPredictor


class WalkForwardValidator:
    """Rigorous expanding-window out-of-sample walk-forward validator (zero data leakage)."""

    def __init__(
        self,
        initial_train_pct: float = 0.55,
        step_pct: float = 0.10,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        allow_short: bool = True,
    ):
        self.initial_train_pct = initial_train_pct
        self.step_pct = step_pct
        self.fee_rate = (fee_bps + slippage_bps) / 10000.0
        self.allow_short = allow_short

    def validate_ticker(
        self, ticker: str, enriched_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute expanding-window walk-forward evaluation across sequential hourly bars."""
        if enriched_df.empty or len(enriched_df) < 6:
            raise ValueError(f"Insufficient bars to execute walk-forward validation for {ticker}")

        predictor = PriceDirectionPredictor()
        X, y_reg, y_clf, full_df = predictor.prepare_features_and_targets(enriched_df)

        n_samples = len(X)
        min_train = max(4, int(n_samples * self.initial_train_pct))
        step_size = max(1, int(n_samples * self.step_pct))

        # Containers for all out-of-sample evaluations
        oos_realized_returns: List[float] = []
        oos_predicted_returns: List[float] = []
        oos_realized_directions: List[int] = []
        oos_predicted_directions: List[int] = []
        oos_realized_prices: List[float] = []
        oos_predicted_prices: List[float] = []
        oos_sentiment_scores: List[float] = []
        oos_timestamps: List[Any] = []

        pointer = min_train
        while pointer < n_samples:
            train_idx = list(range(0, pointer))
            test_idx = list(range(pointer, min(pointer + step_size, n_samples)))

            # 1. Fit models on strictly historical expanding window
            X_train = X.iloc[train_idx]
            y_reg_train = y_reg.iloc[train_idx]
            y_clf_train = y_clf.iloc[train_idx]

            fold_predictor = PriceDirectionPredictor()
            fold_predictor.fit(X_train, y_reg_train, y_clf_train)

            # 2. Out-of-sample sequential step predictions
            for t in test_idx:
                curr_features = X.iloc[t]
                curr_price = float(full_df["close"].iloc[t])
                pred_result = fold_predictor.predict_n_plus_1(curr_features, curr_price)

                # Realized targets
                realized_ret = float(y_reg.iloc[t])
                realized_dir = int(y_clf.iloc[t])
                realized_price = float(full_df["close"].iloc[t + 1])
                sent_score = float(curr_features.get("social_weighted_sentiment", 0.0))

                pred_ret = float(pred_result["predicted_return_pct"] / 100.0)
                pred_dir = 1 if pred_result["predicted_direction"] == "UP" else 0
                pred_price = float(pred_result["predicted_price_n1"])

                oos_realized_returns.append(realized_ret)
                oos_predicted_returns.append(pred_ret)
                oos_realized_directions.append(realized_dir)
                oos_predicted_directions.append(pred_dir)
                oos_realized_prices.append(realized_price)
                oos_predicted_prices.append(pred_price)
                oos_sentiment_scores.append(sent_score)
                oos_timestamps.append(full_df["timestamp"].iloc[t])

            pointer += step_size

        if not oos_realized_returns:
            raise RuntimeError("Walk-forward validation yielded zero out-of-sample bars.")

        # --- Calculate Quant Metrics ---
        total_oos = len(oos_realized_returns)
        correct_directions = sum(
            1 for p, r in zip(oos_predicted_directions, oos_realized_directions) if p == r
        )
        directional_accuracy = round(float(correct_directions / total_oos * 100.0), 2)

        mae = round(float(mean_absolute_error(oos_realized_prices, oos_predicted_prices)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(oos_realized_prices, oos_predicted_prices))), 4)

        # Spearman Information Coefficient (Correlation of Sentiment/Pred with Realized Returns)
        ic_stat, _ = spearmanr(oos_predicted_returns, oos_realized_returns)
        ic = round(float(ic_stat), 4) if not np.isnan(ic_stat) else 0.0

        # --- Simulate Out-of-Sample Strategy vs Benchmark ---
        initial_capital = 100000.0
        equity_strat = [initial_capital]
        equity_bench = [initial_capital]
        curr_pos = 0.0

        for i in range(total_oos):
            p_dir = oos_predicted_directions[i]
            target_pos = 1.0 if p_dir == 1 else (-1.0 if self.allow_short else 0.0)
            ret_i = oos_realized_returns[i]

            turnover = abs(target_pos - curr_pos)
            cost = turnover * self.fee_rate
            strat_ret = (target_pos * ret_i) - cost

            prev_eq_s = equity_strat[-1]
            equity_strat.append(max(0.0, prev_eq_s * (1.0 + strat_ret)))

            prev_eq_b = equity_bench[-1]
            equity_bench.append(max(0.0, prev_eq_b * (1.0 + ret_i)))
            curr_pos = target_pos

        strat_cum_return = round(float((equity_strat[-1] / equity_strat[0] - 1.0) * 100.0), 2)
        bench_cum_return = round(float((equity_bench[-1] / equity_bench[0] - 1.0) * 100.0), 2)
        excess_return = round(float(strat_cum_return - bench_cum_return), 2)

        # Annualized Sharpe (hourly scaling)
        s_rets = np.diff(equity_strat) / (np.array(equity_strat[:-1]) + 1e-8)
        std_s = float(np.std(s_rets))
        mean_s = float(np.mean(s_rets))
        sharpe = (
            round(float((mean_s / (std_s + 1e-8)) * np.sqrt(252.0 * 6.5)), 2)
            if std_s > 0
            else 0.0
        )

        # Max Drawdown
        cum_max = np.maximum.accumulate(equity_strat)
        dd = (np.array(equity_strat) - cum_max) / (cum_max + 1e-8)
        max_dd = round(float(abs(np.min(dd)) * 100.0), 2)

        validation_dossier = {
            "ticker": ticker,
            "total_bars": len(full_df),
            "train_window_bars": min_train,
            "out_of_sample_bars": total_oos,
            "directional_accuracy": directional_accuracy,
            "information_coefficient": ic,
            "mae": mae,
            "rmse": rmse,
            "strategy_cumulative_return": strat_cum_return,
            "benchmark_cumulative_return": bench_cum_return,
            "excess_return": excess_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "out_of_sample_predictions": pd.DataFrame({
                "timestamp": oos_timestamps,
                "realized_price": oos_realized_prices,
                "predicted_price": oos_predicted_prices,
                "realized_direction": oos_realized_directions,
                "predicted_direction": oos_predicted_directions,
                "realized_return": oos_realized_returns,
                "predicted_return": oos_predicted_returns,
            }),
        }
        return validation_dossier

