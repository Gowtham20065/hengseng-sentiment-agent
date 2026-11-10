"""Event-driven backtesting engine for quant sentiment strategy with realistic fees."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class EventDrivenBacktester:
    """Simulates realistic chronological execution of sentiment-driven predictions.

    Features:
    - Market-tailored transaction costs (HKEX: 10-12 bps, KRX: 15-20 bps)
    - Realistic slippage modeling (5 bps)
    - Purged chronological trade execution
    - Risk metrics: Annualized Sharpe, Max Drawdown, Win Rate, Profit Factor, Alpha vs Buy & Hold
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        prob_threshold_high: float = 0.55,
        prob_threshold_low: float = 0.45,
        long_only: bool = False,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = (fee_bps + slippage_bps) / 10000.0
        self.prob_threshold_high = prob_threshold_high
        self.prob_threshold_low = prob_threshold_low
        self.long_only = long_only

    def backtest(self, predicted_df: pd.DataFrame, ticker: str = "") -> Dict[str, Any]:
        """Execute event-driven hourly backtest on predicted DataFrame."""
        if predicted_df.empty or len(predicted_df) < 2:
            return {"error": "Insufficient data to run backtest"}

        df = predicted_df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.sort_values(by="timestamp", ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)

        close = df["close"].astype(float).values
        n = len(df)

        # Extract or generate positions from predictions
        positions = np.zeros(n, dtype=float)
        probs = df.get("up_probability", pd.Series(np.full(n, 0.5))).values
        pred_rets = df.get("predicted_return", pd.Series(np.zeros(n))).values
        signals = df.get("signal", pd.Series(["HOLD"] * n)).values

        curr_pos = 0.0
        for i in range(n):
            sig = signals[i]
            p = probs[i]
            r = pred_rets[i]

            if sig == "BUY" or (p >= self.prob_threshold_high and r > 0):
                curr_pos = 1.0
            elif sig == "SELL" or (p <= self.prob_threshold_low and r < 0):
                curr_pos = 0.0 if self.long_only else -1.0
            else:
                # Retain existing position or stay flat
                pass
            positions[i] = curr_pos

        # Simulate execution bar by bar
        equity_strat = np.zeros(n, dtype=float)
        equity_bench = np.zeros(n, dtype=float)

        equity_strat[0] = self.initial_capital
        equity_bench[0] = self.initial_capital

        trade_returns: List[float] = []
        turnover_count = 0
        gross_profit = 0.0
        gross_loss = 0.0

        for t in range(n - 1):
            pos_t = positions[t]
            pos_prev = positions[t - 1] if t > 0 else 0.0

            # Price return over interval [t, t+1]
            ret_t1 = (close[t + 1] - close[t]) / close[t]

            # Turnover cost for position change
            turnover = abs(pos_t - pos_prev)
            cost = turnover * self.fee_rate
            if turnover > 0:
                turnover_count += 1

            # Strategy net return for bar
            strat_ret = (pos_t * ret_t1) - cost
            equity_strat[t + 1] = max(0.0, equity_strat[t] * (1.0 + strat_ret))
            equity_bench[t + 1] = equity_bench[0] * (close[t + 1] / close[0])

            if pos_t != 0.0:
                trade_returns.append(strat_ret)
                if strat_ret > 0:
                    gross_profit += strat_ret
                else:
                    gross_loss += abs(strat_ret)

        # Calculate Performance Metrics
        total_ret_strat = (equity_strat[-1] / equity_strat[0] - 1.0) * 100.0
        total_ret_bench = (equity_bench[-1] / equity_bench[0] - 1.0) * 100.0
        excess_return = total_ret_strat - total_ret_bench

        # Hourly returns for Sharpe calculation
        strat_returns = np.diff(equity_strat) / (equity_strat[:-1] + 1e-8)
        std_ret = float(np.std(strat_returns))
        mean_ret = float(np.mean(strat_returns))

        # Hourly annualized scaling: sqrt(252 trading days * 6.5 hours/day) ~ 40.47
        annual_factor = np.sqrt(252.0 * 6.5)
        sharpe = round(float((mean_ret / (std_ret + 1e-8)) * annual_factor), 2) if std_ret > 0 else 0.0

        # Maximum Drawdown
        cum_max = np.maximum.accumulate(equity_strat)
        drawdown = (equity_strat - cum_max) / (cum_max + 1e-8)
        max_drawdown = round(float(abs(np.min(drawdown))) * 100.0, 2)

        # Win Rate & Profit Factor
        winning_trades = sum(1 for r in trade_returns if r > 0)
        total_evaluated_trades = len(trade_returns)
        win_rate = round((winning_trades / total_evaluated_trades * 100.0), 2) if total_evaluated_trades > 0 else 0.0
        profit_factor = round(gross_profit / (gross_loss + 1e-8), 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

        df["position"] = positions
        df["strategy_equity"] = equity_strat.round(2)
        df["benchmark_equity"] = equity_bench.round(2)
        df["drawdown_pct"] = (drawdown * 100.0).round(2)

        results = {
            "ticker": ticker,
            "initial_capital": self.initial_capital,
            "final_equity": round(equity_strat[-1], 2),
            "total_return_pct": round(total_ret_strat, 2),
            "benchmark_return_pct": round(total_ret_bench, 2),
            "excess_return_pct": round(excess_return, 2),
            "annualized_sharpe": sharpe,
            "max_drawdown_pct": max_drawdown,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "num_trades": turnover_count,
            "equity_df": df,
        }
        return results
