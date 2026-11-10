"""Interactive Plotly dashboard generator for price, sentiment dynamics, and out-of-sample backtest."""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class QuantVisualizer:
    """Generates publication-quality interactive 4-panel HTML charts for each asset."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_ticker_dashboard(
        self,
        ticker: str,
        enriched_df: pd.DataFrame,
        profile: Dict[str, Any],
        validation_dossier: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Create a self-contained 4-panel HTML dashboard with Plotly."""
        df = enriched_df.copy()
        if df.index.name == "timestamp":
            df.reset_index(drop=True, inplace=True)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.sort_values(by="timestamp", ascending=True, inplace=True)
        else:
            df["timestamp"] = pd.to_datetime(df.index, utc=True)

        asset_name = profile.get("asset_name", ticker)
        curr_price = profile.get("current_price", float(df["close"].iloc[-1]))
        s1 = profile.get("support_levels", {}).get("S1", curr_price * 0.98)
        s2 = profile.get("support_levels", {}).get("S2", curr_price * 0.96)
        r1 = profile.get("resistance_levels", {}).get("R1", curr_price * 1.02)
        r2 = profile.get("resistance_levels", {}).get("R2", curr_price * 1.04)
        direction = profile.get("direction", "NEUTRAL")

        # Extract n+1 prediction if available
        pred_price = profile.get("predicted_price_n1")
        if not pred_price and validation_dossier:
            oos_df = validation_dossier.get("out_of_sample_predictions")
            if oos_df is not None and not oos_df.empty:
                pred_price = float(oos_df["predicted_price"].iloc[-1])
        if not pred_price:
            pred_price = curr_price * 1.005

        # Initialize 4-panel layout
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.40, 0.20, 0.20, 0.20],
            subplot_titles=[
                f"<b>{asset_name} ({ticker}) — OHLCV Price Action & Support / Resistance Nodes</b>",
                "<b>Quant Sentiment Dynamics: Retail Buzz vs. Institutional Smart Money</b>",
                "<b>Technical Momentum & Trend (MACD & RSI 14)</b>",
                "<b>Out-of-Sample Walk-Forward Strategy Equity Curve vs. Benchmark</b>",
            ],
        )

        # ---------------- Panel 1: Candlesticks & S/R ----------------
        fig.add_trace(
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLCV",
                increasing_line_color="#00E676",
                decreasing_line_color="#FF1744",
            ),
            row=1,
            col=1,
        )

        # Horizontal Support lines
        fig.add_hline(
            y=s1,
            line_dash="dash",
            line_color="#00C853",
            line_width=1.5,
            annotation_text=f"S1: {s1}",
            annotation_position="bottom right",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=s2,
            line_dash="dot",
            line_color="#69F0AE",
            line_width=1.2,
            annotation_text=f"S2: {s2}",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

        # Horizontal Resistance lines
        fig.add_hline(
            y=r1,
            line_dash="dash",
            line_color="#D50000",
            line_width=1.5,
            annotation_text=f"R1: {r1}",
            annotation_position="top right",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=r2,
            line_dash="dot",
            line_color="#FF8A80",
            line_width=1.2,
            annotation_text=f"R2: {r2}",
            annotation_position="top right",
            row=1,
            col=1,
        )

        # Predicted n+1 Target Price Marker
        next_ts = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        fig.add_trace(
            go.Scatter(
                x=[next_ts],
                y=[pred_price],
                mode="markers+text",
                name=f"Predicted n+1 ({pred_price})",
                marker=dict(symbol="star", size=14, color="#FFD700", line=dict(width=2, color="#FFFFFF")),
                text=[f"Target: {pred_price} ({direction})"],
                textposition="top center",
                textfont=dict(color="#FFD700", size=11),
            ),
            row=1,
            col=1,
        )

        # ---------------- Panel 2: Sentiment Dynamics ----------------
        soc_sentiment = df.get("social_weighted_sentiment", pd.Series(np.zeros(len(df))))
        bar_colors = ["#00E676" if s >= 0 else "#FF1744" for s in soc_sentiment]

        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=soc_sentiment,
                name="Retail Weighted Sentiment",
                marker_color=bar_colors,
                opacity=0.75,
            ),
            row=2,
            col=1,
        )

        news_sentiment = df.get("news_sentiment", pd.Series(np.zeros(len(df))))
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=news_sentiment,
                mode="lines",
                name="Institutional News Sentiment",
                line=dict(color="#2979FF", width=2),
            ),
            row=2,
            col=1,
        )

        divergence = df.get("news_social_divergence", pd.Series(np.zeros(len(df))))
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=divergence,
                mode="lines",
                name="Divergence (Inst - Retail)",
                line=dict(color="#AA00FF", width=1.5, dash="dot"),
            ),
            row=2,
            col=1,
        )

        # ---------------- Panel 3: Technical Momentum (MACD & RSI) ----------------
        macd = df.get("macd", pd.Series(np.zeros(len(df))))
        macd_sig = df.get("macd_signal", pd.Series(np.zeros(len(df))))
        macd_hist = df.get("macd_hist", pd.Series(np.zeros(len(df))))

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=macd,
                mode="lines",
                name="MACD (12,26)",
                line=dict(color="#00E5FF", width=1.5),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=macd_sig,
                mode="lines",
                name="Signal (9)",
                line=dict(color="#FF9100", width=1.2),
            ),
            row=3,
            col=1,
        )
        hist_colors = ["#00E676" if h >= 0 else "#FF1744" for h in macd_hist]
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=macd_hist,
                name="MACD Hist",
                marker_color=hist_colors,
                opacity=0.5,
            ),
            row=3,
            col=1,
        )

        # ---------------- Panel 4: Out-of-Sample Strategy vs Benchmark ----------------
        if validation_dossier and "out_of_sample_predictions" in validation_dossier:
            oos_data = validation_dossier["out_of_sample_predictions"]
            if isinstance(oos_data, pd.DataFrame) and not oos_data.empty:
                init_cap = 100000.0
                strat_eq = [init_cap]
                bench_eq = [init_cap]
                oos_ts = oos_data.get("timestamp", df["timestamp"].iloc[-len(oos_data):])

                for _, row in oos_data.iterrows():
                    ret_r = float(row.get("realized_return", 0.0))
                    dir_p = int(row.get("predicted_direction", 1))
                    strat_ret = ret_r if dir_p == 1 else -ret_r
                    strat_eq.append(strat_eq[-1] * (1.0 + strat_ret))
                    bench_eq.append(bench_eq[-1] * (1.0 + ret_r))

                x_pts = list(oos_ts)
                if len(x_pts) < len(strat_eq):
                    x_pts = [x_pts[0] - pd.Timedelta(hours=1)] + x_pts

                fig.add_trace(
                    go.Scatter(
                        x=x_pts,
                        y=strat_eq,
                        mode="lines+markers",
                        name="Sentiment Strategy Equity",
                        line=dict(color="#00E676", width=2.5),
                    ),
                    row=4,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_pts,
                        y=bench_eq,
                        mode="lines",
                        name="Buy & Hold Benchmark",
                        line=dict(color="#9E9E9E", width=1.8, dash="dash"),
                    ),
                    row=4,
                    col=1,
                )
        else:
            # Baseline price path
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["close"] / df["close"].iloc[0] * 100000.0,
                    mode="lines",
                    name="Asset Equity (Indexed)",
                    line=dict(color="#00E676", width=2),
                ),
                row=4,
                col=1,
            )

        # Global Layout styling
        hit_rate_str = (
            f"Hit Rate: {validation_dossier['directional_accuracy']}% | Sharpe: {validation_dossier['sharpe_ratio']} | "
            f"MDD: {validation_dossier['max_drawdown']}%"
            if validation_dossier
            else f"Current Price: {curr_price} | Signal: {direction}"
        )

        fig.update_layout(
            template="plotly_dark",
            height=1000,
            title=dict(
                text=(
                    f"<b>CrowdWisdomTrading — Hang Seng & KRX Sentiment Price Prediction Agent</b><br>"
                    f"<span style='font-size:13px; color:#B0BEC5;'>Asset: {asset_name} ({ticker}) | {hit_rate_str}</span>"
                ),
                x=0.02,
                y=0.98,
            ),
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                bgcolor="rgba(0,0,0,0.4)",
            ),
            margin=dict(l=50, r=40, t=100, b=40),
        )

        safe_ticker = ticker.replace(".", "_")
        target_path = Path(output_path) if output_path else self.output_dir / f"{safe_ticker}_dashboard.html"
        fig.write_html(str(target_path), include_plotlyjs="cdn")
        return str(target_path)
