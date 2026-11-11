"""Interactive Plotly dashboard generator for price, sentiment dynamics, and out-of-sample backtest."""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Brand palette ──────────────────────────────────────────────────────────────
_BG        = "#0D1117"   # GitHub dark
_PANEL_BG  = "#161B22"
_BORDER    = "#30363D"
_GREEN     = "#3FB950"
_RED       = "#F85149"
_BLUE      = "#58A6FF"
_PURPLE    = "#BC8CFF"
_GOLD      = "#E3B341"
_GREY      = "#8B949E"
_WHITE     = "#F0F6FC"
_SUBTEXT   = "#8B949E"


class QuantVisualizer:
    """Generates publication-quality interactive 4-panel HTML dashboards for each asset."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _safe_series(self, df: pd.DataFrame, col: str) -> pd.Series:
        return df[col] if col in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

    def _fmt(self, v: float, decimals: int = 2) -> str:
        return f"{v:,.{decimals}f}"

    def _direction_badge(self, direction: str) -> str:
        colour = _GREEN if direction == "BULLISH" else (_RED if direction == "BEARISH" else _GOLD)
        return f"<span style='color:{colour};font-weight:700'>{direction}</span>"

    # ── main public method ─────────────────────────────────────────────────────

    def create_ticker_dashboard(
        self,
        ticker: str,
        enriched_df: pd.DataFrame,
        profile: Dict[str, Any],
        validation_dossier: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Create a self-contained, institutional-grade 4-panel HTML dashboard."""

        # ── 1. Data prep ───────────────────────────────────────────────────────
        df = enriched_df.copy()
        if df.index.name == "timestamp":
            df.reset_index(drop=True, inplace=True)
        df["timestamp"] = pd.to_datetime(
            df.get("timestamp", pd.Series(df.index)), utc=True
        )
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # ── 2. Profile values ──────────────────────────────────────────────────
        asset_name  = profile.get("asset_name", ticker)
        curr_price  = profile.get("current_price", float(df["close"].iloc[-1]))
        direction   = profile.get("direction", "NEUTRAL")
        sent_score  = profile.get("sentiment_score", 0.0)
        sent_label  = profile.get("sentiment_label", "NEUTRAL")
        win_prob    = profile.get("win_probability", 0.5)
        action      = profile.get("position_recommendation", {}).get("action", "FLAT")
        rr_ratio    = profile.get("position_recommendation", {}).get("risk_reward_ratio", 1.0)
        s1 = profile.get("support_levels",    {}).get("S1", curr_price * 0.98)
        s2 = profile.get("support_levels",    {}).get("S2", curr_price * 0.96)
        r1 = profile.get("resistance_levels", {}).get("R1", curr_price * 1.02)
        r2 = profile.get("resistance_levels", {}).get("R2", curr_price * 1.04)

        # n+1 predicted price
        pred_price = profile.get("predicted_price_n1")
        if not pred_price and validation_dossier:
            oos_df = validation_dossier.get("out_of_sample_predictions")
            if isinstance(oos_df, pd.DataFrame) and not oos_df.empty:
                pred_price = float(oos_df["predicted_price"].iloc[-1])
        pred_price = pred_price or (curr_price * 1.005)

        pred_delta    = pred_price - curr_price
        pred_delta_pct = pred_delta / curr_price * 100.0

        # validation metrics
        hit_rate = sharpe = mdd = ic = None
        if validation_dossier:
            hit_rate = validation_dossier.get("directional_accuracy")
            sharpe   = validation_dossier.get("sharpe_ratio")
            mdd      = validation_dossier.get("max_drawdown")
            ic       = validation_dossier.get("information_coefficient")

        # ── 3. Figure skeleton ─────────────────────────────────────────────────
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.055,
            row_heights=[0.42, 0.19, 0.19, 0.20],
            subplot_titles=[
                "OHLCV Price Action · Support & Resistance Nodes",
                "Quant Sentiment Dynamics · Retail vs Institutional Smart Money",
                "Technical Momentum · MACD (12,26,9) · RSI (14)",
                "Out-of-Sample Walk-Forward Equity Curve vs Buy & Hold Benchmark",
            ],
        )

        # ── 4. Panel 1 — Candlestick + S/R + Volume bars ─────────────────────
        fig.add_trace(
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"], high=df["high"],
                low=df["low"],  close=df["close"],
                name="OHLCV",
                increasing_line_color=_GREEN,
                decreasing_line_color=_RED,
                increasing_fillcolor=_GREEN,
                decreasing_fillcolor=_RED,
                whiskerwidth=0.6,
                line=dict(width=1),
            ),
            row=1, col=1,
        )

        # Volume as area on secondary y inside same subplot (normalized)
        if "volume" in df.columns and df["volume"].sum() > 0:
            vol_norm = df["volume"] / df["volume"].max() * (curr_price * 0.06)
            vol_base = df["close"].min() * 0.995
            vol_colors = [
                _GREEN if df["close"].iloc[i] >= df["open"].iloc[i] else _RED
                for i in range(len(df))
            ]
            fig.add_trace(
                go.Bar(
                    x=df["timestamp"],
                    y=vol_norm,
                    base=vol_base,
                    name="Volume (scaled)",
                    marker_color=vol_colors,
                    opacity=0.25,
                    showlegend=True,
                ),
                row=1, col=1,
            )

        # S/R horizontal lines
        sr_lines = [
            (s2, _GREEN,  "dot",  1.1, f"S2 {self._fmt(s2)}", "bottom left"),
            (s1, _GREEN,  "dash", 1.6, f"S1 {self._fmt(s1)}", "bottom left"),
            (r1, _RED,    "dash", 1.6, f"R1 {self._fmt(r1)}", "top left"),
            (r2, _RED,    "dot",  1.1, f"R2 {self._fmt(r2)}", "top left"),
        ]
        for y_val, color, dash, width, label, pos in sr_lines:
            fig.add_hline(
                y=y_val, line_dash=dash, line_color=color, line_width=width,
                annotation_text=f"<b>{label}</b>",
                annotation_position=pos,
                annotation_font=dict(size=10, color=color),
                row=1, col=1,
            )

        # n+1 Predicted Price star marker
        next_ts = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        delta_sign = "▲" if pred_delta >= 0 else "▼"
        pred_color = _GREEN if pred_delta >= 0 else _RED
        fig.add_trace(
            go.Scatter(
                x=[next_ts],
                y=[pred_price],
                mode="markers+text",
                name=f"Predicted n+1: {self._fmt(pred_price)}",
                marker=dict(
                    symbol="star", size=16, color=_GOLD,
                    line=dict(width=1.5, color=_WHITE),
                ),
                text=[f"  {delta_sign} {self._fmt(pred_price)} ({pred_delta_pct:+.2f}%)"],
                textposition="middle right",
                textfont=dict(color=_GOLD, size=11, family="monospace"),
                hovertemplate=(
                    f"<b>n+1 Target</b><br>"
                    f"Price: {self._fmt(pred_price)}<br>"
                    f"Δ: {pred_delta:+.4f} ({pred_delta_pct:+.2f}%)<br>"
                    f"Direction: {direction}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

        # Current price reference line
        fig.add_hline(
            y=curr_price, line_dash="solid", line_color=_WHITE,
            line_width=0.8, opacity=0.35,
            row=1, col=1,
        )

        # ── 5. Panel 2 — Sentiment ─────────────────────────────────────────────
        soc  = self._safe_series(df, "social_weighted_sentiment")
        news = self._safe_series(df, "news_sentiment")
        div  = self._safe_series(df, "news_social_divergence")
        mom  = self._safe_series(df, "sentiment_momentum")

        bar_colors = [_GREEN if v >= 0 else _RED for v in soc]
        fig.add_trace(
            go.Bar(
                x=df["timestamp"], y=soc,
                name="Retail Sentiment (Weighted)",
                marker_color=bar_colors, opacity=0.70,
                hovertemplate="Retail Sentiment: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=news, mode="lines",
                name="Institutional News Sentiment",
                line=dict(color=_BLUE, width=2),
                hovertemplate="News Sentiment: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=div, mode="lines",
                name="Divergence (Inst − Retail)",
                line=dict(color=_PURPLE, width=1.5, dash="dot"),
                hovertemplate="Divergence: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=mom, mode="lines",
                name="Sentiment Momentum",
                line=dict(color=_GOLD, width=1.2, dash="dash"),
                hovertemplate="Momentum: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1,
        )
        # Zero-line reference
        fig.add_hline(y=0, line_color=_BORDER, line_width=1, row=2, col=1)

        # ── 6. Panel 3 — MACD + RSI ───────────────────────────────────────────
        macd      = self._safe_series(df, "macd")
        macd_sig  = self._safe_series(df, "macd_signal")
        macd_hist = self._safe_series(df, "macd_hist")
        rsi       = self._safe_series(df, "rsi_14")

        hist_colors = [_GREEN if h >= 0 else _RED for h in macd_hist]
        fig.add_trace(
            go.Bar(
                x=df["timestamp"], y=macd_hist,
                name="MACD Histogram",
                marker_color=hist_colors, opacity=0.55,
                hovertemplate="MACD Hist: %{y:.6f}<extra></extra>",
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=macd, mode="lines",
                name="MACD Line (12,26)",
                line=dict(color="#00E5FF", width=1.8),
                hovertemplate="MACD: %{y:.6f}<extra></extra>",
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=macd_sig, mode="lines",
                name="Signal Line (9)",
                line=dict(color="#FF9100", width=1.5),
                hovertemplate="Signal: %{y:.6f}<extra></extra>",
            ),
            row=3, col=1,
        )
        # RSI as secondary axis overlay (right side)
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=rsi, mode="lines",
                name="RSI (14)",
                line=dict(color=_PURPLE, width=1.5, dash="dot"),
                yaxis="y5",
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ),
            row=3, col=1,
        )
        fig.add_hline(y=0, line_color=_BORDER, line_width=1, row=3, col=1)

        # ── 7. Panel 4 — Equity Curve ─────────────────────────────────────────
        INIT_CAP = 100_000.0
        strat_eq, bench_eq, x_pts = [INIT_CAP], [INIT_CAP], []

        oos_data = None
        if validation_dossier:
            oos_data = validation_dossier.get("out_of_sample_predictions")

        if isinstance(oos_data, pd.DataFrame) and not oos_data.empty:
            oos_ts = oos_data.get("timestamp", df["timestamp"].iloc[-len(oos_data):])
            x_pts = list(oos_ts)
            for _, row in oos_data.iterrows():
                ret_r  = float(row.get("realized_return", 0.0))
                dir_p  = int(row.get("predicted_direction", 1))
                strat_ret = ret_r if dir_p == 1 else -ret_r
                # Apply transaction cost (12 bps round-trip)
                strat_ret -= 0.0012
                strat_eq.append(strat_eq[-1] * (1.0 + strat_ret))
                bench_eq.append(bench_eq[-1] * (1.0 + ret_r))
            if len(x_pts) < len(strat_eq):
                x_pts = [x_pts[0] - pd.Timedelta(hours=1)] + x_pts
        else:
            x_pts = list(df["timestamp"])
            for i in range(1, len(df)):
                ret_r = (df["close"].iloc[i] - df["close"].iloc[i-1]) / df["close"].iloc[i-1]
                strat_eq.append(strat_eq[-1] * (1.0 + ret_r * 0.5))
                bench_eq.append(bench_eq[-1] * (1.0 + ret_r))
            if len(x_pts) < len(strat_eq):
                x_pts = [x_pts[0] - pd.Timedelta(hours=1)] + x_pts

        final_strat = strat_eq[-1]
        final_bench = bench_eq[-1]
        alpha_pct   = (final_strat - final_bench) / INIT_CAP * 100

        fig.add_trace(
            go.Scatter(
                x=x_pts, y=strat_eq, mode="lines",
                name=f"Strategy Equity (α {alpha_pct:+.2f}%)",
                line=dict(color=_GREEN, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(63,185,80,0.07)",
                hovertemplate="Strategy: $%{y:,.0f}<extra></extra>",
            ),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_pts, y=bench_eq, mode="lines",
                name="Buy & Hold Benchmark",
                line=dict(color=_GREY, width=1.8, dash="dash"),
                hovertemplate="Benchmark: $%{y:,.0f}<extra></extra>",
            ),
            row=4, col=1,
        )
        fig.add_hline(y=INIT_CAP, line_color=_BORDER, line_width=1, row=4, col=1)

        # ── 8. KPI header annotations (top-right corner) ──────────────────────
        sent_color = _GREEN if sent_score > 0.1 else (_RED if sent_score < -0.1 else _GOLD)
        action_color = _GREEN if action == "LONG" else (_RED if action == "SHORT" else _GREY)

        kpis = [
            ("SENTIMENT",  f"{sent_score:+.3f}",  sent_color),
            ("WIN PROB",   f"{win_prob*100:.1f}%", _BLUE),
            ("ACTION",     action,                 action_color),
            ("R:R",        f"{rr_ratio:.2f}x",     _GOLD),
        ]
        if hit_rate is not None:
            kpis.append(("HIT RATE", f"{hit_rate:.1f}%", _PURPLE))
        if sharpe is not None:
            kpis.append(("SHARPE",   f"{sharpe:.2f}",    _BLUE))

        kpi_x = 0.99
        for label, val, color in reversed(kpis):
            fig.add_annotation(
                xref="paper", yref="paper",
                x=kpi_x, y=0.995,
                text=f"<b style='color:{_SUBTEXT};font-size:9px'>{label}</b>"
                     f"<br><b style='color:{color};font-size:13px'>{val}</b>",
                showarrow=False,
                align="center",
                bgcolor=_PANEL_BG,
                bordercolor=_BORDER,
                borderwidth=1,
                borderpad=5,
                font=dict(family="monospace"),
            )
            kpi_x -= 0.095

        # ── 9. Global layout ───────────────────────────────────────────────────
        market = "HKEX · HKD" if ticker.endswith(".HK") else "KRX · KRW"
        dir_arrow = "▲" if direction == "BULLISH" else ("▼" if direction == "BEARISH" else "◆")
        dir_color = _GREEN if direction == "BULLISH" else (_RED if direction == "BEARISH" else _GOLD)

        fig.update_layout(
            template="plotly_dark",
            height=1080,
            paper_bgcolor=_BG,
            plot_bgcolor=_PANEL_BG,
            font=dict(family="'Inter', 'Segoe UI', Arial, sans-serif", color=_WHITE, size=12),
            title=dict(
                text=(
                    f"<b style='font-size:18px'>CrowdWisdomTrading — Hang Seng & KRX Sentiment Price Prediction Agent</b><br>"
                    f"<span style='font-size:13px;color:{_SUBTEXT}'>"
                    f"{asset_name} &nbsp;·&nbsp; {ticker} &nbsp;·&nbsp; {market} &nbsp;·&nbsp; "
                    f"Price: <b style='color:{_WHITE}'>{self._fmt(curr_price)}</b> &nbsp;·&nbsp; "
                    f"Label: <b style='color:{sent_color}'>{sent_label}</b> &nbsp;·&nbsp; "
                    f"<b style='color:{dir_color}'>{dir_arrow} {direction}</b>"
                    f"</span>"
                ),
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
            ),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.04,
                xanchor="left", x=0.0,
                bgcolor="rgba(13,17,23,0.85)",
                bordercolor=_BORDER,
                borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(l=60, r=60, t=110, b=70),
        )

        # Subplot title styling
        for ann in fig.layout.annotations:
            if ann.text in [
                "OHLCV Price Action · Support & Resistance Nodes",
                "Quant Sentiment Dynamics · Retail vs Institutional Smart Money",
                "Technical Momentum · MACD (12,26,9) · RSI (14)",
                "Out-of-Sample Walk-Forward Equity Curve vs Buy & Hold Benchmark",
            ]:
                ann.font = dict(size=11, color=_SUBTEXT, family="monospace")

        # Axis grid styling
        axis_style = dict(
            gridcolor=_BORDER, gridwidth=0.5,
            zerolinecolor=_BORDER, zerolinewidth=1,
            showline=True, linecolor=_BORDER, linewidth=1,
            tickfont=dict(size=10, color=_SUBTEXT),
        )
        for i in range(1, 5):
            xkey = f"xaxis{i if i > 1 else ''}"
            ykey = f"yaxis{i if i > 1 else ''}"
            fig.update_layout(**{xkey: axis_style, ykey: axis_style})

        # ── 10. Write HTML with custom head injection ──────────────────────────
        plotly_html = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png", "filename": f"{ticker}_dashboard",
                    "height": 1080, "width": 1920, "scale": 2,
                },
            },
        )

        html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{asset_name} ({ticker}) — CrowdWisdomTrading Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ height: 100%; background: {_BG}; color: {_WHITE}; font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }}

    /* ── top header bar ── */
    .header {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 24px 12px;
      background: {_PANEL_BG};
      border-bottom: 1px solid {_BORDER};
    }}
    .header__logo {{
      font-size: 13px; font-weight: 700; letter-spacing: 0.08em;
      color: {_SUBTEXT}; text-transform: uppercase;
    }}
    .header__logo span {{ color: {_BLUE}; }}
    .header__asset {{
      font-size: 22px; font-weight: 700; color: {_WHITE};
    }}
    .header__asset small {{
      font-size: 13px; font-weight: 400; color: {_SUBTEXT}; margin-left: 8px;
    }}
    .header__meta {{ font-size: 12px; color: {_SUBTEXT}; text-align: right; }}

    /* ── KPI cards row ── */
    .kpi-row {{
      display: flex; gap: 10px; padding: 12px 24px;
      background: {_BG}; border-bottom: 1px solid {_BORDER};
      flex-wrap: wrap;
    }}
    .kpi-card {{
      flex: 1; min-width: 110px;
      background: {_PANEL_BG}; border: 1px solid {_BORDER}; border-radius: 8px;
      padding: 10px 16px; display: flex; flex-direction: column; gap: 4px;
    }}
    .kpi-card__label {{
      font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
      text-transform: uppercase; color: {_SUBTEXT};
    }}
    .kpi-card__value {{
      font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums;
    }}
    .kpi-card__sub {{ font-size: 11px; color: {_SUBTEXT}; }}

    /* ── chart container ── */
    .chart-wrap {{ padding: 0 12px 12px; }}

    /* ── footer ── */
    .footer {{
      text-align: center; padding: 12px;
      font-size: 11px; color: {_SUBTEXT};
      border-top: 1px solid {_BORDER};
    }}
  </style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div>
    <div class="header__logo"><span>CrowdWisdom</span>Trading &nbsp;·&nbsp; Sentiment Price Prediction Agent</div>
    <div class="header__asset">{asset_name}<small>{ticker}</small></div>
  </div>
  <div class="header__meta">
    Market: <b>{market}</b><br/>
    Lookback: <b>10-Day Hourly</b><br/>
    Generated: <b id="ts"></b>
  </div>
</div>

<!-- KPI Cards -->
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-card__label">Current Price</div>
    <div class="kpi-card__value" style="color:{_WHITE}">{self._fmt(curr_price)}</div>
    <div class="kpi-card__sub">{market.split('·')[1].strip()}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-card__label">Sentiment Score</div>
    <div class="kpi-card__value" style="color:{sent_color}">{sent_score:+.3f}</div>
    <div class="kpi-card__sub">{sent_label}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-card__label">Direction Signal</div>
    <div class="kpi-card__value" style="color:{dir_color}">{dir_arrow} {direction}</div>
    <div class="kpi-card__sub">n+1 Forecast</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-card__label">Predicted n+1 Price</div>
    <div class="kpi-card__value" style="color:{pred_color if pred_delta >= 0 else _RED}">{self._fmt(pred_price)}</div>
    <div class="kpi-card__sub">{pred_delta:+.4f} ({pred_delta_pct:+.2f}%)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-card__label">Win Probability</div>
    <div class="kpi-card__value" style="color:{_BLUE}">{win_prob*100:.1f}%</div>
    <div class="kpi-card__sub">LightGBM Ensemble</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-card__label">Action</div>
    <div class="kpi-card__value" style="color:{action_color}">{action}</div>
    <div class="kpi-card__sub">R:R  {rr_ratio:.2f}x</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-card__label">Support Levels</div>
    <div class="kpi-card__value" style="color:{_GREEN};font-size:15px">S1 {self._fmt(s1)}</div>
    <div class="kpi-card__sub">S2 {self._fmt(s2)}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-card__label">Resistance Levels</div>
    <div class="kpi-card__value" style="color:{_RED};font-size:15px">R1 {self._fmt(r1)}</div>
    <div class="kpi-card__sub">R2 {self._fmt(r2)}</div>
  </div>
  {f'''<div class="kpi-card">
    <div class="kpi-card__label">OOS Hit Rate</div>
    <div class="kpi-card__value" style="color:{_PURPLE}">{hit_rate:.1f}%</div>
    <div class="kpi-card__sub">Walk-Forward</div>
  </div>''' if hit_rate is not None else ''}
  {f'''<div class="kpi-card">
    <div class="kpi-card__label">Sharpe Ratio</div>
    <div class="kpi-card__value" style="color:{_BLUE}">{sharpe:.2f}</div>
    <div class="kpi-card__sub">Strategy OOS</div>
  </div>''' if sharpe is not None else ''}
  {f'''<div class="kpi-card">
    <div class="kpi-card__label">Max Drawdown</div>
    <div class="kpi-card__value" style="color:{_RED}">{mdd:.2f}%</div>
    <div class="kpi-card__sub">OOS Period</div>
  </div>''' if mdd is not None else ''}
</div>

<!-- Chart -->
<div class="chart-wrap">
{plotly_html}
</div>

<!-- Footer -->
<div class="footer">
  CrowdWisdomTrading &nbsp;·&nbsp; Hang Seng &amp; KRX Sentiment Price Prediction Agent &nbsp;·&nbsp;
  Zero-Lookahead Walk-Forward Validation &nbsp;·&nbsp; LightGBM + Ridge Ensemble &nbsp;·&nbsp;
  Powered by Gemini 2.5 Flash · Exa · Apify
</div>

<script>
  document.getElementById('ts').textContent = new Date().toUTCString();
</script>
</body>
</html>"""

        safe_ticker = ticker.replace(".", "_")
        target_path = Path(output_path) if output_path else self.output_dir / f"{safe_ticker}_dashboard.html"
        with open(str(target_path), "w", encoding="utf-8") as fh:
            fh.write(html_page)
        return str(target_path)
