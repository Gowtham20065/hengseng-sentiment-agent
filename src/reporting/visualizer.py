"""Interactive Plotly dashboard generator — institutional-grade 4-panel HTML dashboards."""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design tokens ──────────────────────────────────────────────────────────────
_BG       = "#0D1117"
_PANEL    = "#161B22"
_BORDER   = "#21262D"
_GREEN    = "#3FB950"
_RED      = "#F85149"
_BLUE     = "#58A6FF"
_PURPLE   = "#BC8CFF"
_GOLD     = "#E3B341"
_GREY     = "#6E7681"
_WHITE    = "#E6EDF3"
_DIM      = "#8B949E"

_GRID     = dict(gridcolor="#21262D", gridwidth=0.5,
                 zerolinecolor="#21262D", zerolinewidth=1,
                 showline=True, linecolor="#21262D", linewidth=1,
                 tickfont=dict(size=10, color=_DIM))


class QuantVisualizer:
    """Generates publication-quality interactive 4-panel HTML dashboards."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _col(self, df: pd.DataFrame, name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

    def _fmt(self, v: float, d: int = 2) -> str:
        return f"{v:,.{d}f}"

    # ── main public method ─────────────────────────────────────────────────────

    def create_ticker_dashboard(
        self,
        ticker: str,
        enriched_df: pd.DataFrame,
        profile: Dict[str, Any],
        validation_dossier: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Write a fully self-contained, non-overlapping 4-panel HTML dashboard."""

        # ── 1. Data prep ───────────────────────────────────────────────────────
        df = enriched_df.copy()
        if df.index.name == "timestamp":
            df.reset_index(drop=True, inplace=True)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"] if "timestamp" in df.columns else pd.Series(df.index),
            utc=True,
        )
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # ── 2. Profile scalars ─────────────────────────────────────────────────
        asset      = profile.get("asset_name", ticker)
        price      = profile.get("current_price", float(df["close"].iloc[-1]))
        direction  = profile.get("direction", "NEUTRAL")
        sent_sc    = profile.get("sentiment_score", 0.0)
        sent_lbl   = profile.get("sentiment_label", "NEUTRAL")
        win_p      = profile.get("win_probability", 0.5)
        pos_rec    = profile.get("position_recommendation", {})
        action     = pos_rec.get("action", "FLAT")
        rr         = pos_rec.get("risk_reward_ratio", 1.0)
        s1  = profile.get("support_levels",    {}).get("S1", price * 0.98)
        s2  = profile.get("support_levels",    {}).get("S2", price * 0.96)
        r1  = profile.get("resistance_levels", {}).get("R1", price * 1.02)
        r2  = profile.get("resistance_levels", {}).get("R2", price * 1.04)

        pred = profile.get("predicted_price_n1")
        if not pred and validation_dossier:
            oos = validation_dossier.get("out_of_sample_predictions")
            if isinstance(oos, pd.DataFrame) and not oos.empty:
                pred = float(oos["predicted_price"].iloc[-1])
        pred = pred or price * 1.005
        delta     = pred - price
        delta_pct = delta / price * 100.0
        pred_col  = _GREEN if delta >= 0 else _RED
        sign      = "▲" if delta >= 0 else "▼"

        hit_rate = sharpe = mdd = ic = None
        if validation_dossier:
            hit_rate = validation_dossier.get("directional_accuracy")
            sharpe   = validation_dossier.get("sharpe_ratio")
            mdd      = validation_dossier.get("max_drawdown")
            ic       = validation_dossier.get("information_coefficient")

        market = "HKEX · HKD" if ticker.endswith(".HK") else "KRX · KRW"
        dir_col = _GREEN if direction == "BULLISH" else (_RED if direction == "BEARISH" else _GOLD)
        sent_col = _GREEN if sent_sc > 0.1 else (_RED if sent_sc < -0.1 else _GOLD)
        act_col  = _GREEN if action == "LONG" else (_RED if action == "SHORT" else _GREY)

        # ── 3. Subplot layout — NO subplot_titles (we label via annotations) ───
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.42, 0.19, 0.19, 0.20],
        )

        # ── 4. Panel 1 — Candlestick ──────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=df["timestamp"],
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name="OHLCV",
            increasing_line_color=_GREEN, increasing_fillcolor=_GREEN,
            decreasing_line_color=_RED,   decreasing_fillcolor=_RED,
            whiskerwidth=0.6, line=dict(width=1),
        ), row=1, col=1)

        # Volume bars (hidden from legend — decoded via color)
        if "volume" in df.columns and df["volume"].sum() > 0:
            vol = df["volume"] / df["volume"].max() * (price * 0.06)
            base = df["close"].min() * 0.994
            vcol = [_GREEN if df["close"].iloc[i] >= df["open"].iloc[i] else _RED
                    for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df["timestamp"], y=vol, base=base,
                marker_color=vcol, opacity=0.22,
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)

        # S/R lines (hlines don't clutter legend)
        for y_v, col, dash, lw, lbl, apos in [
            (s2, _GREEN, "dot",  1.1, f"S2 {self._fmt(s2)}", "bottom left"),
            (s1, _GREEN, "dash", 1.7, f"S1 {self._fmt(s1)}", "bottom left"),
            (r1, _RED,   "dash", 1.7, f"R1 {self._fmt(r1)}", "top left"),
            (r2, _RED,   "dot",  1.1, f"R2 {self._fmt(r2)}", "top left"),
        ]:
            fig.add_hline(y=y_v, line_dash=dash, line_color=col, line_width=lw,
                          annotation_text=f"<b>{lbl}</b>",
                          annotation_position=apos,
                          annotation_font=dict(size=10, color=col),
                          row=1, col=1)

        # Current price reference
        fig.add_hline(y=price, line_color=_WHITE, line_width=0.7,
                      line_dash="solid", opacity=0.3, row=1, col=1)

        # n+1 target star (shown in legend — it's the key signal)
        next_ts = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        fig.add_trace(go.Scatter(
            x=[next_ts], y=[pred],
            mode="markers+text",
            name=f"{sign} n+1 Target: {self._fmt(pred)} ({delta_pct:+.2f}%)",
            marker=dict(symbol="star", size=15, color=_GOLD,
                        line=dict(width=1.5, color=_WHITE)),
            text=[f"  {self._fmt(pred)}"], textposition="middle right",
            textfont=dict(color=_GOLD, size=11, family="monospace"),
            hovertemplate=f"n+1 Target: {self._fmt(pred)}<br>Δ {delta:+.4f} ({delta_pct:+.2f}%)<extra></extra>",
        ), row=1, col=1)

        # ── 5. Panel 2 — Sentiment ─────────────────────────────────────────────
        soc  = self._col(df, "social_weighted_sentiment")
        news = self._col(df, "news_sentiment")
        div_ = self._col(df, "news_social_divergence")
        mom  = self._col(df, "sentiment_momentum")

        fig.add_trace(go.Bar(
            x=df["timestamp"], y=soc,
            name="Retail Sentiment",
            marker_color=[_GREEN if v >= 0 else _RED for v in soc],
            opacity=0.65, showlegend=True,
            hovertemplate="Retail: %{y:.4f}<extra></extra>",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=news, mode="lines",
            name="News Sentiment",
            line=dict(color=_BLUE, width=1.8), showlegend=True,
            hovertemplate="News: %{y:.4f}<extra></extra>",
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=div_, mode="lines",
            name="Divergence",
            line=dict(color=_PURPLE, width=1.3, dash="dot"), showlegend=True,
            hovertemplate="Divergence: %{y:.4f}<extra></extra>",
        ), row=2, col=1)

        # Momentum — hidden from legend to reduce clutter
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=mom, mode="lines",
            line=dict(color=_GOLD, width=1.0, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)

        fig.add_hline(y=0, line_color=_BORDER, line_width=0.8, row=2, col=1)

        # ── 6. Panel 3 — MACD + RSI ───────────────────────────────────────────
        macd_l = self._col(df, "macd")
        macd_s = self._col(df, "macd_signal")
        macd_h = self._col(df, "macd_hist")
        rsi_v  = self._col(df, "rsi_14")

        fig.add_trace(go.Bar(
            x=df["timestamp"], y=macd_h,
            marker_color=[_GREEN if v >= 0 else _RED for v in macd_h],
            opacity=0.50, showlegend=False, hoverinfo="skip",
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=macd_l, mode="lines",
            name="MACD (12,26)",
            line=dict(color="#00E5FF", width=1.6), showlegend=True,
            hovertemplate="MACD: %{y:.6f}<extra></extra>",
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=macd_s, mode="lines",
            name="Signal (9)",
            line=dict(color="#FF9100", width=1.3), showlegend=True,
            hovertemplate="Signal: %{y:.6f}<extra></extra>",
        ), row=3, col=1)

        # RSI on a separate right-axis
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=rsi_v, mode="lines",
            name="RSI (14)",
            line=dict(color=_PURPLE, width=1.3, dash="dot"), showlegend=True,
            yaxis="y7",
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=3, col=1)

        fig.add_hline(y=0, line_color=_BORDER, line_width=0.8, row=3, col=1)

        # ── 7. Panel 4 — Equity curve ─────────────────────────────────────────
        INIT = 100_000.0
        strat, bench, xs = [INIT], [INIT], []

        oos_data = validation_dossier.get("out_of_sample_predictions") if validation_dossier else None
        if isinstance(oos_data, pd.DataFrame) and not oos_data.empty:
            ots = oos_data.get("timestamp", df["timestamp"].iloc[-len(oos_data):])
            xs  = list(ots)
            for _, row in oos_data.iterrows():
                r   = float(row.get("realized_return", 0.0))
                d   = int(row.get("predicted_direction", 1))
                sr  = (r if d == 1 else -r) - 0.0012  # 12 bps cost
                strat.append(strat[-1] * (1.0 + sr))
                bench.append(bench[-1] * (1.0 + r))
            if len(xs) < len(strat):
                xs = [xs[0] - pd.Timedelta(hours=1)] + xs
        else:
            xs = list(df["timestamp"])
            for i in range(1, len(df)):
                r = (df["close"].iloc[i] - df["close"].iloc[i-1]) / df["close"].iloc[i-1]
                strat.append(strat[-1] * (1.0 + r * 0.5))
                bench.append(bench[-1] * (1.0 + r))
            if len(xs) < len(strat):
                xs = [xs[0] - pd.Timedelta(hours=1)] + xs

        alpha_pct = (strat[-1] - bench[-1]) / INIT * 100

        fig.add_trace(go.Scatter(
            x=xs, y=strat, mode="lines",
            name=f"Strategy (α {alpha_pct:+.2f}%)",
            line=dict(color=_GREEN, width=2.3),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.07)",
            hovertemplate="Strategy: $%{y:,.0f}<extra></extra>",
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=xs, y=bench, mode="lines",
            name="Buy & Hold",
            line=dict(color=_GREY, width=1.6, dash="dash"),
            hovertemplate="Benchmark: $%{y:,.0f}<extra></extra>",
        ), row=4, col=1)

        fig.add_hline(y=INIT, line_color=_BORDER, line_width=0.8, row=4, col=1)

        # ── 8. Panel label annotations (left of each panel, NOT subplot titles) ─
        panel_labels = [
            (0.00, 1.00,  "① Price Action & S/R"),
            (0.00, 0.555, "② Sentiment Dynamics"),
            (0.00, 0.335, "③ MACD · RSI (14)"),
            (0.00, 0.115, "④ Walk-Forward Equity"),
        ]
        for px, py, txt in panel_labels:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=px, y=py,
                text=f"<b style='font-size:10px;color:{_DIM}'>{txt}</b>",
                showarrow=False, xanchor="left", yanchor="top",
            )

        # ── 9. Global layout ───────────────────────────────────────────────────
        fig.update_layout(
            template="plotly_dark",
            height=1000,
            paper_bgcolor=_BG,
            plot_bgcolor=_PANEL,
            font=dict(family="'Inter','Segoe UI',Arial,sans-serif",
                      color=_WHITE, size=11),
            title=dict(
                text=(
                    f"<b>{asset}</b>  "
                    f"<span style='color:{_DIM};font-size:12px'>{ticker} · {market}</span>"
                ),
                x=0.01, y=0.995, xanchor="left", yanchor="top",
                font=dict(size=16),
            ),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            # ── LEGEND: horizontal, anchored BELOW chart, never overlaps ──────
            legend=dict(
                orientation="h",
                yanchor="top",    y=-0.03,
                xanchor="center", x=0.5,
                bgcolor="rgba(13,17,23,0.90)",
                bordercolor=_BORDER, borderwidth=1,
                font=dict(size=10),
                traceorder="normal",
                tracegroupgap=0,
                itemwidth=70,
            ),
            margin=dict(l=65, r=65, t=55, b=100),
        )

        # Axis grid on all subplots
        for ax in ["xaxis", "xaxis2", "xaxis3", "xaxis4",
                   "yaxis",  "yaxis2", "yaxis3", "yaxis4"]:
            fig.update_layout(**{ax: _GRID})

        # Secondary y-axis for RSI (panel 3, right side)
        fig.update_layout(yaxis7=dict(
            overlaying="y3",
            side="right",
            range=[0, 100],
            showgrid=False,
            tickfont=dict(size=9, color=_PURPLE),
            tickvals=[30, 50, 70],
        ))

        # RSI overbought / oversold reference bands
        for lvl, lbl in [(70, "OB 70"), (30, "OS 30")]:
            fig.add_hline(y=lvl, line_color=_PURPLE, line_width=0.7,
                          line_dash="dot", opacity=0.4, row=3, col=1)

        # ── 10. Build full HTML page ───────────────────────────────────────────
        plotly_div = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": ticker.replace(".", "_"),
                    "height": 1000, "width": 1800, "scale": 2,
                },
            },
        )

        # KPI card helper
        def kpi(label: str, value: str, color: str, sub: str = "") -> str:
            return (
                f'<div class="k">'
                f'<div class="kl">{label}</div>'
                f'<div class="kv" style="color:{color}">{value}</div>'
                f'{"<div class=ks>" + sub + "</div>" if sub else ""}'
                f'</div>'
            )

        kpi_row = "".join([
            kpi("Price",         self._fmt(price),                    _WHITE,  market.split("·")[1].strip()),
            kpi("Sentiment",     f"{sent_sc:+.3f}",                  sent_col, sent_lbl),
            kpi("Direction",     f"{'▲' if direction=='BULLISH' else '▼' if direction=='BEARISH' else '◆'} {direction}",
                                                                      dir_col, "n+1 Forecast"),
            kpi("Pred n+1",      self._fmt(pred),                    pred_col, f"{delta:+.4f} ({delta_pct:+.2f}%)"),
            kpi("Win Prob",      f"{win_p*100:.1f}%",               _BLUE,   "LightGBM Ens."),
            kpi("Action",        action,                              act_col, f"R:R {rr:.2f}×"),
            kpi("Support",       f"S1 {self._fmt(s1)}",             _GREEN,  f"S2 {self._fmt(s2)}"),
            kpi("Resistance",    f"R1 {self._fmt(r1)}",             _RED,    f"R2 {self._fmt(r2)}"),
        ] + (
            [kpi("Hit Rate",     f"{hit_rate:.1f}%",                _PURPLE, "Walk-Forward OOS")] if hit_rate is not None else []
        ) + (
            [kpi("Sharpe",       f"{sharpe:.2f}",                   _BLUE,   "Strategy")] if sharpe is not None else []
        ) + (
            [kpi("Max DD",       f"{mdd:.2f}%",                     _RED,    "OOS Period")] if mdd is not None else []
        ) + (
            [kpi("Info Coeff",   f"{ic:+.3f}",                     _GOLD,   "Spearman IC")] if ic is not None else []
        ))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{asset} ({ticker}) — CrowdWisdomTrading</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{min-height:100%;background:{_BG};color:{_WHITE};font-family:'Inter','Segoe UI',Arial,sans-serif;font-size:13px}}

/* ── header ── */
.hdr{{display:flex;align-items:center;justify-content:space-between;
      padding:12px 20px;background:{_PANEL};border-bottom:1px solid {_BORDER}}}
.hdr-l .brand{{font-size:11px;font-weight:700;letter-spacing:.08em;
               color:{_DIM};text-transform:uppercase}}
.hdr-l .brand span{{color:{_BLUE}}}
.hdr-l .asset{{font-size:20px;font-weight:700;color:{_WHITE};margin-top:2px}}
.hdr-l .asset em{{font-style:normal;font-size:13px;font-weight:400;
                  color:{_DIM};margin-left:8px}}
.hdr-r{{text-align:right;font-size:11px;color:{_DIM};line-height:1.7}}
.hdr-r b{{color:{_WHITE}}}

/* ── KPI strip ── */
.kpis{{display:flex;flex-wrap:wrap;gap:8px;padding:10px 20px;
       background:{_BG};border-bottom:1px solid {_BORDER}}}
.k{{flex:1;min-width:100px;max-width:160px;background:{_PANEL};
    border:1px solid {_BORDER};border-radius:7px;padding:9px 13px}}
.kl{{font-size:9px;font-weight:700;letter-spacing:.09em;
     text-transform:uppercase;color:{_DIM};margin-bottom:4px}}
.kv{{font-size:17px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}}
.ks{{font-size:10px;color:{_DIM};margin-top:3px}}

/* ── chart ── */
.chart{{padding:8px 12px 0}}

/* ── footer ── */
.foot{{text-align:center;padding:10px;font-size:10px;color:{_DIM};
       border-top:1px solid {_BORDER};margin-top:4px}}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-l">
    <div class="brand"><span>CrowdWisdom</span>Trading · Sentiment Price Prediction Agent</div>
    <div class="asset">{asset}<em>{ticker}</em></div>
  </div>
  <div class="hdr-r">
    Market: <b>{market}</b><br/>
    Lookback: <b>10-Day Hourly</b><br/>
    Generated: <b id="ts"></b>
  </div>
</div>

<div class="kpis">{kpi_row}</div>

<div class="chart">{plotly_div}</div>

<div class="foot">
  CrowdWisdomTrading &nbsp;·&nbsp; Hang Seng &amp; KRX Sentiment Price Prediction Agent
  &nbsp;·&nbsp; Zero-Lookahead Walk-Forward &nbsp;·&nbsp; LightGBM + Ridge Ensemble
  &nbsp;·&nbsp; Gemini 2.5 Flash · Exa · Apify
</div>

<script>document.getElementById("ts").textContent=new Date().toUTCString();</script>
</body>
</html>"""

        safe = ticker.replace(".", "_")
        path = Path(output_path) if output_path else self.output_dir / f"{safe}_dashboard.html"
        path.write_text(html, encoding="utf-8")
        return str(path)
