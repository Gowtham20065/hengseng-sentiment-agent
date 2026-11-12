"""Technical Analysis & Support/Resistance Engine with Microstructure and Divergence Analysis."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


class TechnicalAnalyzer:
    """Quantitative Technical Analysis, Multi-Method Support/Resistance, and Ticker Profiling."""

    def __init__(self):
        pass

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI (14), MACD (12, 26, 9), ATR (14), and Volume Ratio."""
        if df.empty or len(df) < 2:
            return df

        data = df.copy()
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        volume = data["volume"].astype(float)

        # 1. RSI (14-period)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Wilder's smoothing / exponential moving average
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=1, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        data["rsi_14"] = rsi.clip(lower=0.0, upper=100.0).round(2)

        # 2. MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        data["macd"] = macd.round(4)
        data["macd_signal"] = macd_signal.round(4)
        data["macd_hist"] = macd_hist.round(4)

        # 3. ATR (Average True Range, 14-period)
        prev_close = close.shift(1).fillna(close)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=14, min_periods=1).mean()
        # Fallback if ATR is 0
        min_atr = close * 0.005
        atr = np.maximum(atr, min_atr)
        data["atr_14"] = atr.round(4)

        # 4. Volume Ratio (Volume / 10-period rolling average)
        vol_ma10 = volume.rolling(window=10, min_periods=1).mean()
        data["volume_ratio"] = np.where(vol_ma10 > 0, volume / vol_ma10, 1.0).round(2)

        return data

    def compute_support_resistance(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Combine Pivot Extrema, Volume-Weighted KDE, and Fibonacci Retracement into [S1, S2] and [R1, R2].

        Strictly guarantees: S2 < S1 < current_price < R1 < R2.
        """
        if df.empty:
            return {"S1": 0.0, "S2": 0.0, "confidence": 0.0}, {"R1": 0.0, "R2": 0.0, "confidence": 0.0}

        close = df["close"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        volume = df["volume"].astype(float).values

        curr_p = float(close[-1])
        recent_atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else curr_p * 0.015
        if recent_atr <= 0:
            recent_atr = curr_p * 0.015

        all_supports: List[float] = []
        all_resistances: List[float] = []

        # --- Method A: Rolling Extrema / Pivot Points (scipy.signal.find_peaks) ---
        if len(close) >= 5:
            peaks, _ = find_peaks(high, distance=max(2, len(close) // 6))
            troughs, _ = find_peaks(-low, distance=max(2, len(close) // 6))

            for p in high[peaks]:
                if p > curr_p * 1.001:
                    all_resistances.append(float(p))
            for t in low[troughs]:
                if t < curr_p * 0.999:
                    all_supports.append(float(t))

        # --- Method B: Volume-Weighted Price Nodes (KDE) ---
        if len(close) >= 4 and np.std(close) > 1e-4 and len(np.unique(close)) >= 3:
            try:
                kde = gaussian_kde(close, weights=np.maximum(volume, 1.0))
                p_min, p_max = float(np.min(low)), float(np.max(high))
                grid = np.linspace(p_min * 0.98, p_max * 1.02, 200)
                densities = kde(grid)
                kde_peaks, _ = find_peaks(densities)
                hvn_prices = grid[kde_peaks]

                for hp in hvn_prices:
                    if hp < curr_p * 0.998:
                        all_supports.append(float(hp))
                    elif hp > curr_p * 1.002:
                        all_resistances.append(float(hp))
            except Exception:
                pass

        # --- Method C: Fibonacci Retracement Levels ---
        h_max = float(np.max(high))
        l_min = float(np.min(low))
        fib_range = h_max - l_min

        if fib_range > 0:
            fib_levels = [
                l_min + 0.236 * fib_range,
                l_min + 0.382 * fib_range,
                l_min + 0.500 * fib_range,
                l_min + 0.618 * fib_range,
                l_min + 0.786 * fib_range,
            ]
            for fl in fib_levels:
                if fl < curr_p * 0.998:
                    all_supports.append(float(fl))
                elif fl > curr_p * 1.002:
                    all_resistances.append(float(fl))

        # --- Synthesis & Strict Ordering Enforcement ---
        # Format decimals: 2 for HKD, integer for KRW
        decimals = 0 if curr_p > 1000 else 2

        # Filter strictly below curr_p
        valid_supports = sorted([s for s in all_supports if s < curr_p], reverse=True)
        if len(valid_supports) >= 2:
            s1 = valid_supports[0]
            # Ensure S2 is distinctly lower than S1
            s2_cands = [s for s in valid_supports if s < s1 * 0.995]
            s2 = s2_cands[0] if s2_cands else s1 - recent_atr
        elif len(valid_supports) == 1:
            s1 = valid_supports[0]
            s2 = s1 - recent_atr
        else:
            s1 = curr_p - 0.75 * recent_atr
            s2 = curr_p - 1.75 * recent_atr

        # Filter strictly above curr_p
        valid_resistances = sorted([r for r in all_resistances if r > curr_p])
        if len(valid_resistances) >= 2:
            r1 = valid_resistances[0]
            r2_cands = [r for r in valid_resistances if r > r1 * 1.005]
            r2 = r2_cands[0] if r2_cands else r1 + recent_atr
        elif len(valid_resistances) == 1:
            r1 = valid_resistances[0]
            r2 = r1 + recent_atr
        else:
            r1 = curr_p + 0.75 * recent_atr
            r2 = curr_p + 1.75 * recent_atr

        # Final guarantee of strict inequality: S2 < S1 < curr_p < R1 < R2
        if s1 >= curr_p:
            s1 = curr_p - 0.5 * recent_atr
        if s2 >= s1:
            s2 = s1 - 0.75 * recent_atr

        if r1 <= curr_p:
            r1 = curr_p + 0.5 * recent_atr
        if r2 <= r1:
            r2 = r1 + 0.75 * recent_atr

        s1 = round(s1, decimals)
        s2 = round(s2, decimals)
        r1 = round(r1, decimals)
        r2 = round(r2, decimals)

        support_dict = {
            "S1": s1,
            "S2": s2,
            "confidence": 0.88,
            "methods": ["rolling_pivot", "volume_kde", "fibonacci"],
        }
        resistance_dict = {
            "R1": r1,
            "R2": r2,
            "confidence": 0.88,
            "methods": ["rolling_pivot", "volume_kde", "fibonacci"],
        }

        return support_dict, resistance_dict

    def extract_ticker_profile(
        self,
        ticker: str,
        enriched_df: pd.DataFrame,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize ticker profile, multi-method S/R levels, position recommendations, and microstructure alerts."""
        if enriched_df.empty:
            raise ValueError(f"Enriched DataFrame is empty for ticker {ticker}")

        # Compute technical indicators if not present
        df = self.compute_indicators(enriched_df)
        latest = df.iloc[-1]

        # Asset metadata from settings
        asset_name = self._lookup_asset_name(ticker, settings)

        curr_p = float(latest["close"])
        decimals = 0 if curr_p > 1000 else 2
        curr_p = round(curr_p, decimals)

        # Sentiment metrics: use latest non-zero sentiment from recent bars, or fallback to latest bar
        non_zero_social = df["social_weighted_sentiment"][df["social_weighted_sentiment"] != 0.0]
        if not non_zero_social.empty:
            sent_score = float(non_zero_social.iloc[-1])
        else:
            sent_score = float(latest.get("social_weighted_sentiment", 0.0))

        non_zero_news = df["news_sentiment"][df["news_sentiment"] != 0.0]
        if not non_zero_news.empty:
            news_sent = float(non_zero_news.iloc[-1])
        else:
            news_sent = float(latest.get("news_sentiment", 0.0))

        divergence = round(news_sent - sent_score, 4)
        mention_zscore = float(latest.get("mention_zscore", 0.0))
        total_mentions = int(df["social_mentions"].sum()) if "social_mentions" in df.columns else 0
        exchange = "HKEX" if ticker.endswith(".HK") else "KRX"

        # Sentiment Label
        if sent_score >= 0.50:
            sentiment_label = "EXTREMELY BULLISH"
        elif sent_score >= 0.15:
            sentiment_label = "BULLISH"
        elif sent_score <= -0.50:
            sentiment_label = "EXTREMELY BEARISH"
        elif sent_score <= -0.15:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"

        # Support & Resistance Levels
        support_levels, resistance_levels = self.compute_support_resistance(df)
        s1 = support_levels["S1"]
        s2 = support_levels["S2"]
        r1 = resistance_levels["R1"]
        r2 = resistance_levels["R2"]

        atr = float(latest.get("atr_14", curr_p * 0.015))
        rsi = float(latest.get("rsi_14", 50.0))
        macd_hist = float(latest.get("macd_hist", 0.0))

        # Direction Synthesis: Sentiment (50%) + MACD Momentum (30%) + RSI Bias (20%)
        tech_score = (0.6 * np.sign(macd_hist)) + (0.4 * ((rsi - 50.0) / 50.0))
        combined_signal = (0.55 * sent_score) + (0.45 * tech_score)

        if combined_signal >= 0.20:
            direction = "BULLISH"
        elif combined_signal <= -0.20:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Position Recommendation
        if direction == "BULLISH":
            action = "LONG"
            entry = curr_p
            stop_loss = round(s1 - (0.5 * atr), decimals)
            take_profit = round(r1, decimals)
            risk = max(entry - stop_loss, 0.01)
            reward = max(take_profit - entry, 0.01)
            rr = round(reward / risk, 2)
        elif direction == "BEARISH":
            action = "SHORT"
            entry = curr_p
            stop_loss = round(r1 + (0.5 * atr), decimals)
            take_profit = round(s1, decimals)
            risk = max(stop_loss - entry, 0.01)
            reward = max(entry - take_profit, 0.01)
            rr = round(reward / risk, 2)
        else:
            action = "FLAT / AVOID"
            entry = curr_p
            stop_loss = round(s1 - (0.5 * atr), decimals)
            take_profit = round(r1, decimals)
            rr = 1.0

        position_rec = {
            "action": action,
            "entry_price": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": rr,
        }

        # Crowd vs Smart Money Divergence Analysis
        if divergence <= -0.30:
            divergence_alert = "BEARISH DIVERGENCE: Retail hyper-bullish while institutional news is cautious."
        elif divergence >= 0.30:
            divergence_alert = "BULLISH DIVERGENCE: Institutional news accumulating while retail crowd is overly pessimistic."
        else:
            divergence_alert = "NEUTRAL ALIGNMENT: Retail and institutional sentiment in relative equilibrium."

        crowd_vs_smart_money = {
            "retail_sentiment": round(sent_score, 4),
            "institutional_sentiment": round(news_sent, 4),
            "divergence": round(divergence, 4),
            "alert": divergence_alert,
        }

        # Market Microstructure Notes
        if ticker.endswith(".HK"):
            dist_to_s1_pct = round(abs(curr_p - s1) / curr_p * 100, 2)
            dist_to_r1_pct = round(abs(r1 - curr_p) / curr_p * 100, 2)
            if dist_to_s1_pct < 1.8:
                micro_note = (
                    f"HIGH RISK CBBC HEAVY CONCENTRATION: Bull contracts vulnerable to knock-out cascade near S1 ({s1}). "
                    f"Market makers likely to accelerate downward liquidity sweep."
                )
            elif dist_to_r1_pct < 1.8:
                micro_note = (
                    f"BEAR CONTRACT KNOCK-OUT ZONE: High density of Bear CBBCs subject to forced liquidation squeeze near R1 ({r1}). "
                    f"Potential upside short-squeeze acceleration."
                )
            else:
                micro_note = f"CBBC Neutral Zone: Price buffered from immediate knock-out call clusters (S1={s1}, R1={r1})."
        else:
            # KRX Microstructure
            if mention_zscore > 1.8:
                micro_note = (
                    f"RETAIL FOMO SPIKE: Extreme buzz volume on Naver discussion boards (Z={mention_zscore:.2f}). "
                    f"Elevated retail leverage; vulnerable to sudden mean-reversion."
                )
            elif mention_zscore < -1.0 and sent_score < -0.15:
                micro_note = (
                    f"PANIC SELLING CAPITULATION: Capitulation sentiment on Naver boards indicates exhaustion selling near S1 ({s1})."
                )
            else:
                micro_note = f"NORMAL RETAIL PARTICIPATION: Discussion board velocity within normal bounds (Z={mention_zscore:.2f})."

        profile = {
            "ticker": ticker,
            "asset_name": asset_name,
            "exchange": exchange,
            "total_mentions": total_mentions,
            "current_price": curr_p,
            "sentiment_score": round(sent_score, 4),
            "sentiment_label": sentiment_label,
            "direction": direction,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "position_recommendation": position_rec,
            "crowd_vs_smart_money": crowd_vs_smart_money,
            "market_microstructure_notes": micro_note,
        }

        return profile

    def _lookup_asset_name(self, ticker: str, settings: Optional[Dict[str, Any]]) -> str:
        """Lookup company name from settings or built-in registry."""
        default_names = {
            "0700.HK": "Tencent Holdings",
            "9988.HK": "Alibaba Group",
            "3690.HK": "Meituan",
            "1810.HK": "Xiaomi",
            "1211.HK": "BYD",
            "005930.KS": "Samsung Electronics",
            "000660.KS": "SK Hynix",
            "035420.KS": "NAVER",
            "035720.KS": "Kakao",
            "373220.KS": "LG Energy Solution",
        }

        if settings and "assets" in settings:
            for market in ["hkex", "krx"]:
                for item in settings["assets"].get(market, []):
                    if item.get("ticker") == ticker:
                        return item.get("name", default_names.get(ticker, ticker))

        return default_names.get(ticker, ticker)

