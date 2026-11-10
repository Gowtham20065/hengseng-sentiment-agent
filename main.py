"""End-to-End Orchestrator Pipeline for Hang Seng & KRX Sentiment Price Prediction Agent."""

import argparse
from pathlib import Path
from typing import Any, Dict, List
import yaml
from dotenv import load_dotenv

from src.ingestion.market_data import MarketDataFetcher
from src.ingestion.apify_collector import ApifySocialCollector
from src.ingestion.exa_collector import ExaNewsCollector
from src.processing.sentiment_engine import MultilingualSentimentEngine
from src.processing.aligner import TimeSeriesAligner
from src.processing.technicals import TechnicalAnalyzer
from src.models.predictor import PriceDirectionPredictor
from src.models.validation import WalkForwardValidator
from src.reporting.visualizer import QuantVisualizer
from src.reporting.report_generator import ReportGenerator

load_dotenv()


def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    tickers: List[str] = None,
    days: int = 10,
    force_refresh: bool = False,
    config_path: str = "config/settings.yaml",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Run full quantitative prediction, validation, and reporting pipeline."""
    settings = load_settings(config_path)

    # Resolve target assets from settings
    all_assets = []
    if "assets" in settings:
        all_assets.extend(settings["assets"].get("hkex", []))
        all_assets.extend(settings["assets"].get("krx", []))

    asset_map = {a["ticker"]: a for a in all_assets}

    if not tickers:
        target_tickers = list(asset_map.keys())
    else:
        target_tickers = tickers

    print(f"\n================================================================================")
    print(f"  Hang Seng & KRX Sentiment Price Prediction Agent — Pipeline Execution")
    print(f"  Target Assets ({len(target_tickers)}): {', '.join(target_tickers)}")
    print(f"  Lookback Window: {days} Days (Hourly Bars) | Force Refresh: {force_refresh}")
    print(f"================================================================================\n")

    # Initialize Components
    market_fetcher = MarketDataFetcher()
    social_collector = ApifySocialCollector(config_path=config_path)
    news_collector = ExaNewsCollector()
    sentiment_engine = MultilingualSentimentEngine()
    aligner = TimeSeriesAligner()
    tech_analyzer = TechnicalAnalyzer()
    validator = WalkForwardValidator()
    visualizer = QuantVisualizer(output_dir=output_dir)
    report_gen = ReportGenerator(output_dir=output_dir)

    all_profiles: List[Dict[str, Any]] = []
    all_validations: List[Dict[str, Any]] = []
    dashboard_files: List[str] = []

    for idx, ticker in enumerate(target_tickers, 1):
        asset_info = asset_map.get(ticker, {"name": ticker, "keywords": []})
        asset_name = asset_info.get("name", ticker)
        keywords = asset_info.get("keywords", [])

        print(f"[{idx}/{len(target_tickers)}] Processing {asset_name} ({ticker})...")

        # 1. Market Data Ingestion
        bars = market_fetcher.fetch_bars(ticker, days=days, interval="1h", force_refresh=force_refresh)
        print(f"   -> Market Bars: {len(bars)} hourly candles (UTC)")

        # 2. Social & News Ingestion
        posts = social_collector.fetch_social_posts(ticker, keywords=keywords, days=days, force_refresh=force_refresh)
        news = news_collector.fetch_news(ticker, asset_name=asset_name, days=days, keywords=keywords, force_refresh=force_refresh)
        print(f"   -> Collected: {len(posts)} social posts, {len(news)} news articles")

        # 3. Multilingual Sentiment Scoring & Engagement Weighting
        scored = sentiment_engine.score_posts_and_news(ticker, posts, news, force_refresh=force_refresh)

        # 4. Zero-Lookahead Time-Series Alignment
        enriched = aligner.align(bars, scored["scored_posts"], scored["scored_news"])
        print(f"   -> Aligned Features Matrix: {enriched.shape}")

        # 5. Technical Analysis & Support/Resistance Levels
        profile = tech_analyzer.extract_ticker_profile(ticker, enriched, settings=settings)

        # 6. ML Model Training & Next-Bar (n+1) Prediction
        predictor = PriceDirectionPredictor()
        X, y_reg, y_clf, full_df = predictor.prepare_features_and_targets(enriched)
        if len(X) >= 4:
            predictor.fit(X, y_reg, y_clf)
            pred_n1 = predictor.predict_n_plus_1(X.iloc[-1], float(full_df["close"].iloc[-1]))
            profile.update({
                "predicted_return_pct": pred_n1["predicted_return_pct"],
                "predicted_price_n1": pred_n1["predicted_price_n1"],
                "predicted_direction": pred_n1["predicted_direction"],
                "win_probability": pred_n1["win_probability"],
                "top_features": pred_n1["feature_importances"],
            })
            print(f"   -> Prediction n+1: {pred_n1['predicted_direction']} (Target: {pred_n1['predicted_price_n1']} | Win Prob: {pred_n1['win_probability']*100:.1f}%)")

        # 7. Out-of-Sample Walk-Forward Validation
        try:
            val_dossier = validator.validate_ticker(ticker, enriched)
            all_validations.append(val_dossier)
            print(f"   -> Walk-Forward Validation: Hit Rate = {val_dossier['directional_accuracy']}% | Sharpe = {val_dossier['sharpe_ratio']} | Alpha = {val_dossier['excess_return']:+}%")
        except Exception as e:
            val_dossier = None
            print(f"   -> Walk-Forward Validation skipped: {e}")

        all_profiles.append(profile)

        # 8. Interactive Plotly Dashboard
        dashboard_path = visualizer.create_ticker_dashboard(
            ticker=ticker,
            enriched_df=enriched,
            profile=profile,
            validation_dossier=val_dossier,
        )
        dashboard_files.append(dashboard_path)
        print(f"   -> Generated Dashboard: {dashboard_path}\n")

    # 9. Comprehensive Assessment Report
    report_file = report_gen.generate_markdown_report(all_profiles, all_validations)
    print(f"================================================================================")
    print(f"  Pipeline Run Finished!")
    print(f"  Master Quant Dossier: {report_file}")
    print(f"  Interactive Dashboards: {len(dashboard_files)} HTML files in '{output_dir}/'")
    print(f"================================================================================\n")

    return {
        "profiles": all_profiles,
        "validations": all_validations,
        "dashboards": dashboard_files,
        "report_file": report_file,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hang Seng & KRX Sentiment Price Prediction Agent")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers (e.g. '0700.HK,005930.KS')")
    parser.add_argument("--days", type=int, default=10, help="Lookback window in days (default: 10)")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh API caches")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports")

    args = parser.parse_args()
    selected_tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None

    run_pipeline(
        tickers=selected_tickers,
        days=args.days,
        force_refresh=args.force_refresh,
        output_dir=args.output_dir,
    )

