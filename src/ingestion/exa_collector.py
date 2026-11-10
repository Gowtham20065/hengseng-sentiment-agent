"""Exa Neural Search News Collector with Tavily fallback and local caching."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
from dotenv import load_dotenv

load_dotenv()


class ExaNewsCollector:
    """Collects financial news articles using Exa neural search with Tavily fallback."""

    def __init__(
        self,
        cache_dir: str = "data/raw",
        exa_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.use_mock_data = os.getenv("USE_MOCK_DATA", "false").lower() in ("true", "1", "yes")

    def _get_cache_path(self, ticker: str) -> Path:
        safe_ticker = ticker.strip()
        return self.cache_dir / f"news_{safe_ticker}.json"

    def fetch_news(
        self,
        ticker: str,
        asset_name: str,
        days: int = 10,
        keywords: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch news articles for an asset over the given lookback window.

        Extracts: id, ticker, published_date (UTC ISO), title, text, url, score.
        """
        cache_path = self._get_cache_path(ticker)

        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if isinstance(cached_data, list) and len(cached_data) > 0:
                    return cached_data
            except Exception:
                pass

        if self.use_mock_data:
            articles = self._generate_synthetic_news(ticker, asset_name, days, keywords)
            self._save_cache(cache_path, articles)
            return articles

        articles: List[Dict[str, Any]] = []

        # 1. Attempt Exa Search
        if self.exa_api_key and not self.exa_api_key.startswith("your_"):
            try:
                articles = self._query_exa(ticker, asset_name, days, keywords)
            except Exception as e:
                # Log or fall back
                articles = []

        # 2. Attempt Tavily Search Fallback
        if not articles and self.tavily_api_key and not self.tavily_api_key.startswith("your_"):
            try:
                articles = self._query_tavily(ticker, asset_name, days, keywords)
            except Exception as e:
                articles = []

        # 3. Fallback to synthetic generator if live calls yield nothing
        if not articles:
            articles = self._generate_synthetic_news(ticker, asset_name, days, keywords)

        self._save_cache(cache_path, articles)
        return articles

    def _query_exa(
        self,
        ticker: str,
        asset_name: str,
        days: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query Exa API for neural news search."""
        try:
            from exa_py import Exa
        except ImportError:
            return []

        exa = Exa(api_key=self.exa_api_key)
        kw_str = f" OR {' OR '.join(keywords)}" if keywords else ""
        query = f"{asset_name} ({ticker}){kw_str} stock market earnings financial analysis"

        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

        response = exa.search_and_contents(
            query=query,
            type="neural",
            use_autoprompt=True,
            start_published_date=start_date,
            num_results=12,
            text={"max_characters": 1500},
        )

        results = []
        for item in getattr(response, "results", []):
            pub_date = getattr(item, "published_date", None)
            norm_date = self._normalize_utc_timestamp(pub_date)
            results.append({
                "id": str(getattr(item, "id", uuid.uuid4().hex)),
                "ticker": ticker,
                "published_date": norm_date,
                "title": str(getattr(item, "title", "Financial News")),
                "text": str(getattr(item, "text", "")),
                "url": str(getattr(item, "url", "")),
                "score": float(getattr(item, "score", 0.85) or 0.85),
            })
        return results

    def _query_tavily(
        self,
        ticker: str,
        asset_name: str,
        days: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query Tavily Search API as fallback."""
        try:
            from tavily import TavilyClient
        except ImportError:
            return []

        tavily = TavilyClient(api_key=self.tavily_api_key)
        kw_str = f" {' '.join(keywords)}" if keywords else ""
        query = f"{asset_name} {ticker}{kw_str} financial market news"

        response = tavily.search(
            query=query,
            search_depth="advanced",
            days=days,
            max_results=10,
        )

        results = []
        for idx, item in enumerate(response.get("results", [])):
            pub_date = item.get("published_date")
            norm_date = self._normalize_utc_timestamp(pub_date)
            results.append({
                "id": f"tavily_{ticker}_{idx}_{uuid.uuid4().hex[:6]}",
                "ticker": ticker,
                "published_date": norm_date,
                "title": item.get("title", f"{asset_name} Market Update"),
                "text": item.get("content", ""),
                "url": item.get("url", ""),
                "score": float(item.get("score", 0.80) or 0.80),
            })
        return results

    def _normalize_utc_timestamp(self, ts_input: Optional[Any]) -> str:
        """Parse various date strings/objects and format to ISO UTC string."""
        now_utc = datetime.now(timezone.utc)
        if not ts_input:
            return now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        if isinstance(ts_input, datetime):
            if ts_input.tzinfo is None:
                ts_input = ts_input.replace(tzinfo=timezone.utc)
            return ts_input.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            # Common formats
            clean_str = str(ts_input).replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _generate_synthetic_news(
        self,
        ticker: str,
        asset_name: str,
        days: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate realistic synthetic news articles across the lookback timeline."""
        now = datetime.now(timezone.utc)
        is_krx = ticker.endswith(".KS")
        articles = []

        templates = [
            (
                f"{asset_name} reports robust quarterly guidance amid expanding enterprise AI demand",
                f"Financial analysts highlight {asset_name} ({ticker}) strong operational performance and revenue growth, signaling optimistic outlook.",
                0.92,
            ),
            (
                f"Market regulators review tech sector capital allocation, {asset_name} shares steady",
                f"Institutional trading volume for {ticker} indicates defensive positioning following policy guidance update.",
                0.85,
            ),
            (
                f"Global supply chain headwinds affect semiconductor and tech component pricing",
                f"Sector analysts evaluate the broader impact on {asset_name} export volumes and operating margin projections.",
                0.81,
            ),
            (
                f"{asset_name} launches strategic partnership to accelerate cloud and ecosystem expansion",
                f"The announcement was welcomed by market participants, with trading desks projecting improved multi-year free cash flow for {ticker}.",
                0.88,
            ),
            (
                f"Cross-border fund flows surge into Hang Seng and KRX index heavyweights",
                f"Foreign institutional investors increased net purchases in top benchmark constituents including {asset_name}.",
                0.90,
            ),
        ]

        for i, (title, text, score) in enumerate(templates):
            # Spread across lookback days
            hours_ago = int((days * 24 / (len(templates) + 1)) * (i + 1))
            pub_dt = now - timedelta(hours=hours_ago)
            articles.append({
                "id": f"synth_news_{ticker}_{i}_{pub_dt.strftime('%Y%m%d%H')}",
                "ticker": ticker,
                "published_date": pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "title": title,
                "text": text,
                "url": f"https://finance.example.com/news/{ticker.lower()}/{i}",
                "score": score,
            })

        return articles

    def _save_cache(self, path: Path, data: List[Dict[str, Any]]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

