"""Multilingual Quant Sentiment Engine using Gemini with robust local lexicon fallback."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class MultilingualSentimentEngine:
    """Extracts granular quant sentiment, retail positioning, and confidence.

    Primary Scorer: google-genai (Gemini 2.5 / 1.5 Flash).
    Fallback Scorer: High-precision Cantonese and Korean financial slang dictionary.
    """

    # Cantonese / Hong Kong financial slang dictionary
    HK_BULLISH = [
        "升", "破頂", "掃貨", "鐵底", "睇好", "買入", "新高", "爆升",
        "回購", "超預期", "加倉", "穩如泰山", "北水", "入通", "突破",
        "雙龍出海", "大升", "好消息", "業績亮眼", "長線持有",
    ]
    HK_BEARISH = [
        "跌", "沽", "爆倉", "蟹貨", "跌穿", "受壓", "斬倉", "大跌",
        "割肉", "價格戰", "虧損", "睇淡", "沽期指", "沽空", "減持",
        "利潤回落", "短缺", "下調", "被套", "高位站崗",
    ]
    HK_BAGHOLDING = [
        "蟹貨", "爆倉", "被套", "高位站崗", "套牢", "唔好斬", "救命",
        "幾時家鄉", "等回鄉", "坐艇", "深套",
    ]
    HK_LONG = ["掃貨", "買入", "加倉", "追入", "all in", "重倉", "長線持有"]
    HK_SHORT = ["沽空", "開put", "做空", "對沖", "沽期指"]

    # Korean financial slang dictionary
    KR_BULLISH = [
        "상한가", "떡상", "가즈아", "신고가", "반등", "호실적", "매수",
        "쌍끌이", "턴어라운드", "수주", "상승", "흑자전환", "바닥 확인",
        "숏스퀴즈", "독점", "상향", "회복", "풀매수", "안착",
    ]
    KR_BEARISH = [
        "떡락", "하한가", "손절", "물린", "곡소리", "급락", "신저가",
        "우려", "적자", "하향", "패닉셀", "공매도", "하락", "캐즘",
        "정체", "과열", "차익실현", "불확실",
    ]
    KR_BAGHOLDING = [
        "물린", "손절", "존버", "탈출", "구조대", "곡소리", "고점 물림",
        "원금 회복", "평단가", "강제 장투",
    ]
    KR_LONG = ["가즈아", "매수", "풀매수", "추매", "상한가", "탑승"]
    KR_SHORT = ["공매도", "숏", "인버스", "곱버스"]

    # English financial terms
    EN_BULLISH = [
        "surge", "beat", "growth", "record", "upgrade", "bullish",
        "profit", "outperform", "rally", "breakthrough", "soar",
    ]
    EN_BEARISH = [
        "plunge", "miss", "drop", "slump", "downgrade", "bearish",
        "loss", "headwind", "decline", "selloff", "penalty", "risk",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        cache_dir: str = "data/processed",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    @staticmethod
    def calculate_engagement_weight(
        views: float = 0,
        likes: float = 0,
        reposts: float = 0,
        author_followers: float = 0,
    ) -> float:
        """Calculate social engagement weight.

        Formula: w = log(1 + views) * (1 + 0.5 * likes + 0.5 * reposts) * log(10 + author_followers)
        """
        v = max(0.0, float(views or 0))
        l = max(0.0, float(likes or 0))
        r = max(0.0, float(reposts or 0))
        f = max(0.0, float(author_followers or 0))

        # Using natural log as per standard quantitative NLP weighting
        weight = math.log(1.0 + v) * (1.0 + 0.5 * l + 0.5 * r) * math.log(10.0 + f)
        return float(weight)

    def score_text(
        self,
        text: str,
        ticker: str,
        views: float = 0,
        likes: float = 0,
        reposts: float = 0,
        author_followers: float = 0,
    ) -> Dict[str, Any]:
        """Score single text via Gemini LLM with instant local lexicon fallback."""
        weight = self.calculate_engagement_weight(views, likes, reposts, author_followers)

        # 1. Try Gemini LLM first if client exists
        if self._client:
            try:
                llm_res = self._call_gemini_scorer(text, ticker)
                if llm_res is not None:
                    llm_res["engagement_weight"] = weight
                    llm_res["weighted_sentiment"] = round(llm_res["sentiment_score"] * weight, 4)
                    return llm_res
            except Exception:
                pass

        # 2. Resilient Fallback to local multilingual financial lexicon
        fallback_res = self._fallback_rule_scorer(text, ticker)
        fallback_res["engagement_weight"] = weight
        fallback_res["weighted_sentiment"] = round(fallback_res["sentiment_score"] * weight, 4)
        return fallback_res

    def _call_gemini_scorer(self, text: str, ticker: str) -> Optional[Dict[str, Any]]:
        """Query Gemini using google-genai for structured financial sentiment."""
        prompt = (
            f"You are a quant financial sentiment analyst specializing in HKEX and KRX equities.\n"
            f"Analyze the following text regarding ticker {ticker}.\n"
            f"Return ONLY valid JSON with this exact structure:\n"
            "{\n"
            '  "sentiment_score": <float between -1.0 (extremely bearish) and +1.0 (extremely bullish)>,\n'
            '  "direction": "<BULLISH, BEARISH, or NEUTRAL>",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "retail_position": "<LONG, SHORT, BAGHOLDING, or SIDELINED>"\n'
            "}\n\n"
            f"Text: {text}"
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        resp_text = response.text.strip()
        if resp_text.startswith("```json"):
            resp_text = resp_text[7:]
        if resp_text.startswith("```"):
            resp_text = resp_text[3:]
        if resp_text.endswith("```"):
            resp_text = resp_text[:-3]

        data = json.loads(resp_text.strip())
        score = float(max(-1.0, min(1.0, data.get("sentiment_score", 0.0))))
        direction = str(data.get("direction", "NEUTRAL")).upper()
        if direction not in ["BULLISH", "BEARISH", "NEUTRAL"]:
            direction = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")

        position = str(data.get("retail_position", "SIDELINED")).upper()
        if position not in ["LONG", "SHORT", "BAGHOLDING", "SIDELINED"]:
            position = "SIDELINED"

        conf = float(max(0.0, min(1.0, data.get("confidence", 0.8))))

        return {
            "sentiment_score": round(score, 4),
            "direction": direction,
            "confidence": round(conf, 4),
            "retail_position": position,
        }

    def _fallback_rule_scorer(self, text: str, ticker: str) -> Dict[str, Any]:
        """Local lexicon analyzer for Cantonese, Korean, and English financial terms."""
        if not text:
            return {
                "sentiment_score": 0.0,
                "direction": "NEUTRAL",
                "confidence": 0.5,
                "retail_position": "SIDELINED",
            }

        bull_hits = 0
        bear_hits = 0
        bagholding_hits = 0
        long_hits = 0
        short_hits = 0

        # Cantonese analysis
        for kw in self.HK_BULLISH:
            if kw in text:
                bull_hits += 1
        for kw in self.HK_BEARISH:
            if kw in text:
                bear_hits += 1
        for kw in self.HK_BAGHOLDING:
            if kw in text:
                bagholding_hits += 1
        for kw in self.HK_LONG:
            if kw in text:
                long_hits += 1
        for kw in self.HK_SHORT:
            if kw in text:
                short_hits += 1

        # Korean analysis
        for kw in self.KR_BULLISH:
            if kw in text:
                bull_hits += 1
        for kw in self.KR_BEARISH:
            if kw in text:
                bear_hits += 1
        for kw in self.KR_BAGHOLDING:
            if kw in text:
                bagholding_hits += 1
        for kw in self.KR_LONG:
            if kw in text:
                long_hits += 1
        for kw in self.KR_SHORT:
            if kw in text:
                short_hits += 1

        # English analysis
        text_lower = text.lower()
        for kw in self.EN_BULLISH:
            if kw in text_lower:
                bull_hits += 1
        for kw in self.EN_BEARISH:
            if kw in text_lower:
                bear_hits += 1

        net_hits = bull_hits - bear_hits
        total_hits = bull_hits + bear_hits

        # Compute sentiment score in [-1.0, 1.0]
        if total_hits == 0:
            sentiment_score = 0.0
            direction = "NEUTRAL"
            confidence = 0.5
        else:
            sentiment_score = math.tanh(0.4 * net_hits)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            if sentiment_score >= 0.15:
                direction = "BULLISH"
            elif sentiment_score <= -0.15:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
            confidence = min(0.95, 0.55 + 0.10 * total_hits)

        # Detect retail crowd positioning
        if bagholding_hits > 0:
            retail_position = "BAGHOLDING"
        elif long_hits > short_hits and direction == "BULLISH":
            retail_position = "LONG"
        elif short_hits > 0 or direction == "BEARISH":
            retail_position = "SHORT"
        elif direction == "BULLISH":
            retail_position = "LONG"
        else:
            retail_position = "SIDELINED"

        return {
            "sentiment_score": round(sentiment_score, 4),
            "direction": direction,
            "confidence": round(confidence, 4),
            "retail_position": retail_position,
        }

    def score_posts_and_news(
        self,
        ticker: str,
        posts: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        force_refresh: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Batch score social posts and news articles, caching to data/processed/sentiment_{ticker}.json."""
        cache_path = self.cache_dir / f"sentiment_{ticker}.json"

        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if "scored_posts" in cached_data and "scored_news" in cached_data:
                    return cached_data
            except Exception:
                pass

        scored_posts = []
        for post in posts:
            res = self.score_text(
                text=post.get("text", ""),
                ticker=ticker,
                views=post.get("views", 0),
                likes=post.get("likes", 0),
                reposts=post.get("reposts", 0),
                author_followers=post.get("author_followers", 0),
            )
            item = dict(post)
            item.update(res)
            scored_posts.append(item)

        scored_news = []
        for n in news:
            news_text = f"{n.get('title', '')} {n.get('text', '')}".strip()
            # News items typically have higher default institutional reach
            res = self.score_text(
                text=news_text,
                ticker=ticker,
                views=1500,
                likes=10,
                reposts=5,
                author_followers=2000,
            )
            item = dict(n)
            item.update(res)
            scored_news.append(item)

        output = {
            "ticker": ticker,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "scored_posts": scored_posts,
            "scored_news": scored_news,
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return output

