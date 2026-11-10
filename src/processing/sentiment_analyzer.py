"""LLM-based and domain-specific sentiment analysis for financial text."""

from datetime import datetime, timezone
import json
import math
import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class SentimentAnalyzer:
    """Evaluates sentiment, confidence, relevance, and aspect classification."""

    # Curated financial sentiment dictionaries for Chinese/Cantonese, Korean, and English
    LEXICON_BULLISH = {
        # Traditional Chinese / Cantonese
        "突破": 0.8, "回購": 0.7, "增長": 0.6, "掃貨": 0.8, "看好": 0.7, "睇好": 0.8,
        "淨買入": 0.7, "超預期": 0.9, "爆升": 0.9, "鐵底": 0.7, "新高": 0.8,
        "大升": 0.8, "加倉": 0.7, "好消息": 0.6, "賺錢": 0.6, "穩如泰山": 0.8,
        "交付量破紀錄": 0.9, "創新高": 0.8, "價值重估": 0.7, "領先": 0.6,

        # Korean
        "가즈아": 0.8, "매수": 0.6, "호실적": 0.8, "승인": 0.7, "턴어라운드": 0.8,
        "반등": 0.7, "신고가": 0.9, "수주": 0.7, "상향": 0.7, "급등": 0.8,
        "쌍끌이": 0.8, "바닥 확인": 0.7, "숏스퀴즈": 0.8, "흑자전환": 0.9,
        "회복": 0.6, "쇄신": 0.5, "독점": 0.7, "상승": 0.6,

        # English
        "surge": 0.8, "beat": 0.8, "growth": 0.6, "record": 0.7, "upgrade": 0.8,
        "bullish": 0.9, "profit": 0.6, "expansion": 0.6, "outperform": 0.8,
        "soar": 0.8, "rally": 0.8, "dividend": 0.5, "breakthrough": 0.8,
    }

    LEXICON_BEARISH = {
        # Traditional Chinese / Cantonese
        "沽空": -0.8, "沽期指": -0.7, "跌穿": -0.8, "壓力": -0.5, "減持": -0.6,
        "價格戰": -0.6, "拖低": -0.6, "虧損": -0.8, "暴跌": -0.9, "斬倉": -0.8,
        "受壓": -0.6, "下調": -0.7, "看淡": -0.7, "睇淡": -0.8, "大跌": -0.8,
        "利潤回落": -0.6, "短缺": -0.5, "沽售": -0.7,

        # Korean
        "매도": -0.6, "곡소리": -0.8, "하락": -0.6, "적자": -0.8, "급락": -0.9,
        "과열": -0.5, "신저가": -0.9, "우려": -0.6, "하향": -0.7, "공매도": -0.6,
        "손절": -0.7, "물린": -0.7, "불확실": -0.6, "캐즘": -0.5, "정체": -0.6,

        # English
        "plunge": -0.8, "miss": -0.8, "drop": -0.6, "slump": -0.8, "downgrade": -0.8,
        "bearish": -0.9, "loss": -0.7, "headwind": -0.6, "decline": -0.6,
        "investigation": -0.7, "penalty": -0.7, "risk": -0.5, "selloff": -0.8,
    }

    NEGATORS = ["不", "未", "無", "唔", "非", "안", "못", "아닌", "not", "never", "no", "without"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._client = None
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize Google GenAI client if api key is valid."""
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    def calculate_engagement_weight(self, item: Dict[str, Any]) -> float:
        """Calculate engagement weight based on likes, reposts, and views.

        Formula: w = log(1 + likes + 2*reposts + 0.05*views)
        Normalized to a reasonable scaling range [1.0, 5.0].
        """
        likes = float(item.get("likes", 0))
        reposts = float(item.get("reposts", 0))
        views = float(item.get("views", 0))

        raw_metric = 1.0 + likes + (2.0 * reposts) + (0.05 * views)
        weight = math.log(max(raw_metric, 1.0))
        # Keep weight within [1.0, 5.0] to avoid single viral posts completely dominating
        return round(max(1.0, min(weight, 5.0)), 3)

    def analyze_item(self, item: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """Analyze a news article or social post, returning structured sentiment payload."""
        text = str(item.get("text") or item.get("title") or "")

        # Try LLM inference first if client is configured
        analysis = None
        if self._client:
            try:
                analysis = self._call_gemini_sentiment(text, ticker)
            except Exception:
                analysis = None

        if not analysis:
            analysis = self._rule_based_sentiment(text, ticker)

        weight = self.calculate_engagement_weight(item)
        weighted_score = round(analysis["sentiment_score"] * weight, 4)

        result = dict(item)
        result.update({
            "sentiment_score": analysis["sentiment_score"],
            "confidence": analysis["confidence"],
            "relevance": analysis["relevance"],
            "reasoning": analysis["reasoning"],
            "aspects": analysis["aspects"],
            "engagement_weight": weight,
            "weighted_score": weighted_score,
        })
        return result

    def analyze_batch(self, items: List[Dict[str, Any]], ticker: str) -> List[Dict[str, Any]]:
        """Batch process a list of items."""
        return [self.analyze_item(item, ticker) for item in items]

    def _call_gemini_sentiment(self, text: str, ticker: str) -> Optional[Dict[str, Any]]:
        """Call Gemini model using google-genai SDK with structured output prompt."""
        prompt = (
            f"Analyze the financial sentiment of the following post/news regarding equity asset {ticker}.\n"
            f"Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "sentiment_score": <float between -1.0 (strongly bearish) and +1.0 (strongly bullish)>,\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "relevance": <float between 0.0 and 1.0>,\n'
            '  "reasoning": "<concise explanation in under 20 words>",\n'
            '  "aspects": ["<list of tags: e.g. earnings, product, ai, macro, technical>"]\n'
            "}\n\n"
            f"Text: {text}"
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        resp_text = response.text.strip()
        # Clean json backticks if any
        if resp_text.startswith("```json"):
            resp_text = resp_text[7:]
        if resp_text.startswith("```"):
            resp_text = resp_text[3:]
        if resp_text.endswith("```"):
            resp_text = resp_text[:-3]

        data = json.loads(resp_text.strip())
        score = float(max(-1.0, min(1.0, data.get("sentiment_score", 0.0))))
        conf = float(max(0.0, min(1.0, data.get("confidence", 0.8))))
        rel = float(max(0.0, min(1.0, data.get("relevance", 0.9))))

        return {
            "sentiment_score": round(score, 3),
            "confidence": round(conf, 3),
            "relevance": round(rel, 3),
            "reasoning": str(data.get("reasoning", "LLM classified sentiment")),
            "aspects": list(data.get("aspects", ["general"])),
        }

    def _rule_based_sentiment(self, text: str, ticker: str) -> Dict[str, Any]:
        """Resilient rule-based lexicon scorer for Chinese, Korean, and English text."""
        if not text:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.5,
                "relevance": 0.5,
                "reasoning": "Empty content",
                "aspects": ["neutral"],
            }

        score = 0.0
        bull_matches = []
        bear_matches = []

        # Check bullish keywords
        for word, weight in self.LEXICON_BULLISH.items():
            if word in text:
                score += weight
                bull_matches.append(word)

        # Check bearish keywords
        for word, weight in self.LEXICON_BEARISH.items():
            if word in text:
                score += weight  # weight is already negative
                bear_matches.append(word)

        # Check negations nearby
        has_negation = any(neg in text for neg in self.NEGATORS)
        if has_negation and (bull_matches or bear_matches):
            score *= 0.6  # attenuate confidence on negation

        # Normalize score into [-1.0, 1.0] using hyperbolic tangent
        norm_score = math.tanh(score)
        norm_score = max(-1.0, min(1.0, norm_score))

        # Assign confidence based on keyword count
        total_hits = len(bull_matches) + len(bear_matches)
        confidence = min(0.95, 0.60 + (0.10 * total_hits)) if total_hits > 0 else 0.50

        # Aspects detection
        aspects = []
        if any(w in text for w in ["AI", "雲", "반도체", "HBM", "기술", "tech"]):
            aspects.append("tech_ai")
        if any(w in text for w in ["業績", "利潤", "실적", "영업익", "revenue", "guidance"]):
            aspects.append("earnings")
        if any(w in text for w in ["外資", "北水", "외인", "기관", "funds", "flow"]):
            aspects.append("capital_flows")
        if not aspects:
            aspects.append("general")

        reasoning = f"Matched bullish={bull_matches[:2]}, bearish={bear_matches[:2]}"

        return {
            "sentiment_score": round(norm_score, 3),
            "confidence": round(confidence, 3),
            "relevance": 0.85 if total_hits > 0 else 0.60,
            "reasoning": reasoning,
            "aspects": aspects,
        }

