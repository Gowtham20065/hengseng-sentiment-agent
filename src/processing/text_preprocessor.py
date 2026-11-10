"""Text preprocessor for multilingual financial text (Cantonese, Traditional Chinese, Korean, English)."""

import html
import re
from typing import Dict, List, Optional, Set, Tuple


class TextPreprocessor:
    """Cleans, normalizes, and extracts financial entities from social and news text."""

    def __init__(self, keyword_map: Optional[Dict[str, List[str]]] = None):
        """keyword_map: Mapping of ticker -> list of aliases, cashtags, and slang keywords."""
        self.keyword_map = keyword_map or {}
        # Precompile standard cleaning regex
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._html_tag_pattern = re.compile(r"<.*?>")
        self._cashtag_pattern = re.compile(r"\$[A-Za-z0-9.]+")
        self._whitespace_pattern = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """Strip URLs, HTML tags, normalize whitespace, and decode entities."""
        if not text:
            return ""

        # Decode HTML entities (e.g. &amp;, &lt;)
        clean = html.unescape(text)
        # Remove URLs
        clean = self._url_pattern.sub("", clean)
        # Remove HTML tags
        clean = self._html_tag_pattern.sub("", clean)
        # Normalize whitespace
        clean = self._whitespace_pattern.sub(" ", clean).strip()
        return clean

    def extract_cashtags(self, text: str) -> List[str]:
        """Extract uppercase cashtags like $0700, $TCEHY, $005930."""
        if not text:
            return []
        tags = self._cashtag_pattern.findall(text)
        return [t.upper() for t in tags]

    def match_asset_keywords(self, text: str, ticker: str) -> Tuple[bool, List[str]]:
        """Check if text references the target ticker, aliases, or local slang."""
        matched: List[str] = []
        clean = text.lower()
        clean_upper = text.upper()

        # Always check raw ticker (e.g. 0700.HK or 0700)
        raw_code = ticker.split(".")[0]
        if raw_code in text or ticker in clean_upper:
            matched.append(raw_code)

        keywords = self.keyword_map.get(ticker, [])
        for kw in keywords:
            if kw.startswith("$"):
                if kw.upper() in clean_upper:
                    matched.append(kw)
            else:
                if kw.lower() in clean:
                    matched.append(kw)

        # Remove duplicates while preserving order
        unique_matches = list(dict.fromkeys(matched))
        return (len(unique_matches) > 0, unique_matches)

    def is_korean(self, text: str) -> bool:
        """Detect presence of Korean Hangul characters."""
        return any("\uac00" <= ch <= "\ud7a3" for ch in text)

    def is_chinese(self, text: str) -> bool:
        """Detect presence of CJK Unified Ideographs."""
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

