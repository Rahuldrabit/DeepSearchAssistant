"""Multi-language support — language detection and model routing.

Provides:
  - Language detection (langdetect or langid, with regex fallback)
  - Embedding model selection per detected language
  - Query translation via LLM when source and index languages differ
  - Multilingual tokenisation hints for BM25 stopword filtering

Design philosophy:
  - All heavy deps (langdetect, langid) are lazy-imported.
  - Falls back gracefully through: langdetect → langid → regex heuristics.
  - Never raises on unknown language; defaults to "en" with a warning.

Usage::

    lm = LanguageManager()
    lang = lm.detect("Quelle est la capitale de la France?")  # → "fr"

    model = lm.embedding_model_for(lang)
    # → "paraphrase-multilingual-MiniLM-L12-v2" for non-English

    # Translate if needed
    en_query = lm.translate_to_english(query="...", lang="de", llm=llm_backend)
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# ── Embedding model routing ───────────────────────────────────────────────────
# English gets the smaller, faster monolingual model.
# Everything else uses the multilingual model (covers 50+ languages).
_ENGLISH_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_MULTILINGUAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Languages well-served by the multilingual model (non-exhaustive)
_MULTILINGUAL_LANGUAGES = frozenset({
    "fr", "de", "es", "it", "pt", "nl", "pl", "ru", "zh-cn", "zh-tw",
    "ja", "ko", "ar", "tr", "sv", "da", "fi", "no", "cs", "sk",
    "hu", "ro", "hr", "bg", "el", "he", "hi", "th", "vi", "id",
    "ms", "uk", "ca", "lt", "lv", "et", "sl", "mk",
})

# BM25 stopwords per language (minimal, keep as tuples for speed)
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "is", "was", "are", "were",
    }),
    "fr": frozenset({
        "le", "la", "les", "un", "une", "des", "et", "ou", "mais",
        "dans", "sur", "de", "du", "au", "aux", "par", "est", "sont",
    }),
    "de": frozenset({
        "der", "die", "das", "ein", "eine", "und", "oder", "aber",
        "in", "auf", "an", "zu", "von", "mit", "ist", "sind",
    }),
    "es": frozenset({
        "el", "la", "los", "las", "un", "una", "unos", "unas", "y",
        "o", "pero", "en", "de", "a", "por", "para", "es", "son",
    }),
    "zh-cn": frozenset({"的", "了", "和", "是", "在", "我", "有", "他", "这", "中"}),
}

# Simple script-level heuristics for fast pre-detection
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def _script_hint(text: str) -> Optional[str]:
    """Fast script-level language hint (not reliable for specific language)."""
    sample = text[:200]
    if _CJK_RE.search(sample):
        return "zh-cn"   # could be zh/ja/ko — langdetect will refine
    if _ARABIC_RE.search(sample):
        return "ar"
    if _CYRILLIC_RE.search(sample):
        return "ru"
    if not _LATIN_RE.search(sample) and len(sample.strip()) > 5:
        return "xx"  # unknown non-Latin
    return None


class LanguageManager:
    """Language detection and routing for multi-language queries.

    Args:
        default_language: Fallback when detection fails (default "en").
        min_text_length:  Minimum characters needed for reliable detection.
    """

    def __init__(
        self,
        default_language: str = "en",
        min_text_length: int = 20,
    ) -> None:
        self._default = default_language
        self._min_len = min_text_length

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #

    def detect(self, text: str) -> str:
        """Detect the language of *text*.

        Detection chain:
          1. Script hint (fast, zero-dep).
          2. langdetect (if installed).
          3. langid (if installed).
          4. Default fallback.
        """
        if not text or len(text.strip()) < self._min_len:
            log.debug("Text too short for reliable detection — defaulting to %s", self._default)
            return self._default

        hint = _script_hint(text)
        if hint == "xx":
            return self._default  # unrecognised non-Latin

        # Try langdetect
        lang = self._try_langdetect(text)
        if lang:
            return lang

        # Try langid
        lang = self._try_langid(text)
        if lang:
            return lang

        # Fall back to script hint or default
        return hint or self._default

    def detect_batch(self, texts: list[str]) -> list[str]:
        return [self.detect(t) for t in texts]

    # ------------------------------------------------------------------ #
    # Model routing                                                        #
    # ------------------------------------------------------------------ #

    def embedding_model_for(self, language: str) -> str:
        """Return the recommended embedding model name for *language*."""
        if language.lower() in ("en", ""):
            return _ENGLISH_EMBEDDING_MODEL
        return _MULTILINGUAL_EMBEDDING_MODEL

    def stopwords_for(self, language: str) -> frozenset[str]:
        """Return BM25 stopwords for *language*."""
        return _STOPWORDS.get(language, _STOPWORDS.get("en", frozenset()))

    def is_supported(self, language: str) -> bool:
        return language in ("en",) | _MULTILINGUAL_LANGUAGES

    # ------------------------------------------------------------------ #
    # Translation                                                          #
    # ------------------------------------------------------------------ #

    def translate_to_english(
        self,
        query: str,
        lang: str,
        llm,
    ) -> str:
        """Translate *query* from *lang* to English using the LLM.

        Returns the original query unchanged if lang is already English or
        the translation fails.
        """
        if lang.lower().startswith("en"):
            return query

        from ..backends.base import GenerationConfig
        system = (
            "You are a precise translator.  Translate the following text to English.  "
            "Output ONLY the translation — no explanation, no quotes."
        )
        cfg = GenerationConfig(max_tokens=200, temperature=0.1, stop=["\n\n"])
        prompt = llm.build_prompt(system, query)
        try:
            translated = llm.generate(prompt, cfg).strip()
            if translated:
                log.debug("Translated '%s' (%s) → '%s'", query[:40], lang, translated[:40])
                return translated
        except Exception as exc:
            log.warning("Translation failed: %s", exc)
        return query

    # ------------------------------------------------------------------ #
    # Private detection helpers                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _try_langdetect(text: str) -> Optional[str]:
        try:
            from langdetect import detect, LangDetectException  # type: ignore[import]
            return detect(text)
        except Exception:
            return None

    @staticmethod
    def _try_langid(text: str) -> Optional[str]:
        try:
            import langid  # type: ignore[import]
            lang, _ = langid.classify(text)
            return lang
        except Exception:
            return None
