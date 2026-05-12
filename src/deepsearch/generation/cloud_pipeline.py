"""Cloud fallback pipeline — PII scrubbing + cloud LLM dispatch.

This module sits above CloudAPIBackend and handles:
  1. PII detection in context + question (presidio, optional)
  2. Scrubbing or rejecting requests that contain too much sensitive data
  3. Dispatching the clean prompt to the cloud backend
  4. Audit-logging every cloud call (question hash, entity types found)

The class is intentionally separate from LLMPipeline so it can be
unit-tested without a live API key.

Usage:
    pipeline = CloudPipeline(settings=cfg)
    answer = pipeline.generate(question="What is the revenue?", context="...")
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

log = logging.getLogger(__name__)

# PII entity types that should block the request entirely.
# Less sensitive types (like dates, URLs) are scrubbed but don't block.
_BLOCK_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "CREDIT_CARD",
        "CRYPTO",
        "IBAN_CODE",
        "IP_ADDRESS",         # rare in docs; can be sensitive
        "US_SSN",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "AU_TFN",
        "AU_ABN",
        "SG_NRIC_FIN",
    }
)

# If the fraction of scrubbed tokens exceeds this, reject the request
# (the context is too sensitive to send usefully).
_MAX_SCRUB_RATIO = 0.4


@dataclass
class CloudCallRecord:
    """Audit record for a single cloud dispatch."""
    question_hash: str             # sha256[:16] of original question
    entity_types_found: list[str]  # PII types detected (not the values)
    context_tokens_before: int
    context_tokens_after: int
    scrub_ratio: float
    blocked: bool
    model_used: str = ""


@dataclass
class CloudPipelineResult:
    """Result from a cloud pipeline generate() call."""
    answer: str
    blocked: bool = False
    block_reason: str = ""
    audit: Optional[CloudCallRecord] = None


class PIIScrubber:
    """Thin wrapper around presidio AnalyzerEngine + AnonymizerEngine.

    Falls back to a no-op if presidio is not installed.
    """

    def __init__(self) -> None:
        self._available = False
        self._analyzer = None
        self._anonymizer = None
        self._load()

    def _load(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine        # type: ignore[import]
            from presidio_anonymizer import AnonymizerEngine    # type: ignore[import]

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._available = True
            log.info("PIIScrubber: presidio loaded")
        except ImportError:
            log.warning(
                "presidio not installed — PII scrubbing disabled for cloud pipeline. "
                "Run: pip install presidio-analyzer presidio-anonymizer"
            )

    @property
    def available(self) -> bool:
        return self._available

    def analyze(self, text: str, language: str = "en") -> list:
        """Return list of presidio RecognizerResult objects (empty if unavailable)."""
        if not self._available:
            return []
        return self._analyzer.analyze(text=text, language=language)

    def anonymize(self, text: str, results: list) -> str:
        """Anonymize detected entities.  Returns original text if unavailable."""
        if not self._available or not results:
            return text
        return self._anonymizer.anonymize(text=text, analyzer_results=results).text

    def scrub(self, text: str, language: str = "en") -> tuple[str, list[str]]:
        """Analyze + anonymize in one call.

        Returns:
            (scrubbed_text, entity_types_found)
        """
        results = self.analyze(text, language)
        entity_types = sorted({r.entity_type for r in results})
        scrubbed = self.anonymize(text, results)
        return scrubbed, entity_types


class CloudPipeline:
    """High-level cloud fallback pipeline with PII gate.

    Args:
        settings:    App settings (Settings object with .cloud sub-config).
        scrubber:    PIIScrubber instance.  Created automatically if None.
        model_id:    Cloud model identifier.  Falls back to settings if None.
        api_key:     Cloud API key.  Falls back to settings if None.
    """

    def __init__(
        self,
        settings=None,
        scrubber: Optional[PIIScrubber] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._cfg = settings
        self._scrubber = scrubber or PIIScrubber()
        self._model_id = model_id
        self._api_key = api_key
        self._audit_log: list[CloudCallRecord] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        question: str,
        context: str,
        system_prompt: str = "",
        language: str = "en",
    ) -> CloudPipelineResult:
        """Scrub PII, then generate an answer via the cloud backend.

        Returns a CloudPipelineResult.  When blocked=True, answer is empty
        and block_reason explains why.
        """
        # ── PII gate ─────────────────────────────────────────────────────
        scrubbed_ctx, ctx_entities = self._scrubber.scrub(context, language)
        scrubbed_q, q_entities    = self._scrubber.scrub(question, language)
        all_entities = sorted(set(ctx_entities + q_entities))

        tokens_before = len(context.split()) + len(question.split())
        tokens_after  = len(scrubbed_ctx.split()) + len(scrubbed_q.split())
        scrub_ratio   = 1.0 - (tokens_after / max(tokens_before, 1))

        blocked_types = [e for e in all_entities if e in _BLOCK_ENTITY_TYPES]

        audit = CloudCallRecord(
            question_hash=hashlib.sha256(question.encode()).hexdigest()[:16],
            entity_types_found=all_entities,
            context_tokens_before=tokens_before,
            context_tokens_after=tokens_after,
            scrub_ratio=round(scrub_ratio, 4),
            blocked=bool(blocked_types) or scrub_ratio > _MAX_SCRUB_RATIO,
        )

        if blocked_types:
            reason = f"Request blocked: sensitive PII detected ({', '.join(blocked_types)})"
            log.warning("Cloud call blocked — %s (hash=%s)", reason, audit.question_hash)
            self._audit_log.append(audit)
            return CloudPipelineResult(answer="", blocked=True, block_reason=reason, audit=audit)

        if scrub_ratio > _MAX_SCRUB_RATIO:
            reason = (
                f"Request blocked: scrub ratio {scrub_ratio:.0%} exceeds "
                f"maximum {_MAX_SCRUB_RATIO:.0%}"
            )
            log.warning("Cloud call blocked — %s (hash=%s)", reason, audit.question_hash)
            self._audit_log.append(audit)
            return CloudPipelineResult(answer="", blocked=True, block_reason=reason, audit=audit)

        if all_entities:
            log.info(
                "Cloud call: PII scrubbed %s | hash=%s",
                all_entities, audit.question_hash,
            )

        # ── Dispatch to cloud backend ─────────────────────────────────────
        try:
            answer = self._call_backend(scrubbed_q, scrubbed_ctx, system_prompt)
            audit.model_used = self._resolve_model_id()
            self._audit_log.append(audit)
            return CloudPipelineResult(answer=answer, audit=audit)
        except Exception as exc:
            log.error("Cloud call failed (hash=%s): %s", audit.question_hash, exc)
            raise

    def stream(
        self,
        question: str,
        context: str,
        system_prompt: str = "",
        language: str = "en",
    ) -> Iterator[str]:
        """Streaming variant.  Raises immediately if the request is blocked."""
        scrubbed_ctx, ctx_entities = self._scrubber.scrub(context, language)
        scrubbed_q, q_entities    = self._scrubber.scrub(question, language)
        all_entities = sorted(set(ctx_entities + q_entities))
        blocked_types = [e for e in all_entities if e in _BLOCK_ENTITY_TYPES]

        if blocked_types:
            raise PermissionError(
                f"Cloud stream blocked: sensitive PII detected ({', '.join(blocked_types)})"
            )

        yield from self._stream_backend(scrubbed_q, scrubbed_ctx, system_prompt)

    @property
    def audit_log(self) -> list[CloudCallRecord]:
        """All audit records from this session (read-only copy)."""
        return list(self._audit_log)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_model_id(self) -> str:
        if self._model_id:
            return self._model_id
        if self._cfg is not None:
            try:
                return self._cfg.cloud.model
            except AttributeError:
                pass
        return "claude-haiku-4-5-20251001"

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        if self._cfg is not None:
            try:
                return self._cfg.cloud_api_key
            except AttributeError:
                pass
        return ""

    def _build_prompt(self, question: str, context: str, system_prompt: str) -> str:
        parts: list[str] = []
        if system_prompt:
            parts.append(system_prompt)
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"Question: {question}")
        return "\n\n".join(parts)

    def _call_backend(self, question: str, context: str, system_prompt: str) -> str:
        from ..backends.cloud_backend import CloudAPIBackend
        from ..backends.base import GenerationConfig

        backend = CloudAPIBackend()
        model = self._resolve_model_id()
        api_key = self._resolve_api_key()
        backend.load(model, api_key=api_key)

        prompt = self._build_prompt(question, context, system_prompt)
        cfg = GenerationConfig(max_tokens=1024, temperature=0.2)
        try:
            return backend.generate(prompt, cfg)
        finally:
            backend.unload()

    def _stream_backend(
        self, question: str, context: str, system_prompt: str
    ) -> Iterator[str]:
        from ..backends.cloud_backend import CloudAPIBackend
        from ..backends.base import GenerationConfig

        backend = CloudAPIBackend()
        model = self._resolve_model_id()
        api_key = self._resolve_api_key()
        backend.load(model, api_key=api_key)

        prompt = self._build_prompt(question, context, system_prompt)
        cfg = GenerationConfig(max_tokens=1024, temperature=0.2, stream=True)
        try:
            yield from backend.stream(prompt, cfg)
        finally:
            backend.unload()
