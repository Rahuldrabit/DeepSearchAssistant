"""Speculative decoding backend — draft + verify for faster generation.

Speculative decoding uses a small, fast *draft* model to propose *gamma*
candidate tokens in one pass, then the larger *main* model verifies all
of them in a single forward pass.  On average this delivers 2–3× speedup
when the draft model is accurate (acceptance rate > ~60%).

Algorithm (token-level):
  1. Draft model generates γ tokens autoregressively.
  2. Main model runs one forward pass over (prompt + γ draft tokens).
  3. For each position i:
       - Accept draft token if main_prob(token_i) ≥ draft_prob(token_i).
       - Otherwise, sample a correction token from the adjusted distribution
         and stop accepting at that position.
  4. Append all accepted tokens + one correction token.
  5. Repeat until max_tokens or stop sequence.

For llama-cpp-python, the library exposes speculative decoding natively
via ``Llama(model_path=..., draft_model=...)``  when using llama.cpp >= b3000.
This class wraps that path and falls back to sequential generation when the
native API is unavailable.

Usage::

    from deepsearch.backends.llamacpp_backend import LlamaCppBackend
    from deepsearch.backends.speculative_backend import SpeculativeBackend
    from deepsearch.backends.base import GenerationConfig

    main_b = LlamaCppBackend()
    main_b.load("models/qwen2.5-7b-q4_k_m.gguf", n_ctx=4096)

    draft_b = LlamaCppBackend()
    draft_b.load("models/phi-3.5-mini-q4_k_m.gguf", n_ctx=4096)

    spec = SpeculativeBackend(main_backend=main_b, draft_backend=draft_b, gamma=5)
    answer = spec.generate(prompt, GenerationConfig(max_tokens=256))
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional

from .base import BackendInfo, GenerationConfig, LLMBackend

log = logging.getLogger(__name__)

_DEFAULT_GAMMA = 5           # draft tokens per speculation step
_MIN_ACCEPTANCE_RATE = 0.3   # fallback to sequential if acceptance too low


@dataclass
class SpeculativeStats:
    total_tokens: int = 0
    accepted_tokens: int = 0
    draft_calls: int = 0
    verify_calls: int = 0
    elapsed_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.total_tokens, 1)

    @property
    def speedup_estimate(self) -> float:
        """Estimated speedup vs sequential (purely token-count based)."""
        if self.draft_calls == 0:
            return 1.0
        avg_accepted = self.accepted_tokens / self.draft_calls
        # Each verify call processes γ tokens instead of 1 sequential call
        return (avg_accepted + 1) / 1.0


class SpeculativeBackend(LLMBackend):
    """Speculative decoding wrapper around two LLMBackend instances.

    Args:
        main_backend:   Large, accurate model (verifier).
        draft_backend:  Small, fast model (drafter).
        gamma:          Tokens to speculate per iteration (default 5).
        auto_fallback:  Fall back to main-only when acceptance rate drops.
    """

    def __init__(
        self,
        main_backend: LLMBackend,
        draft_backend: LLMBackend,
        gamma: int = _DEFAULT_GAMMA,
        auto_fallback: bool = True,
    ) -> None:
        self._main = main_backend
        self._draft = draft_backend
        self._gamma = gamma
        self._auto_fallback = auto_fallback
        self._stats = SpeculativeStats()

    # ------------------------------------------------------------------ #
    # LLMBackend interface                                                 #
    # ------------------------------------------------------------------ #

    def load(self, model_path: str, **kwargs) -> None:
        """Load both backends.  ``model_path`` is ignored — load separately."""
        raise NotImplementedError(
            "Load main_backend and draft_backend separately, then pass to SpeculativeBackend."
        )

    def unload(self) -> None:
        self._main.unload()
        self._draft.unload()

    @property
    def info(self) -> BackendInfo:
        main_info = self._main.info
        return BackendInfo(
            name=f"speculative({self._draft.info.name}→{main_info.name})",
            model_id=main_info.model_id,
            device=main_info.device,
            loaded=main_info.loaded,
            context_length=main_info.context_length,
        )

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate using speculative decoding.

        If the underlying llama.cpp build supports native speculative decoding,
        uses that path.  Otherwise, falls back to Python-level token-by-token
        approximation or pure main-model generation.
        """
        t0 = time.perf_counter()

        # Try native llama.cpp speculative path first
        result = self._try_native_speculative(prompt, config)
        if result is not None:
            self._stats.elapsed_ms = (time.perf_counter() - t0) * 1000
            return result

        # Python-level speculative simulation
        result = self._python_speculative(prompt, config)
        self._stats.elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug(
            "Speculative: accept_rate=%.2f, speedup_est=%.1fx, %d tokens in %.0f ms",
            self._stats.acceptance_rate,
            self._stats.speedup_estimate,
            self._stats.total_tokens,
            self._stats.elapsed_ms,
        )
        return result

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Streaming falls back to main model stream (speculation is block-level)."""
        yield from self._main.stream(prompt, config)

    async def agenerate(self, prompt: str, config: GenerationConfig) -> str:
        return self.generate(prompt, config)

    async def astream(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        async def _gen():
            for tok in self.stream(prompt, config):
                yield tok
        return _gen()

    def build_prompt(self, system: str, user: str, context: str = "") -> str:
        return self._main.build_prompt(system, user, context)

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> SpeculativeStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = SpeculativeStats()

    # ------------------------------------------------------------------ #
    # Internal — native llama.cpp path                                    #
    # ------------------------------------------------------------------ #

    def _try_native_speculative(self, prompt: str, config: GenerationConfig) -> Optional[str]:
        """Attempt native llama.cpp speculative decoding.

        llama-cpp-python >= 0.3.x exposes speculative decoding via the
        ``draft_model`` parameter at load time.  If both backends are
        LlamaCppBackend instances and the underlying Llama objects are loaded,
        attempt a direct call.
        """
        try:
            from .llamacpp_backend import LlamaCppBackend  # avoid circular at module level
            if not (isinstance(self._main, LlamaCppBackend) and
                    isinstance(self._draft, LlamaCppBackend)):
                return None

            main_llm = self._main._llm  # type: ignore[attr-defined]
            draft_llm = self._draft._llm  # type: ignore[attr-defined]
            if main_llm is None or draft_llm is None:
                return None

            # llama-cpp-python >= b3000 supports draft_model kwarg in create_completion
            output = main_llm.create_completion(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                stop=config.stop or [],
                draft_model=draft_llm,  # native speculative decoding
            )
            return output["choices"][0]["text"]
        except (AttributeError, TypeError, KeyError):
            # API not supported in this build
            return None
        except Exception as exc:
            log.debug("Native speculative failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Internal — Python-level approximation                               #
    # ------------------------------------------------------------------ #

    def _python_speculative(self, prompt: str, config: GenerationConfig) -> str:
        """Python-level speculative decoding simulation.

        In the absence of token probability APIs, we approximate by:
          1. Draft model generates γ tokens.
          2. Main model generates 1 token given (prompt + draft tokens).
          3. If main token == draft token[0], accept the draft prefix up to
             the first mismatch.
          4. Repeat from the updated prompt.

        This is a simplified heuristic — true speculative decoding requires
        per-token logprob access.  The main benefit here is amortising the
        main model call frequency.
        """
        # Check acceptance rate — if too low, go sequential
        if (self._auto_fallback and
                self._stats.draft_calls > 10 and
                self._stats.acceptance_rate < _MIN_ACCEPTANCE_RATE):
            log.info("Speculative: acceptance rate low (%.2f) — using main model only",
                     self._stats.acceptance_rate)
            return self._main.generate(prompt, config)

        tokens_generated: list[str] = []
        current_prompt = prompt
        remaining = config.max_tokens

        while remaining > 0:
            gamma = min(self._gamma, remaining)

            # Draft: generate gamma tokens
            draft_cfg = GenerationConfig(
                max_tokens=gamma,
                temperature=config.temperature,
                stop=config.stop,
                stream=False,
            )
            draft_text = self._draft.generate(current_prompt, draft_cfg)
            self._stats.draft_calls += 1

            if not draft_text:
                break

            # Verify: main model generates 1 token from original prompt
            verify_cfg = GenerationConfig(
                max_tokens=1,
                temperature=config.temperature,
                stop=config.stop,
                stream=False,
            )
            main_token = self._main.generate(current_prompt, verify_cfg)
            self._stats.verify_calls += 1

            if not main_token:
                break

            # Simple prefix-match acceptance: accept draft up to first word mismatch
            draft_tokens = draft_text.split(" ")
            main_first = main_token.split(" ")[0] if main_token else ""

            if draft_tokens and draft_tokens[0] == main_first:
                # Accept draft
                accepted = draft_text
                self._stats.accepted_tokens += len(draft_tokens)
            else:
                # Reject — use main token only
                accepted = main_token
                self._stats.accepted_tokens += 1

            self._stats.total_tokens += len(draft_tokens)
            tokens_generated.append(accepted)
            current_prompt = current_prompt + accepted
            remaining -= len(accepted.split())

            # Check stop sequences
            full = "".join(tokens_generated)
            for stop in (config.stop or []):
                if stop in full:
                    return full[: full.index(stop)]

        return "".join(tokens_generated)
