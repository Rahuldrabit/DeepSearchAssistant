"""Unit tests for all 7 future-extension modules.

Covers:
  1. HyDE retriever
  2. Sub-query decomposer
  3. Agentic retriever
  4. GraphRAG (KnowledgeGraph + GraphRAGRetriever)
  5. PartitionedVectorStore routing
  6. SpeculativeBackend
  7. LanguageManager
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_hit(chunk_id: str, score: float = 0.8, text: str = "sample text") -> dict:
    return {
        "id": chunk_id,
        "score": score,
        "payload": {"chunk_id": chunk_id, "text": text, "file_id": "f1", "file_type": "pdf"},
    }


def _make_llm(response: str = "mocked answer") -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = response
    llm.build_prompt.side_effect = lambda sys_, user, context="": f"{sys_}\n{user}"
    return llm


def _make_search(hits: list[dict] | None = None) -> MagicMock:
    search = MagicMock()
    default = hits if hits is not None else [_make_hit("c1"), _make_hit("c2")]
    search.search.return_value = default
    search.search_dense_only.return_value = default
    search._build_filter.return_value = None
    search._vs = MagicMock()
    search._vs.search_dense.return_value = default
    return search


def _make_embedder() -> MagicMock:
    emb = MagicMock()
    emb.embed_query.return_value = ([0.1] * 384, [0, 1], [0.5, 0.3])
    return emb


# ═══════════════════════════════════════════════════════════════════════════ #
# 1. HyDE                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestHyDE:
    def _make(self, hits=None, llm_response="hypothetical answer text"):
        from deepsearch.retrieval.hyde import HyDERetriever
        return HyDERetriever(
            search=_make_search(hits),
            llm=_make_llm(llm_response),
            embedder=_make_embedder(),
        )

    def test_retrieve_returns_hits(self):
        hyde = self._make()
        hits = hyde.retrieve("What is X?", top_k=5)
        assert isinstance(hits, list)
        assert len(hits) == 2

    def test_llm_called_for_hypothesis(self):
        hyde = self._make()
        hyde.retrieve("Some question", top_k=5)
        hyde._llm.generate.assert_called_once()

    def test_hypothesis_text_used_for_embedding(self):
        hyde = self._make(llm_response="specific hypothetical passage")
        hyde.retrieve("question", top_k=5)
        call_args = hyde._embedder.embed_query.call_args[0][0]
        assert "specific hypothetical passage" in call_args

    def test_fallback_on_llm_error(self):
        from deepsearch.retrieval.hyde import HyDERetriever
        llm = _make_llm()
        llm.generate.side_effect = RuntimeError("LLM unavailable")
        search = _make_search()
        hyde = HyDERetriever(search=search, llm=llm, embedder=_make_embedder(), fallback=True)
        hits = hyde.retrieve("question", top_k=5)
        search.search_dense_only.assert_called_once()
        assert isinstance(hits, list)

    def test_no_fallback_raises_on_llm_error(self):
        from deepsearch.retrieval.hyde import HyDERetriever
        llm = _make_llm()
        llm.generate.side_effect = RuntimeError("LLM unavailable")
        hyde = HyDERetriever(search=_make_search(), llm=llm, embedder=_make_embedder(), fallback=False)
        with pytest.raises(RuntimeError):
            hyde.retrieve("question")

    def test_generate_hypothesis_returns_string(self):
        hyde = self._make(llm_response="the capital is Paris")
        hyp = hyde.generate_hypothesis("What is the capital of France?")
        assert isinstance(hyp, str)
        assert "Paris" in hyp

    def test_multi_hypothesis_picks_longest(self):
        from deepsearch.retrieval.hyde import HyDERetriever
        responses = iter(["short", "this is a much longer hypothetical answer text"])
        llm = MagicMock()
        llm.generate.side_effect = lambda p, c: next(responses)
        llm.build_prompt.side_effect = lambda s, u, ctx="": u
        hyde = HyDERetriever(search=_make_search(), llm=llm, embedder=_make_embedder(), num_hypotheses=2)
        hyp = hyde._average_hypothesis("question")
        assert "longer" in hyp


# ═══════════════════════════════════════════════════════════════════════════ #
# 2. Sub-query decomposer                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestQueryDecomposer:
    def _make(self, llm_response="1. sub one\n2. sub two\n3. sub three"):
        from deepsearch.retrieval.query_decomposer import QueryDecomposer
        return QueryDecomposer(llm=_make_llm(llm_response), search=_make_search())

    def test_parse_numbered_list(self):
        from deepsearch.retrieval.query_decomposer import QueryDecomposer
        raw = "1. What is revenue?\n2. What are the costs?\n3. Compare both."
        subs = QueryDecomposer._parse_numbered_list(raw)
        assert subs == ["What is revenue?", "What are the costs?", "Compare both."]

    def test_parse_empty_returns_empty(self):
        from deepsearch.retrieval.query_decomposer import QueryDecomposer
        assert QueryDecomposer._parse_numbered_list("") == []

    def test_short_query_skipped(self):
        d = self._make()
        result = d.retrieve("short", top_k=5)
        assert result.skipped is True
        assert result.sub_queries == ["short"]

    def test_complex_query_decomposed(self):
        d = self._make("1. What is revenue?\n2. What are costs?")
        result = d.retrieve("What is the revenue and how does it compare to total costs?", top_k=5)
        assert result.skipped is False
        assert len(result.sub_queries) == 2
        assert result.chunks  # retrieved from mocked search

    def test_llm_error_falls_back_to_single_search(self):
        from deepsearch.retrieval.query_decomposer import QueryDecomposer
        llm = _make_llm()
        llm.generate.side_effect = RuntimeError("LLM down")
        search = _make_search()
        d = QueryDecomposer(llm=llm, search=search, min_words=3)
        result = d.retrieve("what happened with revenues and costs", top_k=5)
        assert result.skipped is True
        search.search.assert_called_once()

    def test_rrf_merge_deduplicates(self):
        from deepsearch.retrieval.query_decomposer import QueryDecomposer
        d = QueryDecomposer(llm=_make_llm(), search=_make_search())
        lists = [
            [_make_hit("c1", 0.9), _make_hit("c2", 0.7)],
            [_make_hit("c2", 0.8), _make_hit("c3", 0.6)],
        ]
        merged = d._rrf_merge(lists)
        ids = [h["id"] for h in merged]
        # c2 appears in both lists, should only appear once
        assert ids.count("c2") == 1
        assert len(ids) == 3

    def test_max_sub_queries_capped(self):
        from deepsearch.retrieval.query_decomposer import QueryDecomposer
        raw = "\n".join(f"{i}. sub query {i}" for i in range(1, 10))
        subs = QueryDecomposer._parse_numbered_list(raw)
        assert len(subs) == 4   # _MAX_SUB_QUERIES = 4


# ═══════════════════════════════════════════════════════════════════════════ #
# 3. Agentic retriever                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestAgenticRetriever:
    def _make(self, llm_response="search for more info"):
        from deepsearch.retrieval.agentic_retriever import AgenticRetriever
        return AgenticRetriever(llm=_make_llm(llm_response), search=_make_search(), max_steps=3)

    def test_retrieve_returns_result_object(self):
        from deepsearch.retrieval.agentic_retriever import AgentRetrievalResult
        agent = self._make()
        result = agent.retrieve("complex question", top_k=10)
        assert isinstance(result, AgentRetrievalResult)
        assert isinstance(result.chunks, list)
        assert isinstance(result.steps, list)

    def test_done_marker_stops_loop(self):
        from deepsearch.retrieval.agentic_retriever import AgenticRetriever, _DONE_MARKER
        llm = _make_llm(_DONE_MARKER)
        agent = AgenticRetriever(llm=llm, search=_make_search(), max_steps=5)
        result = agent.retrieve("question", top_k=5)
        assert result.terminated_early is True
        assert len(result.steps) == 0  # stopped before first search

    def test_max_steps_respected(self):
        from deepsearch.retrieval.agentic_retriever import AgenticRetriever
        agent = AgenticRetriever(llm=_make_llm("search query"), search=_make_search(), max_steps=2)
        result = agent.retrieve("complex question", top_k=10)
        assert len(result.steps) <= 2

    def test_no_new_chunks_stops_early(self):
        """If step N returns no new chunks, loop breaks."""
        from deepsearch.retrieval.agentic_retriever import AgenticRetriever
        # All searches return the same chunk → after step 2 no new chunks
        agent = AgenticRetriever(
            llm=_make_llm("same search"),
            search=_make_search([_make_hit("only-one")]),
            max_steps=4,
        )
        result = agent.retrieve("question", top_k=10)
        # Should stop well before step 4
        assert len(result.steps) <= 3

    def test_summarise_chunks_empty(self):
        from deepsearch.retrieval.agentic_retriever import AgenticRetriever
        summary = AgenticRetriever._summarise_chunks([])
        assert "nothing" in summary.lower()

    def test_summarise_chunks_truncates(self):
        from deepsearch.retrieval.agentic_retriever import AgenticRetriever
        chunks = [_make_hit(f"c{i}", text="word " * 200) for i in range(10)]
        summary = AgenticRetriever._summarise_chunks(chunks)
        assert "more passages" in summary

    def test_search_error_handled_gracefully(self):
        from deepsearch.retrieval.agentic_retriever import AgenticRetriever
        search = MagicMock()
        search.search.side_effect = RuntimeError("Qdrant down")
        agent = AgenticRetriever(llm=_make_llm("search x"), search=search, max_steps=2)
        result = agent.retrieve("question")   # should not raise
        assert isinstance(result.chunks, list)


# ═══════════════════════════════════════════════════════════════════════════ #
# 4. GraphRAG                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestKnowledgeGraph:
    def test_ingest_chunk_adds_to_graph(self):
        from deepsearch.retrieval.graph_rag import KnowledgeGraph
        g = KnowledgeGraph()
        g.ingest_chunk({"chunk_id": "c1", "text": "Alice works at Acme Corp"})
        assert g.stats.chunks_indexed == 1

    def test_entity_chunk_mapping(self):
        from deepsearch.retrieval.graph_rag import KnowledgeGraph
        g = KnowledgeGraph()
        g.ingest_chunk({"chunk_id": "c1", "text": "Apple revenue grew in 2023"})
        # Should find at least one entity and link it to c1
        found = any("c1" in ids for ids in g._entity_chunks.values())
        assert found

    def test_cooccurrence_bidirectional(self):
        from deepsearch.retrieval.graph_rag import KnowledgeGraph
        g = KnowledgeGraph()
        g.ingest_chunk({"chunk_id": "c1", "text": "Alice and Bob met at Microsoft headquarters"})
        # At least some co-occurrences should be bidirectional
        for a, neighbours in g._cooccurrences.items():
            for b in neighbours:
                assert a in g._cooccurrences.get(b, set())

    def test_expand_query_entities_returns_list(self):
        from deepsearch.retrieval.graph_rag import KnowledgeGraph
        g = KnowledgeGraph()
        g.ingest_chunk({"chunk_id": "c1", "text": "Apple acquired Microsoft in 2025"})
        expanded = g.expand_query_entities("Apple acquisition", hops=1)
        assert isinstance(expanded, list)

    def test_clear_resets_all(self):
        from deepsearch.retrieval.graph_rag import KnowledgeGraph
        g = KnowledgeGraph()
        g.ingest_chunk({"chunk_id": "c1", "text": "Some text here"})
        g.clear()
        assert g.stats.chunks_indexed == 0
        assert g.stats.nodes == 0

    def test_empty_text_ignored(self):
        from deepsearch.retrieval.graph_rag import KnowledgeGraph
        g = KnowledgeGraph()
        g.ingest_chunk({"chunk_id": "c1", "text": ""})
        assert g.stats.chunks_indexed == 0

    def test_build_from_hits(self):
        from deepsearch.retrieval.graph_rag import KnowledgeGraph
        g = KnowledgeGraph()
        hits = [_make_hit("c1", text="Tesla launched new model in California")]
        g.build_from_chunks(hits)
        assert g.stats.chunks_indexed == 1


class TestGraphRAGRetriever:
    def test_retrieve_returns_merged_hits(self):
        from deepsearch.retrieval.graph_rag import GraphRAGRetriever
        search = _make_search([_make_hit("c1", text="Apple revenue")])
        gr = GraphRAGRetriever(search=search)
        hits = gr.retrieve("Apple revenue question", top_k=5)
        assert isinstance(hits, list)

    def test_graph_updated_after_retrieve(self):
        from deepsearch.retrieval.graph_rag import GraphRAGRetriever, KnowledgeGraph
        g = KnowledgeGraph()
        search = _make_search([_make_hit("c1", text="Tesla and Elon Musk")])
        gr = GraphRAGRetriever(search=search, graph=g)
        gr.retrieve("Tesla news", top_k=5)
        assert g.stats.chunks_indexed >= 1


# ═══════════════════════════════════════════════════════════════════════════ #
# 5. PartitionedVectorStore routing                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestPartitionedStoreRouting:
    """Test the modality routing logic without a real Qdrant instance."""

    def test_type_to_modality_pdf_is_text(self):
        from deepsearch.storage.partitioned_store import _modality_for
        assert _modality_for({"file_type": "pdf"}) == "text"

    def test_type_to_modality_jpg_is_image(self):
        from deepsearch.storage.partitioned_store import _modality_for
        assert _modality_for({"file_type": "jpg"}) == "image"

    def test_type_to_modality_mp3_is_audio(self):
        from deepsearch.storage.partitioned_store import _modality_for
        assert _modality_for({"file_type": "mp3"}) == "audio"

    def test_type_to_modality_mp4_is_video(self):
        from deepsearch.storage.partitioned_store import _modality_for
        assert _modality_for({"file_type": "mp4"}) == "video"

    def test_unknown_type_defaults_to_text(self):
        from deepsearch.storage.partitioned_store import _modality_for
        assert _modality_for({"file_type": "xyz"}) == "text"

    def test_collection_name(self):
        from deepsearch.storage.partitioned_store import _collection_name
        assert _collection_name("text") == "documents_text"
        assert _collection_name("image") == "documents_image"

    def test_rrf_merge_deduplicates(self):
        from deepsearch.storage.partitioned_store import PartitionedVectorStore
        lists = [
            [_make_hit("c1", 0.9), _make_hit("c2", 0.7)],
            [_make_hit("c2", 0.8), _make_hit("c3", 0.6)],
        ]
        merged = PartitionedVectorStore._rrf_merge(lists, top_k=10)
        ids = [h["id"] for h in merged]
        assert ids.count("c2") == 1
        assert len(ids) == 3

    def test_rrf_merge_top_k_cap(self):
        from deepsearch.storage.partitioned_store import PartitionedVectorStore
        lists = [[_make_hit(f"c{i}") for i in range(20)]]
        merged = PartitionedVectorStore._rrf_merge(lists, top_k=5)
        assert len(merged) == 5


# ═══════════════════════════════════════════════════════════════════════════ #
# 6. SpeculativeBackend                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestSpeculativeBackend:
    def _make(self, main_answer="main answer", draft_answer="draft token"):
        from deepsearch.backends.speculative_backend import SpeculativeBackend
        from deepsearch.backends.base import BackendInfo
        main = MagicMock()
        main.generate.return_value = main_answer
        main.stream.return_value = iter([main_answer])
        main.build_prompt.side_effect = lambda s, u, ctx="": f"{s}\n{u}"
        main.info = BackendInfo("llamacpp", "main-model", "cpu", loaded=True)
        draft = MagicMock()
        draft.generate.return_value = draft_answer
        draft.info = BackendInfo("llamacpp", "draft-model", "cpu", loaded=True)
        return SpeculativeBackend(main_backend=main, draft_backend=draft, gamma=3)

    def test_load_raises_not_implemented(self):
        from deepsearch.backends.speculative_backend import SpeculativeBackend
        spec = SpeculativeBackend(main_backend=MagicMock(), draft_backend=MagicMock())
        with pytest.raises(NotImplementedError):
            spec.load("some/path")

    def test_info_reflects_main_model(self):
        spec = self._make()
        assert "main-model" in spec.info.model_id

    def test_generate_returns_string(self):
        from deepsearch.backends.base import GenerationConfig
        spec = self._make()
        result = spec.generate("prompt text", GenerationConfig(max_tokens=50))
        assert isinstance(result, str)

    def test_stream_delegates_to_main(self):
        from deepsearch.backends.base import GenerationConfig
        spec = self._make(main_answer="streamed token")
        tokens = list(spec.stream("prompt", GenerationConfig()))
        assert tokens == ["streamed token"]

    def test_stats_initialized(self):
        spec = self._make()
        assert spec.stats.total_tokens == 0
        assert spec.stats.acceptance_rate == 0.0

    def test_unload_calls_both_backends(self):
        spec = self._make()
        spec.unload()
        spec._main.unload.assert_called_once()
        spec._draft.unload.assert_called_once()

    def test_stats_reset(self):
        from deepsearch.backends.base import GenerationConfig
        spec = self._make()
        spec.generate("prompt", GenerationConfig(max_tokens=20))
        spec.reset_stats()
        assert spec.stats.total_tokens == 0


# ═══════════════════════════════════════════════════════════════════════════ #
# 7. LanguageManager                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestLanguageManager:
    def test_detect_short_text_returns_default(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager(default_language="en")
        assert lm.detect("hi") == "en"

    def test_detect_english_text(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager()
        # Force langdetect to return "en"
        with patch("deepsearch.core.language_manager.LanguageManager._try_langdetect", return_value="en"):
            lang = lm.detect("This is a normal English sentence about technology.")
        assert lang == "en"

    def test_detect_french_text(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager()
        with patch("deepsearch.core.language_manager.LanguageManager._try_langdetect", return_value="fr"):
            lang = lm.detect("Quelle est la capitale de la France aujourd'hui?")
        assert lang == "fr"

    def test_cjk_script_hint(self):
        from deepsearch.core.language_manager import _script_hint
        assert _script_hint("这是一段中文文字内容") == "zh-cn"

    def test_arabic_script_hint(self):
        from deepsearch.core.language_manager import _script_hint
        assert _script_hint("مرحبا بكم في هذا النص العربي") == "ar"

    def test_embedding_model_english(self):
        from deepsearch.core.language_manager import LanguageManager, _ENGLISH_EMBEDDING_MODEL
        lm = LanguageManager()
        assert lm.embedding_model_for("en") == _ENGLISH_EMBEDDING_MODEL

    def test_embedding_model_non_english(self):
        from deepsearch.core.language_manager import LanguageManager, _MULTILINGUAL_EMBEDDING_MODEL
        lm = LanguageManager()
        assert lm.embedding_model_for("fr") == _MULTILINGUAL_EMBEDDING_MODEL
        assert lm.embedding_model_for("zh-cn") == _MULTILINGUAL_EMBEDDING_MODEL

    def test_stopwords_for_english(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager()
        sw = lm.stopwords_for("en")
        assert "the" in sw

    def test_stopwords_for_french(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager()
        sw = lm.stopwords_for("fr")
        assert "le" in sw

    def test_translate_en_to_en_no_op(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager()
        llm = _make_llm("should not be called")
        result = lm.translate_to_english("hello world", "en", llm)
        assert result == "hello world"
        llm.generate.assert_not_called()

    def test_translate_calls_llm_for_non_english(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager()
        llm = _make_llm("translated text")
        result = lm.translate_to_english("bonjour monde", "fr", llm)
        assert result == "translated text"
        llm.generate.assert_called_once()

    def test_translate_falls_back_on_llm_error(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager()
        llm = _make_llm()
        llm.generate.side_effect = RuntimeError("LLM unavailable")
        result = lm.translate_to_english("bonjour", "fr", llm)
        assert result == "bonjour"  # original returned unchanged

    def test_langdetect_not_installed_falls_back(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager(default_language="en")
        with patch.dict(sys.modules, {"langdetect": None, "langid": None}):
            lang = lm.detect("This is a test sentence for language detection.")
        # Should not raise; returns default or script hint
        assert isinstance(lang, str)

    def test_detect_batch(self):
        from deepsearch.core.language_manager import LanguageManager
        lm = LanguageManager()
        with patch.object(lm, "detect", side_effect=["en", "fr", "de"]):
            langs = lm.detect_batch(["text1", "text2", "text3"])
        assert langs == ["en", "fr", "de"]
