"""Unit tests for Phase 3 remaining components:
ImageParser, AudioParser, VideoParser, ConfidenceScorer, CloudPipeline.

All heavy optional dependencies (paddleocr, faster-whisper, cv2, presidio)
are mocked via sys.modules or unittest.mock — no real libraries needed.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ===========================================================================
# ConfidenceScorer
# ===========================================================================

class TestConfidenceScorer:
    def _make(self, **kwargs):
        from deepsearch.generation.confidence_scorer import ConfidenceScorer
        return ConfidenceScorer(**kwargs)

    def test_high_confidence_clean_answer(self):
        scorer = self._make()
        conf = scorer.score("The capital of France is Paris.", retrieval_scores=[0.9, 0.85])
        assert conf > 0.7

    def test_low_confidence_refusal_phrase(self):
        scorer = self._make()
        conf = scorer.score("I couldn't find that in your documents.")
        assert conf < 0.5

    def test_dont_know_phrase(self):
        scorer = self._make()
        conf = scorer.score("I don't know the answer to that.")
        assert conf < 0.6

    def test_empty_answer_scores_zero(self):
        scorer = self._make()
        conf = scorer.score("")
        assert conf == 0.0 or conf < 0.2

    def test_short_answer_penalised(self):
        scorer = self._make()
        long_conf  = scorer.score("A " * 50)   # 50 words
        short_conf = scorer.score("Maybe.")     # 1 word
        assert long_conf > short_conf

    def test_retrieval_signal_high_scores_boost(self):
        scorer = self._make()
        high = scorer.score("The answer is X.", retrieval_scores=[0.95, 0.90, 0.88])
        low  = scorer.score("The answer is X.", retrieval_scores=[0.1, 0.05, 0.02])
        assert high > low

    def test_retrieval_signal_missing_uses_neutral(self):
        scorer = self._make()
        conf = scorer.score("The answer is X.", retrieval_scores=None)
        # Should produce a moderate score — not 0 or 1
        assert 0.2 < conf < 0.95

    def test_score_detailed_returns_breakdown(self):
        from deepsearch.generation.confidence_scorer import ConfidenceBreakdown
        scorer = self._make()
        bd = scorer.score_detailed("The answer is X.", retrieval_scores=[0.8])
        assert isinstance(bd, ConfidenceBreakdown)
        assert 0.0 <= bd.lexical   <= 1.0
        assert 0.0 <= bd.length    <= 1.0
        assert 0.0 <= bd.retrieval <= 1.0
        assert 0.0 <= bd.final     <= 1.0

    def test_weight_normalisation(self):
        # Weights sum > 1 should still produce a valid [0,1] score
        scorer = self._make(weight_lexical=5, weight_length=3, weight_retrieval=2)
        conf = scorer.score("Good answer here.", retrieval_scores=[0.8])
        assert 0.0 <= conf <= 1.0

    def test_invalid_zero_weights_raises(self):
        import pytest
        with pytest.raises(ValueError):
            from deepsearch.generation.confidence_scorer import ConfidenceScorer
            ConfidenceScorer(weight_lexical=0, weight_length=0, weight_retrieval=0)

    def test_multiple_hedges_stack(self):
        scorer = self._make()
        one_hedge = scorer.score("I'm not sure about this.")
        two_hedge = scorer.score("I'm not sure — unclear and uncertain.")
        assert two_hedge <= one_hedge   # more hedging → lower or equal score


# ===========================================================================
# ImageParser
# ===========================================================================

class TestImageParser:
    def test_supported_extensions(self):
        from deepsearch.ingestion.parsers.image_parser import ImageParser
        p = ImageParser()
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            assert ext in p.supported_extensions

    def test_parse_with_mock_ocr(self, tmp_path):
        from deepsearch.ingestion.parsers.image_parser import ImageParser

        img = tmp_path / "slide.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_ocr_instance = MagicMock()
        mock_ocr_instance.ocr.return_value = [
            [
                ([[0, 0], [100, 0], [100, 20], [0, 20]], ("Hello World", 0.95)),
                ([[0, 25], [100, 25], [100, 45], [0, 45]], ("Second line", 0.88)),
            ]
        ]
        mock_paddle = MagicMock()
        mock_paddle.PaddleOCR.return_value = mock_ocr_instance

        with patch.dict(sys.modules, {"paddleocr": mock_paddle}):
            parser = ImageParser(ocr_enabled=True)
            doc = parser.parse(img, "img-1")

        assert "Hello World" in doc.text
        assert "Second line" in doc.text
        assert doc.metadata["modality"] == "image"
        assert doc.metadata["format"] == "image"

    def test_parse_ocr_disabled(self, tmp_path):
        from deepsearch.ingestion.parsers.image_parser import ImageParser

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

        parser = ImageParser(ocr_enabled=False)
        doc = parser.parse(img, "img-2")
        # No OCR — should return empty or caption-only text
        assert doc.metadata["modality"] == "image"

    def test_parse_ocr_import_error_graceful(self, tmp_path):
        from deepsearch.ingestion.parsers.image_parser import ImageParser

        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        # Simulate paddleocr not installed
        with patch.dict(sys.modules, {"paddleocr": None}):
            parser = ImageParser(ocr_enabled=True)
            # Should not raise — warnings logged, empty text returned
            doc = parser.parse(img, "img-err")
            assert isinstance(doc.text, str)

    def test_parse_low_confidence_boxes_excluded(self, tmp_path):
        from deepsearch.ingestion.parsers.image_parser import ImageParser

        img = tmp_path / "noisy.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)

        mock_ocr_instance = MagicMock()
        # One high-conf box and one low-conf box
        mock_ocr_instance.ocr.return_value = [
            [
                ([[0, 0], [100, 0], [100, 20], [0, 20]], ("Visible text", 0.92)),
                ([[0, 25], [100, 25], [100, 45], [0, 45]], ("Noise", 0.20)),  # < threshold
            ]
        ]
        mock_paddle = MagicMock()
        mock_paddle.PaddleOCR.return_value = mock_ocr_instance

        with patch.dict(sys.modules, {"paddleocr": mock_paddle}):
            parser = ImageParser(ocr_enabled=True)
            doc = parser.parse(img, "img-3")

        assert "Visible text" in doc.text
        assert "Noise" not in doc.text


# ===========================================================================
# AudioParser
# ===========================================================================

class TestAudioParser:
    def test_supported_extensions(self):
        from deepsearch.ingestion.parsers.audio_parser import AudioParser
        p = AudioParser()
        for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
            assert ext in p.supported_extensions

    def _make_mock_whisper(self, segments):
        """Build a mock WhisperModel that returns given segments."""
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.duration = sum(s["end"] for s in segments) if segments else 0.0
        mock_info.language = "en"

        mock_segs = []
        for s in segments:
            seg = MagicMock()
            seg.start = s["start"]
            seg.end   = s["end"]
            seg.text  = s["text"]
            mock_segs.append(seg)

        mock_model.transcribe.return_value = (iter(mock_segs), mock_info)
        return mock_model

    def test_parse_returns_document(self, tmp_path):
        from deepsearch.ingestion.parsers.audio_parser import AudioParser

        audio = tmp_path / "lecture.mp3"
        audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)

        segments = [
            {"start": 0.0, "end": 10.0, "text": " Hello everyone."},
            {"start": 10.0, "end": 20.0, "text": " Welcome to the lecture."},
        ]
        mock_model = self._make_mock_whisper(segments)
        mock_fw = MagicMock()
        mock_fw.WhisperModel.return_value = mock_model

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            parser = AudioParser(model_size="tiny")
            doc = parser.parse(audio, "aud-1")

        assert "Hello everyone" in doc.text
        assert "Welcome to the lecture" in doc.text
        assert doc.metadata["modality"] == "audio"
        assert doc.metadata["language"] == "en"

    def test_parse_chunks_long_audio(self, tmp_path):
        from deepsearch.ingestion.parsers.audio_parser import AudioParser

        audio = tmp_path / "long.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        # 3 minutes of 10-second segments → should produce multiple pages
        segments = [
            {"start": float(i * 10), "end": float((i + 1) * 10), "text": f" Segment {i}."}
            for i in range(18)
        ]
        mock_model = self._make_mock_whisper(segments)
        mock_fw = MagicMock()
        mock_fw.WhisperModel.return_value = mock_model

        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            parser = AudioParser(model_size="tiny")
            doc = parser.parse(audio, "aud-long")

        assert len(doc.pages) > 1  # Multiple 60s chunks

    def test_parse_import_error_raises(self, tmp_path):
        import pytest
        from deepsearch.ingestion.parsers.audio_parser import AudioParser

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\xff\xfb" + b"\x00" * 50)

        with patch.dict(sys.modules, {"faster_whisper": None}):
            parser = AudioParser()
            with pytest.raises(ImportError):
                parser.parse(audio, "aud-err")

    def test_timestamp_format(self):
        from deepsearch.ingestion.parsers.audio_parser import _fmt_timestamp
        assert _fmt_timestamp(0)   == "[00:00]"
        assert _fmt_timestamp(65)  == "[01:05]"
        assert _fmt_timestamp(125) == "[02:05]"

    def test_group_segments_empty(self):
        from deepsearch.ingestion.parsers.audio_parser import _group_segments
        assert _group_segments([]) == []

    def test_group_segments_single_chunk(self):
        from deepsearch.ingestion.parsers.audio_parser import _group_segments
        segs = [{"start": 0.0, "end": 5.0, "text": " Hello."}]
        chunks = _group_segments(segs)
        assert len(chunks) == 1
        assert "Hello" in chunks[0]


# ===========================================================================
# VideoParser
# ===========================================================================

class TestVideoParser:
    def test_supported_extensions(self):
        from deepsearch.ingestion.parsers.video_parser import VideoParser
        p = VideoParser()
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            assert ext in p.supported_extensions

    def test_parse_no_keyframes_no_audio_returns_fallback(self, tmp_path):
        from deepsearch.ingestion.parsers.video_parser import VideoParser
        from deepsearch.ingestion.parsers.image_parser import ImageParser
        from deepsearch.ingestion.parsers.audio_parser import AudioParser

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 100)

        img_parser = MagicMock(spec=ImageParser)
        aud_parser = MagicMock(spec=AudioParser)

        with patch("deepsearch.ingestion.parsers.video_parser._extract_keyframes", return_value=[]):
            with patch("deepsearch.ingestion.parsers.video_parser._extract_audio_track", return_value=None):
                parser = VideoParser(image_parser=img_parser, audio_parser=aud_parser)
                doc = parser.parse(video, "vid-1")

        assert doc.metadata["modality"] == "video"
        assert doc.metadata["keyframe_count"] == 0
        # Falls back to "[No extractable content]"
        assert "[No extractable content]" in doc.text

    def test_parse_with_keyframes(self, tmp_path):
        from deepsearch.ingestion.parsers.video_parser import VideoParser
        from deepsearch.ingestion.parsers.base import ParsedDocument
        from deepsearch.ingestion.parsers.image_parser import ImageParser
        from deepsearch.ingestion.parsers.audio_parser import AudioParser

        video = tmp_path / "talk.mp4"
        video.write_bytes(b"\x00" * 100)

        # Create a fake keyframe image file
        kf_img = tmp_path / "frame_0000.jpg"
        kf_img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

        fake_keyframes = [{"timestamp": 0.0, "path": kf_img}]

        img_parser = MagicMock(spec=ImageParser)
        img_parser.parse.return_value = ParsedDocument(
            file_id="kf", source_path=str(kf_img),
            text="Slide title: Introduction", pages=["Slide title: Introduction"],
            metadata={},
        )
        aud_parser = MagicMock(spec=AudioParser)

        with patch("deepsearch.ingestion.parsers.video_parser._extract_keyframes", return_value=fake_keyframes):
            with patch("deepsearch.ingestion.parsers.video_parser._extract_audio_track", return_value=None):
                parser = VideoParser(image_parser=img_parser, audio_parser=aud_parser)
                doc = parser.parse(video, "vid-2")

        assert "Slide title" in doc.text
        assert doc.metadata["keyframe_count"] == 1

    def test_fmt_hms(self):
        from deepsearch.ingestion.parsers.video_parser import _fmt_hms
        assert _fmt_hms(0)    == "[00:00:00]"
        assert _fmt_hms(65)   == "[00:01:05]"
        assert _fmt_hms(3661) == "[01:01:01]"

    def test_keyframe_import_error_graceful(self, tmp_path):
        from deepsearch.ingestion.parsers.video_parser import VideoParser
        from deepsearch.ingestion.parsers.image_parser import ImageParser
        from deepsearch.ingestion.parsers.audio_parser import AudioParser

        video = tmp_path / "clip.mkv"
        video.write_bytes(b"\x00" * 100)

        img_parser = MagicMock(spec=ImageParser)
        aud_parser = MagicMock(spec=AudioParser)

        def raise_import(*a, **kw):
            raise ImportError("cv2 not installed")

        with patch("deepsearch.ingestion.parsers.video_parser._extract_keyframes", side_effect=raise_import):
            with patch("deepsearch.ingestion.parsers.video_parser._extract_audio_track", return_value=None):
                parser = VideoParser(image_parser=img_parser, audio_parser=aud_parser)
                # Should not propagate ImportError
                doc = parser.parse(video, "vid-3")

        assert doc.metadata["format"] == "video"


# ===========================================================================
# CloudPipeline
# ===========================================================================

class TestCloudPipeline:
    def _make_pipeline(self, scrubber=None):
        from deepsearch.generation.cloud_pipeline import CloudPipeline
        return CloudPipeline(scrubber=scrubber)

    def test_generate_calls_backend(self, monkeypatch):
        from deepsearch.generation.cloud_pipeline import CloudPipeline, PIIScrubber

        scrubber = MagicMock(spec=PIIScrubber)
        scrubber.scrub.return_value = ("clean context", [])

        pipeline = CloudPipeline(scrubber=scrubber)

        with patch.object(pipeline, "_call_backend", return_value="Cloud answer") as mock_call:
            result = pipeline.generate("What is AI?", "AI stands for...")

        assert result.answer == "Cloud answer"
        assert not result.blocked
        mock_call.assert_called_once()

    def test_generate_blocked_on_pii_entity(self):
        from deepsearch.generation.cloud_pipeline import CloudPipeline, PIIScrubber

        scrubber = MagicMock(spec=PIIScrubber)
        # Simulates SSN detected in context
        scrubber.scrub.side_effect = [
            ("context with [REDACTED]", ["US_SSN"]),
            ("question", []),
        ]

        pipeline = CloudPipeline(scrubber=scrubber)
        result = pipeline.generate("What is his SSN?", "SSN: 123-45-6789")

        assert result.blocked
        assert "US_SSN" in result.block_reason
        assert result.answer == ""

    def test_generate_blocked_on_high_scrub_ratio(self):
        from deepsearch.generation.cloud_pipeline import CloudPipeline, PIIScrubber

        scrubber = MagicMock(spec=PIIScrubber)
        # Simulate massive scrubbing: 10 tokens → 1 token after scrub
        scrubber.scrub.side_effect = [
            ("x", ["EMAIL_ADDRESS"] * 10),   # context: many entities, short output
            ("q", []),
        ]

        pipeline = CloudPipeline(scrubber=scrubber)
        # context has 11 words, scrubbed has 1 → ratio ~ 0.91 > 0.4
        result = pipeline.generate("q", "one two three four five six seven eight nine ten eleven")

        assert result.blocked

    def test_audit_log_populated(self, monkeypatch):
        from deepsearch.generation.cloud_pipeline import CloudPipeline, PIIScrubber

        scrubber = MagicMock(spec=PIIScrubber)
        scrubber.scrub.return_value = ("clean", [])

        pipeline = CloudPipeline(scrubber=scrubber)
        with patch.object(pipeline, "_call_backend", return_value="OK"):
            pipeline.generate("question", "context")

        assert len(pipeline.audit_log) == 1
        record = pipeline.audit_log[0]
        assert len(record.question_hash) == 16  # sha256[:16]

    def test_scrubber_no_presidio_passthrough(self):
        from deepsearch.generation.cloud_pipeline import PIIScrubber

        # Simulate presidio not installed
        with patch.dict(sys.modules, {"presidio_analyzer": None, "presidio_anonymizer": None}):
            scrubber = PIIScrubber()

        assert not scrubber.available
        text = "My SSN is 123-45-6789"
        scrubbed, entities = scrubber.scrub(text)
        # No-op: text returned unchanged, no entities
        assert scrubbed == text
        assert entities == []

    def test_stream_blocked_raises(self):
        import pytest
        from deepsearch.generation.cloud_pipeline import CloudPipeline, PIIScrubber

        scrubber = MagicMock(spec=PIIScrubber)
        scrubber.scrub.side_effect = [
            ("", ["CREDIT_CARD"]),
            ("q", []),
        ]

        pipeline = CloudPipeline(scrubber=scrubber)
        with pytest.raises(PermissionError, match="CREDIT_CARD"):
            list(pipeline.stream("q", "context with CC"))


# ===========================================================================
# LLMPipeline — confidence scorer integration
# ===========================================================================

class TestPipelineConfidenceIntegration:
    def _make_pipeline(self):
        from deepsearch.generation.llm_pipeline import LLMPipeline
        from deepsearch.storage.cache import CacheBundle

        search = MagicMock()
        search.search_dense_only.return_value = [
            {"id": "c1", "score": 0.85, "text": "Paris is the capital."},
        ]
        ctx_builder = MagicMock()
        ctx_builder.build.return_value = ("Paris is the capital of France.", [])

        mm = MagicMock()
        llm = MagicMock()
        llm.build_prompt.return_value = "prompt"
        llm.generate.return_value = "Paris is the capital of France."
        mm.llm_main = llm
        mm.llm_small = None

        cfg = MagicMock()
        cfg.cloud.confidence_threshold = 0.3
        cfg.cloud.enabled = False
        cfg.cloud.max_tokens = 512

        pipeline = LLMPipeline(
            hybrid_search=search,
            context_builder=ctx_builder,
            model_manager=mm,
            cache=CacheBundle(),
            settings=cfg,
        )
        return pipeline

    def test_result_confidence_in_range(self):
        pipeline = self._make_pipeline()
        result = pipeline.query("What is the capital of France?", mode="fast")
        assert 0.0 <= result.confidence <= 1.0

    def test_refusal_answer_gets_low_confidence(self):
        from deepsearch.generation.llm_pipeline import LLMPipeline
        from deepsearch.storage.cache import CacheBundle

        search = MagicMock()
        search.search_dense_only.return_value = [{"id": "c1", "score": 0.2}]
        ctx_builder = MagicMock()
        ctx_builder.build.return_value = ("Some context.", [])

        mm = MagicMock()
        llm = MagicMock()
        llm.build_prompt.return_value = "prompt"
        llm.generate.return_value = "I couldn't find that in your documents."
        mm.llm_main = llm
        mm.llm_small = None

        cfg = MagicMock()
        cfg.cloud.confidence_threshold = 0.3
        cfg.cloud.enabled = False
        cfg.cloud.max_tokens = 512

        pipeline = LLMPipeline(
            hybrid_search=search,
            context_builder=ctx_builder,
            model_manager=mm,
            cache=CacheBundle(),
            settings=cfg,
        )
        result = pipeline.query("Unknown question?", mode="fast")
        assert result.confidence < 0.5
