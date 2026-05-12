"""Audio parser — speech-to-text via faster-whisper (CTranslate2).

Transcription pipeline:
  1. Load audio with faster-whisper (handles MP3/WAV/M4A/OGG/FLAC/etc.)
  2. Transcribe with word-level timestamps
  3. Chunk output into ~60-second segments with 15-second overlap
  4. Each chunk stored as a page with [MM:SS] timestamp prefix

Lazy imports: faster-whisper is only imported inside parse() so the
app starts normally when the library is not installed.

Supported: .mp3 .wav .m4a .ogg .flac .opus .aac .wma .webm (audio)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from .base import BaseParser, ParsedDocument

log = logging.getLogger(__name__)

_SUPPORTED = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".aac", ".wma"]

# Segment grouping parameters
_SEGMENT_DURATION = 60.0   # seconds per chunk page
_OVERLAP_DURATION  = 15.0  # seconds of overlap between adjacent chunks


def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as [MM:SS]."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"[{m:02d}:{s:02d}]"


def _group_segments(
    segments: list[dict],
    chunk_duration: float = _SEGMENT_DURATION,
    overlap: float = _OVERLAP_DURATION,
) -> list[str]:
    """Group whisper segments into time-windowed text chunks.

    Args:
        segments:       List of {"start": float, "end": float, "text": str}
        chunk_duration: Target seconds per output chunk.
        overlap:        How many seconds from the end of the previous chunk
                        to repeat at the start of the next (context window).

    Returns:
        List of chunk strings with timestamp prefix.
    """
    if not segments:
        return []

    chunks: list[str] = []
    chunk_start = segments[0]["start"]
    chunk_texts: list[str] = []

    i = 0
    while i < len(segments):
        seg = segments[i]
        chunk_texts.append(seg["text"].strip())

        # Close chunk when we've exceeded the window
        if seg["end"] - chunk_start >= chunk_duration:
            ts = _fmt_timestamp(chunk_start)
            chunks.append(f"{ts} {' '.join(chunk_texts)}")

            # Find overlap start point
            overlap_start = seg["end"] - overlap
            chunk_start = overlap_start
            chunk_texts = []

            # Rewind i to include overlapping segments in the next chunk
            while i > 0 and segments[i]["start"] > overlap_start:
                i -= 1

        i += 1

    # Flush remaining text
    if chunk_texts:
        ts = _fmt_timestamp(chunk_start)
        chunks.append(f"{ts} {' '.join(chunk_texts)}")

    return chunks


class AudioParser(BaseParser):
    """Transcribes audio files to text using faster-whisper.

    Args:
        model_size:    Whisper model variant ("tiny", "base", "small",
                       "medium", "large-v3"). "medium" gives best
                       speed/accuracy on CPU.
        device:        "cpu" or "cuda". Auto-detected if not set.
        compute_type:  "int8" (CPU default), "float16" (GPU).
        language:      Force language code (e.g. "en") or None for auto-detect.
        beam_size:     Beam search width. Higher = more accurate, slower.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
        beam_size: int = 5,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size

    @property
    def supported_extensions(self) -> list[str]:
        return _SUPPORTED

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            ) from exc

        log.info("Transcribing audio: %s", path.name)

        # Resolve device
        device = self._device
        if device == "auto":
            try:
                import torch  # type: ignore[import]
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = self._compute_type
        if compute_type == "default":
            compute_type = "int8" if device == "cpu" else "float16"

        model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
        )

        whisper_segments, info = model.transcribe(
            str(path),
            beam_size=self._beam_size,
            language=self._language,
            word_timestamps=True,
        )

        # Materialise the lazy generator once
        raw_segments: list[dict] = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in whisper_segments
        ]

        duration = info.duration if hasattr(info, "duration") else 0.0
        detected_lang = info.language if hasattr(info, "language") else "unknown"

        log.debug(
            "Audio transcribed: %s — %.1fs, lang=%s, %d segments",
            path.name, duration, detected_lang, len(raw_segments),
        )

        # Build time-windowed chunk pages
        chunk_pages = _group_segments(raw_segments)
        full_text = "\n\n".join(chunk_pages)

        metadata = {
            "file_name": path.name,
            "format": "audio",
            "modality": "audio",
            "duration_seconds": round(duration, 2),
            "language": detected_lang,
            "whisper_model": self._model_size,
            "segment_count": len(raw_segments),
        }

        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=full_text,
            pages=chunk_pages,
            metadata=metadata,
        )
