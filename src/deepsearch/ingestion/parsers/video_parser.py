"""Video parser — keyframe extraction + audio transcription.

Processing pipeline:
  1. OpenCV  → scene-change-based keyframe extraction (one image per scene)
  2. ImageParser → OCR + optional LLaVA caption for each keyframe
  3. AudioParser → faster-whisper transcription of the audio track
  4. Per-scene chunk: "[HH:MM:SS] Visual: <caption/OCR>  Audio: <transcript>"

All heavy imports (cv2, faster-whisper, paddleocr) are lazy, so the app
runs normally when these optional dependencies are not installed.

Supported: .mp4 .avi .mov .mkv .webm .flv .wmv .m4v
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .audio_parser import AudioParser
from .base import BaseParser, ParsedDocument
from .image_parser import ImageParser

log = logging.getLogger(__name__)

_SUPPORTED = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]

# Scene-change detection threshold (0–100).  Higher = fewer keyframes.
_SCENE_THRESHOLD = 30.0
# Maximum keyframes to extract (guards against very long/noisy videos)
_MAX_KEYFRAMES = 200
# Minimum gap between keyframes (seconds) to avoid near-duplicate frames
_MIN_KEYFRAME_GAP = 2.0


def _extract_keyframes(
    video_path: Path,
    output_dir: Path,
    threshold: float = _SCENE_THRESHOLD,
    max_frames: int = _MAX_KEYFRAMES,
    min_gap: float = _MIN_KEYFRAME_GAP,
) -> list[dict]:
    """Extract scene-change keyframes using OpenCV.

    Returns:
        List of {"timestamp": float, "path": Path} dicts.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless not installed. "
            "Run: pip install opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    keyframes: list[dict] = []
    prev_gray = None
    frame_idx = 0
    last_saved_ts = -_MIN_KEYFRAME_GAP  # allow first frame

    try:
        while len(keyframes) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_idx / fps
            frame_idx += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                # Always save the first frame
                is_scene_change = True
            else:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = float(diff.mean())
                is_scene_change = mean_diff > threshold

            prev_gray = gray

            if is_scene_change and (timestamp - last_saved_ts) >= min_gap:
                img_name = f"frame_{len(keyframes):04d}_{int(timestamp):06d}s.jpg"
                img_path = output_dir / img_name
                cv2.imwrite(str(img_path), frame)
                keyframes.append({"timestamp": timestamp, "path": img_path})
                last_saved_ts = timestamp

    finally:
        cap.release()

    log.debug("Extracted %d keyframes from %s", len(keyframes), video_path.name)
    return keyframes


def _extract_audio_track(video_path: Path, output_dir: Path) -> Path | None:
    """Demux audio to a temporary WAV file using ffmpeg (subprocess)."""
    import subprocess  # noqa: PLC0415 — stdlib, no lazy import needed
    import shutil

    if shutil.which("ffmpeg") is None:
        log.warning("ffmpeg not found — audio track skipped for %s", video_path.name)
        return None

    audio_path = output_dir / "audio_track.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # PCM WAV
        "-ar", "16000",           # 16 kHz (Whisper native)
        "-ac", "1",               # mono
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        log.warning(
            "ffmpeg audio extract failed for %s: %s",
            video_path.name,
            result.stderr.decode(errors="replace")[:200],
        )
        return None

    if not audio_path.exists() or audio_path.stat().st_size < 1000:
        return None

    return audio_path


def _fmt_hms(seconds: float) -> str:
    """Format seconds as [HH:MM:SS]."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"[{h:02d}:{m:02d}:{s:02d}]"


class VideoParser(BaseParser):
    """Parses video files into multi-modal text chunks.

    Args:
        image_parser:  ImageParser instance (OCR + optional caption).
                       Defaults to OCR-only (no LLaVA).
        audio_parser:  AudioParser instance.  Defaults to medium Whisper.
        scene_threshold:  OpenCV mean-pixel-diff threshold for scene changes.
        max_keyframes: Hard cap on keyframes extracted per video.
    """

    def __init__(
        self,
        image_parser: ImageParser | None = None,
        audio_parser: AudioParser | None = None,
        scene_threshold: float = _SCENE_THRESHOLD,
        max_keyframes: int = _MAX_KEYFRAMES,
    ) -> None:
        self._img_parser = image_parser or ImageParser(ocr_enabled=True)
        self._aud_parser = audio_parser or AudioParser()
        self._threshold = scene_threshold
        self._max_keyframes = max_keyframes

    @property
    def supported_extensions(self) -> list[str]:
        return _SUPPORTED

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        log.info("Parsing video: %s", path.name)

        with tempfile.TemporaryDirectory(prefix="deepsearch_video_") as tmp_str:
            tmp = Path(tmp_str)

            # ── Keyframes ────────────────────────────────────────────────────
            kf_dir = tmp / "keyframes"
            kf_dir.mkdir()

            keyframes: list[dict] = []
            try:
                keyframes = _extract_keyframes(
                    path, kf_dir,
                    threshold=self._threshold,
                    max_frames=self._max_keyframes,
                )
            except ImportError as exc:
                log.warning("Keyframe extraction disabled: %s", exc)
            except Exception as exc:
                log.warning("Keyframe extraction failed for %s: %s", path.name, exc)

            # OCR/caption each keyframe
            kf_texts: dict[float, str] = {}  # timestamp → visual text
            for kf in keyframes:
                try:
                    kf_doc = self._img_parser.parse(kf["path"], file_id=f"{file_id}_kf")
                    if kf_doc.text.strip():
                        kf_texts[kf["timestamp"]] = kf_doc.text.strip()
                except Exception as exc:
                    log.debug("Keyframe parse failed at %.1fs: %s", kf["timestamp"], exc)

            # ── Audio track ──────────────────────────────────────────────────
            audio_segments: list[dict] = []  # {"start", "end", "text"}
            try:
                audio_path = _extract_audio_track(path, tmp)
                if audio_path is not None:
                    aud_doc = self._aud_parser.parse(audio_path, file_id=f"{file_id}_audio")
                    # Re-parse raw segment info from the text pages for alignment
                    # (AudioParser stores per-chunk text; we use page boundaries)
                    audio_segments = [
                        {"page_text": p, "page_idx": i}
                        for i, p in enumerate(aud_doc.pages)
                    ]
            except ImportError as exc:
                log.warning("Audio transcription disabled: %s", exc)
            except Exception as exc:
                log.warning("Audio transcription failed for %s: %s", path.name, exc)

        # ── Merge into scene chunks ──────────────────────────────────────────
        pages = self._merge_scenes(keyframes, kf_texts, audio_segments)
        full_text = "\n\n".join(pages)

        metadata = {
            "file_name": path.name,
            "format": "video",
            "modality": "video",
            "keyframe_count": len(keyframes),
            "scene_count": len(pages),
        }

        log.debug(
            "Video parsed: %s — %d keyframes, %d scenes",
            path.name, len(keyframes), len(pages),
        )

        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=full_text,
            pages=pages,
            metadata=metadata,
            images=[
                {"page": i, "timestamp": kf["timestamp"], "path": str(kf["path"])}
                for i, kf in enumerate(keyframes)
            ],
        )

    # ------------------------------------------------------------------ #

    def _merge_scenes(
        self,
        keyframes: list[dict],
        kf_texts: dict[float, str],
        audio_segments: list[dict],
    ) -> list[str]:
        """Combine visual and audio content into per-scene text chunks."""
        pages: list[str] = []

        # Build visual-only pages from keyframes
        for i, kf in enumerate(keyframes):
            ts = kf["timestamp"]
            visual_text = kf_texts.get(ts, "").strip()

            # Attach the audio segment whose index most closely matches
            # (simple round-robin assignment; full alignment needs timestamps
            # from whisper which would require restructuring AudioParser)
            audio_text = ""
            if audio_segments and i < len(audio_segments):
                audio_text = audio_segments[i]["page_text"].strip()

            parts = [_fmt_hms(ts)]
            if visual_text:
                parts.append(f"Visual: {visual_text}")
            if audio_text:
                parts.append(f"Audio: {audio_text}")

            if len(parts) > 1:  # at least one content element
                pages.append("  ".join(parts))

        # If no keyframes but we have audio, fall back to audio-only pages
        if not keyframes and audio_segments:
            for seg in audio_segments:
                pages.append(seg["page_text"])

        return pages if pages else ["[No extractable content]"]
