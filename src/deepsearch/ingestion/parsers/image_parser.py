"""Image parser — OCR text extraction + optional LLaVA captioning.

Extraction pipeline:
  1. PaddleOCR  → structured text from image (receipts, slides, diagrams)
  2. LLaVA GGUF → natural-language caption for visual understanding (optional)

Both steps use lazy imports so the app runs without these heavy dependencies.

Supported extensions: .jpg .jpeg .png .bmp .tiff .tif .webp .gif
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .base import BaseParser, ParsedDocument

log = logging.getLogger(__name__)

_SUPPORTED = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"]

# Minimum confidence threshold for PaddleOCR text boxes (0-1)
_OCR_CONF_THRESHOLD = 0.5


def _ocr_image(path: Path) -> tuple[str, list[dict]]:
    """Run PaddleOCR on an image file.

    Returns:
        (full_text, boxes) where boxes is a list of
        {"text": str, "confidence": float, "bbox": list}
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PaddleOCR not installed. Run: pip install paddleocr"
        ) from exc

    # use_angle_cls handles rotated text; lang='en' covers most cases
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr.ocr(str(path), cls=True)

    lines: list[str] = []
    boxes: list[dict] = []

    if not result or result[0] is None:
        return "", []

    for page_result in result:
        if page_result is None:
            continue
        for line in page_result:
            bbox, (text, conf) = line
            if conf >= _OCR_CONF_THRESHOLD and text.strip():
                lines.append(text.strip())
                boxes.append({"text": text.strip(), "confidence": float(conf), "bbox": bbox})

    return "\n".join(lines), boxes


def _caption_image(path: Path, llava_model_path: str | None) -> str:
    """Generate a natural-language caption via LLaVA GGUF (optional).

    If llava_model_path is None or llama-cpp-python is not installed,
    returns an empty string (caption is additive — not required).
    """
    if llava_model_path is None:
        return ""

    try:
        from llama_cpp import Llama  # type: ignore[import]
        from llama_cpp.llama_chat_format import Llava15ChatHandler  # type: ignore[import]
    except ImportError:
        log.debug("llama-cpp-python not available — skipping LLaVA caption")
        return ""

    try:
        import base64
        image_bytes = path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode()
        data_url = f"data:image/png;base64,{b64}"

        chat_handler = Llava15ChatHandler(clip_model_path=llava_model_path)
        llm = Llama(
            model_path=llava_model_path,
            chat_handler=chat_handler,
            n_ctx=1024,
            n_gpu_layers=-1,
            verbose=False,
        )
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "Describe this image in detail, focusing on any text, charts, diagrams, or key visual elements."},
                    ],
                }
            ],
            max_tokens=256,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        log.warning("LLaVA captioning failed for %s: %s", path.name, exc)
        return ""


class ImageParser(BaseParser):
    """Parses image files into text via OCR and optional LLaVA captioning.

    Args:
        llava_model_path: Path to a LLaVA GGUF model file.  When None,
                          captioning is skipped (OCR-only mode).
        ocr_enabled:      Set to False to disable PaddleOCR (caption-only mode).
    """

    def __init__(
        self,
        llava_model_path: str | None = None,
        ocr_enabled: bool = True,
    ) -> None:
        self._llava_path = llava_model_path
        self._ocr_enabled = ocr_enabled

    @property
    def supported_extensions(self) -> list[str]:
        return _SUPPORTED

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        parts: list[str] = []
        metadata: dict = {
            "file_name": path.name,
            "format": "image",
            "modality": "image",
        }

        # ── OCR ──────────────────────────────────────────────────────────────
        ocr_text = ""
        ocr_boxes: list[dict] = []
        if self._ocr_enabled:
            try:
                ocr_text, ocr_boxes = _ocr_image(path)
                if ocr_text:
                    parts.append(f"[OCR Text]\n{ocr_text}")
                    metadata["ocr_word_count"] = len(ocr_text.split())
                    metadata["ocr_box_count"] = len(ocr_boxes)
            except ImportError as exc:
                log.warning("OCR skipped — PaddleOCR not installed: %s", exc)
            except Exception as exc:
                log.warning("OCR failed for %s: %s", path.name, exc)

        # ── Caption ──────────────────────────────────────────────────────────
        caption = _caption_image(path, self._llava_path)
        if caption:
            parts.append(f"[Visual Description]\n{caption}")
            metadata["has_caption"] = True

        if not parts:
            log.warning("No text extracted from image: %s", path.name)

        full_text = "\n\n".join(parts)
        log.debug(
            "Image parsed: %s — ocr_chars=%d caption_chars=%d",
            path.name, len(ocr_text), len(caption),
        )

        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=full_text,
            pages=[full_text],
            metadata=metadata,
            images=[{"page": 0, "path": str(path), "ocr_boxes": ocr_boxes}],
        )
