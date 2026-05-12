"""PDF parser using PyMuPDF (fitz).

Text extraction: page-by-page via get_text("text")
Image extraction: saves embedded images to a cache dir for multimodal pipeline
Table extraction: simple heuristic using word blocks
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from .base import BaseParser, ParsedDocument

log = logging.getLogger(__name__)

# Minimum image dimension to be worth extracting (skip tiny icons/logos)
_MIN_IMAGE_DIM = 100  # pixels


class PDFParser(BaseParser):
    """Parses PDF files into text, pages, and image references.

    Args:
        extract_images:  if True, embedded images are saved to image_cache_dir
        image_cache_dir: directory to write extracted images (defaults to temp)
        max_images:      maximum number of images to extract per PDF
    """

    def __init__(
        self,
        extract_images: bool = True,
        image_cache_dir: Path | None = None,
        max_images: int = 50,
    ) -> None:
        self._extract_images = extract_images
        self._image_cache_dir = image_cache_dir
        self._max_images = max_images

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        try:
            import fitz  # PyMuPDF  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF") from exc

        pages: list[str] = []
        images: list[dict] = []
        tables: list[str] = []
        metadata: dict = {}

        with fitz.open(str(path)) as doc:
            raw_meta = doc.metadata or {}
            title = (raw_meta.get("title") or "").strip() or path.stem
            metadata = {
                "title": title,
                "author": raw_meta.get("author", ""),
                "page_count": len(doc),
                "file_name": path.name,
                "format": "pdf",
            }

            image_dir = self._get_image_dir(file_id)
            extracted_count = 0

            for page_num, page in enumerate(doc):
                # Text
                text = page.get_text("text")
                pages.append(text)

                # Tables (heuristic: dense word blocks with tab alignment)
                block_text = page.get_text("blocks")
                table_lines = self._extract_table_lines(block_text)
                tables.extend(table_lines)

                # Images
                if self._extract_images and extracted_count < self._max_images:
                    for img_info in page.get_images(full=True):
                        if extracted_count >= self._max_images:
                            break
                        xref = img_info[0]
                        img_meta = self._extract_image(doc, xref, page_num, image_dir, extracted_count)
                        if img_meta:
                            images.append(img_meta)
                            extracted_count += 1
                elif not self._extract_images:
                    # Still record references for later extraction
                    for img_info in page.get_images(full=True):
                        images.append({"page": page_num, "xref": img_info[0]})

        full_text = "\n\n".join(pages)
        log.debug(
            "PDF parsed: %s — %d pages, %d chars, %d images",
            path.name, len(pages), len(full_text), len(images),
        )
        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=full_text,
            pages=pages,
            metadata=metadata,
            images=images,
            tables=tables,
        )

    # ------------------------------------------------------------------ #

    def _get_image_dir(self, file_id: str) -> Path:
        if self._image_cache_dir is not None:
            d = self._image_cache_dir / file_id
        else:
            d = Path(tempfile.gettempdir()) / "deepsearch_images" / file_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _extract_image(doc, xref: int, page_num: int, image_dir: Path, idx: int) -> dict | None:
        """Extract a single image by xref, save to disk, return metadata dict."""
        try:
            img_data = doc.extract_image(xref)
            if not img_data:
                return None

            w, h = img_data.get("width", 0), img_data.get("height", 0)
            if w < _MIN_IMAGE_DIM or h < _MIN_IMAGE_DIM:
                return None  # skip tiny images

            ext = img_data.get("ext", "png")
            filename = f"page{page_num + 1}_img{idx}.{ext}"
            img_path = image_dir / filename
            img_path.write_bytes(img_data["image"])

            return {
                "page": page_num,
                "xref": xref,
                "path": str(img_path),
                "width": w,
                "height": h,
                "colorspace": img_data.get("colorspace", ""),
            }
        except Exception as exc:
            log.debug("Image extraction failed for xref %d: %s", xref, exc)
            return None

    @staticmethod
    def _extract_table_lines(blocks) -> list[str]:
        """Heuristic: find blocks with multiple tab-separated columns."""
        table_lines: list[str] = []
        for block in blocks:
            if not isinstance(block, tuple) or len(block) < 5:
                continue
            text = block[4]
            if isinstance(text, str) and "\t" in text:
                row = " | ".join(cell.strip() for cell in text.split("\t") if cell.strip())
                if row:
                    table_lines.append(row)
        return table_lines
