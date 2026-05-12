"""PowerPoint (.pptx) parser — extracts text from slides, notes, and shapes.

Requires: python-pptx  (pip install python-pptx)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import BaseParser, ParsedDocument

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def _shape_text(shape) -> str:
    """Recursively extract text from a pptx Shape, including grouped shapes."""
    parts: list[str] = []
    # Direct text frame
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            line = " ".join(run.text for run in para.runs if run.text.strip())
            if line.strip():
                parts.append(line.strip())
    # Tables
    if shape.has_table:
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    # Group shapes (recurse)
    if shape.shape_type == 6:  # MSO_SHAPE_TYPE.GROUP
        for child in shape.shapes:
            child_text = _shape_text(child)
            if child_text:
                parts.append(child_text)
    return "\n".join(parts)


class PPTXParser(BaseParser):
    """Parses PowerPoint files into slide-by-slide text."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".pptx", ".ppt"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        try:
            from pptx import Presentation  # type: ignore[import]
            from pptx.exc import PackageNotFoundError  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "python-pptx is required for PPTX parsing. "
                "Run: pip install python-pptx"
            ) from exc

        try:
            prs = Presentation(str(path))
        except Exception as exc:
            return ParsedDocument(
                file_id=file_id,
                source_path=str(path),
                text=f"[PPTX parse error: {exc}]",
                metadata={"title": path.stem, "format": "pptx", "file_name": path.name},
            )

        pages: list[str] = []
        all_text_parts: list[str] = []
        title: str = path.stem

        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_parts: list[str] = []

            # Slide title (first placeholder with idx=0 or type TITLE)
            slide_title = self._slide_title(slide)
            if slide_num == 1 and slide_title:
                title = slide_title

            header = f"=== Slide {slide_num}" + (f": {slide_title}" if slide_title else "") + " ==="
            slide_parts.append(header)

            # All shapes on the slide
            for shape in slide.shapes:
                text = _shape_text(shape)
                if text.strip():
                    slide_parts.append(text)

            # Speaker notes
            if slide.has_notes_slide:
                notes_frame = slide.notes_slide.notes_text_frame
                if notes_frame:
                    notes = notes_frame.text.strip()
                    if notes:
                        slide_parts.append(f"[Notes: {notes}]")

            page_text = "\n".join(slide_parts)
            pages.append(page_text)
            all_text_parts.append(page_text)

        full_text = "\n\n".join(all_text_parts)

        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=full_text,
            pages=pages,
            metadata={
                "title": title,
                "format": "pptx",
                "file_name": path.name,
                "slide_count": len(prs.slides),
            },
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _slide_title(slide) -> str:
        """Return the slide title or empty string."""
        try:
            from pptx.enum.shapes import PP_PLACEHOLDER  # type: ignore[import]
            for ph in slide.placeholders:
                if ph.placeholder_format.idx == 0:  # TITLE placeholder
                    return ph.text.strip()
        except Exception:
            pass
        return ""
