"""Hierarchical text chunker.

Produces chunks at three granularity levels:
  level 0 — parent  : ~1024 tokens (broad context)
  level 1 — chunk   : ~512  tokens (primary retrieval unit)
  level 2 — child   : ~128  tokens (precise answer location)

Each child stores parent_id so the retrieval layer can fetch
the parent chunk for richer context window stuffing.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    chunk_id: str
    file_id: str
    text: str
    level: int                       # 0=parent, 1=chunk, 2=child
    chunk_index: int                 # position within file
    parent_id: Optional[str]         # None for level-0
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token count: ~4 chars per token."""
        return len(self.text) // 4


def _split_by_sentences(text: str, max_chars: int) -> list[str]:
    """Split text into segments of up to max_chars, respecting sentence
    boundaries where possible."""
    # Sentence boundary regex
    sentence_end = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end.split(text.strip())

    segments: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                segments.append(current)
            # If a single sentence exceeds max_chars, hard-split it
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    segments.append(sent[i : i + max_chars])
            else:
                current = sent
    if current:
        segments.append(current)
    return segments


class HierarchicalChunker:
    """Three-level hierarchical chunker.

    Args:
        parent_size_chars: target parent chunk size in characters (~4 chars/token)
        chunk_size_chars: target level-1 chunk size
        child_size_chars: target level-2 chunk size
        overlap_chars: overlap between consecutive level-1 chunks
    """

    def __init__(
        self,
        parent_size_chars: int = 4096,  # ~1024 tokens
        chunk_size_chars: int = 2048,   # ~512 tokens
        child_size_chars: int = 512,    # ~128 tokens
        overlap_chars: int = 200,
    ) -> None:
        self.parent_size = parent_size_chars
        self.chunk_size = chunk_size_chars
        self.child_size = child_size_chars
        self.overlap = overlap_chars

    def chunk(self, text: str, file_id: str, base_metadata: dict | None = None) -> list[Chunk]:
        """Return flat list of all chunks (all levels) for a document."""
        meta = base_metadata or {}
        all_chunks: list[Chunk] = []
        chunk_index = 0

        parent_segs = _split_by_sentences(text, self.parent_size)

        for p_seg in parent_segs:
            parent_id = str(uuid.uuid4())
            parent_chunk = Chunk(
                chunk_id=parent_id,
                file_id=file_id,
                text=p_seg,
                level=0,
                chunk_index=chunk_index,
                parent_id=None,
                metadata={**meta, "chunk_level": 0},
            )
            all_chunks.append(parent_chunk)
            chunk_index += 1

            # Level-1 chunks (with overlap)
            level1_segs = self._split_with_overlap(p_seg, self.chunk_size, self.overlap)
            for l1_seg in level1_segs:
                l1_id = str(uuid.uuid4())
                l1_chunk = Chunk(
                    chunk_id=l1_id,
                    file_id=file_id,
                    text=l1_seg,
                    level=1,
                    chunk_index=chunk_index,
                    parent_id=parent_id,
                    metadata={**meta, "chunk_level": 1},
                )
                all_chunks.append(l1_chunk)
                chunk_index += 1

                # Level-2 children
                level2_segs = _split_by_sentences(l1_seg, self.child_size)
                for l2_seg in level2_segs:
                    l2_chunk = Chunk(
                        chunk_id=str(uuid.uuid4()),
                        file_id=file_id,
                        text=l2_seg,
                        level=2,
                        chunk_index=chunk_index,
                        parent_id=l1_id,
                        metadata={**meta, "chunk_level": 2},
                    )
                    all_chunks.append(l2_chunk)
                    chunk_index += 1

        return all_chunks

    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
        segments: list[str] = []
        start = 0
        while start < len(text):
            end = start + size
            segments.append(text[start:end])
            start += size - overlap
        return segments
