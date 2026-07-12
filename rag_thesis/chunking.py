"""Chunking-Strategien für die Vektordatenbank.

Zwei Verfahren:
- semantic_chunking_with_overlap respektiert Markdown-Überschriften, was
  H3 (semantische Vorverarbeitung verbessert Faktentreue) prüfbar macht.
- raw_chunking ignoriert Struktur und dient als Vergleichs-Baseline.

Beide Funktionen sind reine Stringoperationen ohne Side-Effects, 
damit sie vollständig unit-testbar bleiben.
"""

from __future__ import annotations

import re
from typing import List

from .config import CHUNK_OVERLAP, CHUNK_TARGET_SIZE


def semantic_chunking_with_overlap(
    text: str,
    *,
    target_size: int = CHUNK_TARGET_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Markdown-strukturbewusstes Chunking mit Overlap.

    Splittet zuerst entlang von #/##/###-Überschriften. Sehr lange
    Sektionen werden zusätzlich entlang von Absätzen geteilt. Zwischen
    aufeinanderfolgenden Chunks wird ein Overlap (in Zeichen) erhalten,
    damit Kontextinformation an den Schnittstellen nicht verloren geht.
    """
    sections = [s.strip() for s in re.split(r"(?=\n#{1,3} )", text) if s.strip()]
    final_chunks: List[str] = []
    current_chunk = ""

    for section in sections:
        if len(section) > target_size * 2:
            if current_chunk:
                final_chunks.append(current_chunk.strip())
                current_chunk = ""
            para_buffer = ""
            for para in re.split(r"\n\n+", section):
                if not para.strip():
                    continue
                if len(para_buffer) + len(para) <= target_size:
                    para_buffer += "\n\n" + para
                else:
                    if para_buffer.strip():
                        final_chunks.append(para_buffer.strip())
                    overlap_text = para_buffer[-overlap:] if len(para_buffer) > overlap else para_buffer
                    para_buffer = overlap_text + "\n\n" + para
            if para_buffer.strip():
                final_chunks.append(para_buffer.strip())
        else:
            if len(current_chunk) + len(section) <= target_size:
                current_chunk += "\n\n" + section
            else:
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + section

    if current_chunk.strip():
        final_chunks.append(current_chunk.strip())
    return [c for c in final_chunks if c.strip()]


def raw_chunking(text: str, *, target_size: int = CHUNK_TARGET_SIZE) -> List[str]:
    """Naives Absatz-basiertes Chunking ohne Strukturwissen (Vergleichs-Baseline)."""
    chunks: List[str] = []
    curr = ""
    for p in re.split(r"\n\n+", text):
        if not p.strip():
            continue
        if len(curr) + len(p) <= target_size:
            curr += p + "\n\n"
        else:
            if curr.strip():
                chunks.append(curr.strip())
            curr = p + "\n\n"
    if curr.strip():
        chunks.append(curr.strip())
    return chunks
