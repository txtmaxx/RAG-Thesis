"""Tests für den angereicherten kanonischen Kontext.

Verifiziert, dass:
- _build_canonical_context_lookup einen Callable zurückliefert (oder None),
- der gelieferte Text PDF-Rohtext plus Bildbeschreibungen kombiniert,
- der Lookup auf source_reference-Strings korrekt mappt,
- bei fehlenden Eingabedaten sauber None zurückkommt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_thesis import config, s5_evaluation as evaluation


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def fake_corpus(tmp_path, monkeypatch):
    """Lege ein minimales Korpus an: 2 Seiten, je 1 Chunk, 1 Seite mit Bild."""
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(config, "DIR_INGESTION", tmp_path / "ingestion")
    monkeypatch.setattr(config, "FILE_PDF_PAGES_RAW",
                         tmp_path / "ingestion" / "1_pdf_pages_raw.json")
    monkeypatch.setattr(config, "FILE_INGESTION_SEMANTIC",
                         tmp_path / "ingestion" / "1_pdf_ingestion_semantic.json")

    _write_json(config.FILE_PDF_PAGES_RAW, [
        {"page_number": 1, "text": "Seite eins Text.", "image_descriptions": []},
        {"page_number": 2, "text": "Seite zwei Text.",
         "image_descriptions": ["Diagramm: Architektur eines RAG-Systems."]},
    ])
    _write_json(config.FILE_INGESTION_SEMANTIC, [
        {"chunk_id": "chunk_0", "content": "### Seite 1\nSeite eins Text.",
         "page_number": 1},
        {"chunk_id": "chunk_1", "content": "### Seite 2\nSeite zwei Text.",
         "page_number": 2},
    ])
    return tmp_path


def test_lookup_returns_callable_when_data_present(fake_corpus):
    lookup = evaluation._build_canonical_context_lookup()
    assert callable(lookup)


def test_lookup_includes_image_descriptions(fake_corpus):
    lookup = evaluation._build_canonical_context_lookup()
    assert lookup is not None
    context = lookup("chunk_1_to_chunk_1")
    assert "Seite zwei Text." in context
    assert "BILD-INFO" in context
    assert "Architektur eines RAG-Systems" in context


def test_lookup_spans_pages_with_buffer(fake_corpus):
    lookup = evaluation._build_canonical_context_lookup()
    assert lookup is not None
    # chunk_0 -> Seite 1; mit ±1-Puffer werden Seiten 1 und 2 zurückgegeben.
    context = lookup("chunk_0_to_chunk_0")
    assert "Seite eins Text." in context
    assert "Seite zwei Text." in context


def test_lookup_returns_none_when_data_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FILE_PDF_PAGES_RAW", tmp_path / "missing_pages.json")
    monkeypatch.setattr(config, "FILE_INGESTION_SEMANTIC",
                         tmp_path / "missing_chunks.json")
    assert evaluation._build_canonical_context_lookup() is None


def test_lookup_handles_pages_without_image_descs(fake_corpus):
    # Seite 1 hat keine Bildbeschreibungen -> kein BILD-INFO-Block angehängt.
    lookup = evaluation._build_canonical_context_lookup()
    assert lookup is not None
    context = lookup("chunk_0_to_chunk_0")
    seite1_block, _, _rest = context.partition("Seite zwei Text.")
    assert "BILD-INFO" not in seite1_block
