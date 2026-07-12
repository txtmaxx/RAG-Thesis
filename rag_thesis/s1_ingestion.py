"""Schritt 1 - PDF-Ingestion: OCR-Bereinigung, Bildanalyse, Chunking, Vektor-DB.

Output:
- 1_pdf_ingestion_semantic.json: semantisch bereinigte, gechunkte Inhalte
- 1_pdf_ingestion_raw.json: Rohtext-Chunks (Baseline für H3)
- 1_pdf_pages_raw.json: Roh-Text + Bildbeschreibungen pro PDF-Seite
  (kanonische Quelle für die überarbeitete Faithfulness-Bewertung in Schritt 5)
- chroma_db/: persistente ChromaDB-Collections (cosine)

Methodisch zentral: Damit H3 (Semantic vs. Raw) fair messbar ist, enthält die
kanonische Referenz dieselben Informationen, die der Semantic-Mode hinzufügt
(Bildbeschreibungen). Sonst würden semantische Antworten, die auf Bildinhalte
referenzieren, systematisch als unfaithful gewertet, obwohl sie korrekt sind.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, cast

import chromadb
import fitz
import pymupdf4llm
from PIL import Image, ImageOps

from . import config, prompts
from .chunking import raw_chunking, semantic_chunking_with_overlap
from .io_utils import load_json, rel, save_json, setup_logger
from .llm_client import chat_complete, embed_texts


# ─── Bildverarbeitung ─────────────────────────────────────────────────────────

def _normalize_image_bytes(img: Image.Image) -> bytes:
    """Wandle ein PIL-Bild in deterministische PNG-Bytes (EXIF korrigiert, RGB).

    Voraussetzung für stabile Hashes. Sonst würden identische Bilder mit
    unterschiedlicher EXIF-Orientierung als verschieden gelten.
    """
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _image_hash(img: Image.Image) -> str:
    """SHA-256-Fingerprint eines Bildes (für Duplikat-Erkennung über Seiten hinweg)."""
    return hashlib.sha256(_normalize_image_bytes(img)).hexdigest()


def _render_full_page(page: fitz.Page) -> Image.Image:
    """Rendere eine PDF-Seite als hochauflösendes Bild (2x Skalierung)."""
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _extract_embedded_images(page: fitz.Page) -> List[Image.Image]:
    """Liefere alle in der Seite eingebetteten Bilder oberhalb der Mindestgröße."""
    images: List[Image.Image] = []
    doc = page.parent
    if not doc:
        return images
    for img in page.get_images(full=True):
        try:
            base = doc.extract_image(img[0])
            pil_img = ImageOps.exif_transpose(
                Image.open(BytesIO(base["image"]))
            ).convert("RGB")
            if pil_img.width >= config.MIN_IMAGE_SIZE and pil_img.height >= config.MIN_IMAGE_SIZE:
                images.append(pil_img)
        except Exception:
            continue
    return images


# Prompt-Texte liegen zentral in prompts.py (Prompt Engineering an einer Stelle).
_IMG_INSTRUCTION_FULL = prompts.IMG_INSTRUCTION_FULL
_IMG_INSTRUCTION_EMBEDDED = prompts.IMG_INSTRUCTION_EMBEDDED
_TABLE_GUARDRAILS = prompts.TABLE_GUARDRAILS


def _describe_image_content(
    img: Image.Image,
    context: str,
    seen_hashes: Set[str],
    *,
    is_full_page: bool = False,
    images_dir: Optional[str] = None,
) -> Optional[str]:
    """Erzeuge eine Vision-Beschreibung des Bildes oder None bei Duplikat/Leer.

    Voll-Seiten-Scans werden nie als Duplikate verworfen (jede Seite zählt einmal).
    Bei images_dir werden die analysierten Bilder zur Nachprüfbarkeit gespeichert.
    """
    img_hash = _image_hash(img)
    if not is_full_page and img_hash in seen_hashes:
        return None
    seen_hashes.add(img_hash)
    if max(img.size) > config.MAX_IMAGE_DIM:
        img.thumbnail((config.MAX_IMAGE_DIM, config.MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    prompt = prompts.image_analysis_prompt(is_full_page=is_full_page, context=context)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_str}",
                               "detail": config.IMAGE_DETAIL}},
            ],
        }
    ]
    try:
        desc = chat_complete(messages=messages, model=config.MODEL_VISION,
                             max_tokens=800, temperature=config.TEMPERATURE).strip()
        if not desc or "KEINE_INFO" in desc or "KEINE_GRAFIK" in desc:
            return None
        if images_dir:
            from pathlib import Path
            try:
                with open(Path(images_dir) / f"{img_hash}.jpg", "wb") as fh:
                    fh.write(buffered.getvalue())
            except Exception:
                pass
        return desc
    except Exception as e:
        logging.error(f"Bildbeschreibung fehlgeschlagen: {e}")
        return None


# ─── Text-Bereinigung ─────────────────────────────────────────────────────────

def _post_process(content: str) -> str:
    """Entferne typische LLM-Artefakte (Einleitungssätze, Code-Fences, Seitenzahlen)."""
    lines = content.split("\n")
    if lines and (lines[0].startswith("Hier ist") or lines[0].startswith("Gerne")):
        content = "\n".join(lines[1:]).strip()
    if content.startswith("```"):
        content = content.strip("`").replace("markdown", "").replace("latex", "").strip()
    content = re.sub(r"\n\s*\$?\d+\$?\s*$", "", content)
    return content.strip()


_MATH_HEAVY_PATTERNS = ("\\frac", "\\sum", "\\int", "\\begin", "\\prod", "\\lim")


def _is_math_heavy(text: str) -> bool:
    """Heuristik: Lohnt sich das stärkere Modell für die OCR-Bereinigung?"""
    if len(text) < 50:
        return False
    special = len(re.findall(r"[\\$|_^{}\[\]]", text))
    density = special / len(text)
    return density > 0.05 or text.count("|") > 15 or any(p in text for p in _MATH_HEAVY_PATTERNS)


def _clean_text_with_llm(text_content: str) -> str:
    """OCR-Text in sauberes Markdown wandeln, Smart-Routing nach Komplexität."""
    if len(text_content) < 50:
        return text_content
    model = config.MODEL_TEXT_ADVANCED if _is_math_heavy(text_content) else config.MODEL_TEXT
    safe_input = text_content[:12000]
    if len(text_content) > 12000:
        logging.warning(f"Text gekürzt von {len(text_content)} auf 12000 Zeichen.")
    prompt = prompts.text_cleaning_prompt(safe_input)
    try:
        content = chat_complete(
            messages=[{"role": "user", "content": prompt}],
            model=model, max_tokens=4096, temperature=config.TEMPERATURE,
        )
        return _post_process(content)
    except Exception as e:
        logging.error(f"Text-Bereinigung fehlgeschlagen: {e}")
        return text_content


# ─── Page-Helper ──────────────────────────────────────────────────────────────

def _apply_cropping(doc: fitz.Document, page_indices: List[int]) -> None:
    """Schneide den oberen und unteren Seitenrand jeder Seite ab.

    HEADER_CUTOFF_RATIO entfernt den oberen Anteil (Kopfzeilen),
    FOOTER_CUTOFF_RATIO den unteren Anteil (Fußzeilen, Seitenzahlen, Logos).
    Das geschieht vor der Textextraktion durch PyMuPDF und verhindert, dass
    wiederkehrendes Rand-Rauschen in die Chunks gerät. Der Header-Default ist
    0,0, sodass standardmäßig nur die Fußzeile betroffen ist.
    """
    for i in page_indices:
        page = doc[i]
        rect = page.rect
        page.set_cropbox(fitz.Rect(
            rect.x0,
            rect.y0 + rect.height * config.HEADER_CUTOFF_RATIO,
            rect.x1,
            rect.y1 - rect.height * config.FOOTER_CUTOFF_RATIO,
        ))


def _page_text_from_md_chunks(md_chunks: list, page_index: int) -> str:
    """Sammle alle Markdown-Chunks, die page_index zugeordnet sind, zu einem Text.

    Robust gegenüber unterschiedlichen Metadaten-Schemas in pymupdf4llm-Versionen.
    """
    texts: List[str] = []
    for chunk in md_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_page = next(
            (chunk[k] for k in ("page", "page_index", "start_page", "start") if k in chunk),
            None,
        )
        if chunk_page == page_index:
            texts.append(chunk.get("text", ""))
    if not texts:
        for chunk in md_chunks:
            if isinstance(chunk, dict):
                meta = chunk.get("metadata", {})
                if isinstance(meta, dict) and meta.get("page") == page_index:
                    texts.append(chunk.get("text", ""))
    return "\n\n".join(texts).strip()


_DIAGRAM_KEYWORDS = (
    "abbildung", "figur", "graph", "diagramm", "skizze", "schema", "pfeil", "darstellung",
)


# ─── Vektor-DB ────────────────────────────────────────────────────────────────

def _build_vector_db_and_save(
    chunks: List[str],
    output_path,
    collection_name: str,
    *,
    page_mapping: Optional[Dict[int, int]] = None,
) -> None:
    """Persistiere chunks als JSON UND embedde sie in eine ChromaDB-Collection.

    page_mapping (Chunk-Index -> Seitennummer) wird als Metadatum mitgespeichert
    und ist Voraussetzung für die H3-Bewertung gegen den kanonischen PDF-Text.
    """
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    json_output: List[Dict] = []
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        try:
            embeddings = embed_texts(batch)
            ids = [f"chunk_{i + j}" for j in range(len(batch))]
            metadatas: List[Dict] = []
            for j in range(len(batch)):
                meta: Dict = {"chunk_index": int(i + j)}
                if page_mapping and (i + j) in page_mapping:
                    meta["page_number"] = int(page_mapping[i + j])
                metadatas.append(meta)
            # cast: chromadb-Stubs verlangen OneOrMany[...]. Laufzeit-Listen sind korrekt.
            collection.add(documents=batch, embeddings=cast(Any, embeddings),
                           ids=ids, metadatas=cast(Any, metadatas))
            for j, text in enumerate(batch):
                entry: Dict = {"chunk_id": ids[j], "content": text}
                if page_mapping and (i + j) in page_mapping:
                    entry["page_number"] = page_mapping[i + j]
                json_output.append(entry)
        except Exception as e:
            logging.error(f"Embedding-Batch {i} fehlgeschlagen: {e}")

    save_json(output_path, json_output)
    logging.info(f"{len(json_output)} Chunks gespeichert ({rel(output_path)}, Collection '{collection_name}').")


def _save_pdf_pages_raw(doc: fitz.Document, page_indices: List[int]) -> None:
    """Persistiere den PDF-Roh-Text pro Seite, initialer kanonischer Quelltext.

    image_descriptions startet leer und wird im Anschluss durch _enrich_pages_with_image_descriptions 
    mit den Bildanalysen des semantischen Modus angereichert. So bleibt die Datei selbst dann gültig,
    wenn nur der raw-Modus läuft.
    """
    pages: List[Dict] = []
    for p in page_indices:
        # Cropping wurde global angewandt. Gelesen wird der (gecroppte) Seitentext.
        pages.append({
            "page_number": p + 1,
            "text": str(doc[p].get_text("text")),
            "image_descriptions": [],
        })
    save_json(config.FILE_PDF_PAGES_RAW, pages)
    logging.info(f"PDF-Roh-Seitentexte gespeichert: {len(pages)} Seiten.")


def _enrich_pages_with_image_descriptions(
    page_to_descs: Dict[int, List[str]],
) -> None:
    """Reichere die kanonische Seiten-JSON um die Bildbeschreibungen an,
    die der Semantic-Mode pro Seite erzeugt hat. Wird einmal am Ende der
    semantischen Ingestion aufgerufen."""
    if not config.FILE_PDF_PAGES_RAW.exists():
        logging.warning("Keine Seiten-JSON zum Anreichern gefunden - überspringe.")
        return
    pages = load_json(config.FILE_PDF_PAGES_RAW)
    n_total = 0
    for entry in pages:
        descs = page_to_descs.get(int(entry["page_number"]), [])
        entry["image_descriptions"] = descs
        n_total += len(descs)
    save_json(config.FILE_PDF_PAGES_RAW, pages)
    logging.info(
        f"Kanonische Seiten-JSON angereichert: "
        f"{n_total} Bildbeschreibungen über {len(pages)} Seiten."
    )


_PAGE_HEADER_RE = re.compile(r"^###\s+Seite\s+(\d+)\b", re.MULTILINE)


def _infer_chunk_page_mapping(chunks: List[str]) -> Dict[int, int]:
    """Leite pro semantischem Chunk die Start-Seitennummer ab.

    Der Chunker konsumiert den joined-text ### Seite N\\n…-Blöcken. Nach
    dem Chunking sind die Marker für nahezu alle Chunks rekonstruierbar, indem
    die Chunks in Reihenfolge durchgegangen werden und der jeweils zuletzt
    gesehene ### Seite N-Marker mitgeführt wird. So bekommt auch ein Chunk
    ohne eigenen Marker (z.B. Fortsetzung einer langen Seite) eine korrekte
    Page-Zuordnung, Voraussetzung für die methodisch faire H3-Bewertung in
    Schritt 5 (Faithfulness gegen kanonischen Quelltext).

    Ohne diese Zuordnung fielen Items in Schritt 5 auf den retrieved_context-Fallback zurück, 
    sobald page_number=None blieb. Die H3-Aussage wäre damit invalidiert 
    (jedes System misst gegen seinen eigenen Kontext).
    """
    page_mapping: Dict[int, int] = {}
    current_page: Optional[int] = None
    for idx, chunk in enumerate(chunks):
        m = _PAGE_HEADER_RE.search(chunk)
        if m:
            current_page = int(m.group(1))
        if current_page is not None:
            page_mapping[idx] = current_page
    return page_mapping


# ─── Hauptpipeline ────────────────────────────────────────────────────────────

def ingest(pdf_path, mode: str) -> None:
    """Kompletter Ingestion-Lauf für einen Modus (semantic oder raw)."""
    logging.info(f"=== INGESTION (mode={mode}) ===")
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    real_end = total if config.END_PAGE is None or config.END_PAGE > total else config.END_PAGE
    real_start = max(0, config.START_PAGE - 1)
    page_indices = list(range(real_start, real_end))
    _apply_cropping(doc, page_indices)

    # Persistiere Roh-Seitentexte einmal. Beim ersten Modus reicht es.
    if mode == "semantic" or not config.FILE_PDF_PAGES_RAW.exists():
        _save_pdf_pages_raw(doc, page_indices)

    full_parts: List[str] = []

    if mode == "semantic":
        md_chunks_raw = pymupdf4llm.to_markdown(doc, pages=page_indices, page_chunks=True)
        md_chunks: list = md_chunks_raw if isinstance(md_chunks_raw, list) else [{"text": str(md_chunks_raw)}]
        seen_hashes: Set[str] = set()
        images_dir = config.DIR_INGESTION / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        # Bildbeschreibungen pro Seite sammeln -> kanonischer Kontext (Schritt 5)
        # enthält dieselbe Info wie der Semantic-Index (faire H3-Bedingung).
        page_to_image_descs: Dict[int, List[str]] = {}

        for p in page_indices:
            page_num = p + 1
            combined_text = _page_text_from_md_chunks(md_chunks, p)
            if not combined_text:
                combined_text = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in md_chunks
                )[:5000]

            page_obj = doc[p]
            cleaned_text = _clean_text_with_llm(combined_text)

            text_lower = cleaned_text.lower()
            has_diagram_ref = any(k in text_lower for k in _DIAGRAM_KEYWORDS)
            embedded_imgs: List[Image.Image] = []
            use_full_page_scan = has_diagram_ref or len(cleaned_text) < 200

            if not use_full_page_scan:
                embedded_imgs = _extract_embedded_images(page_obj)
                use_full_page_scan = len(embedded_imgs) > 2
                if use_full_page_scan:
                    embedded_imgs = []

            page_descs: List[str] = []
            if embedded_imgs:
                descs = [
                    _describe_image_content(img, cleaned_text, seen_hashes,
                                             images_dir=str(images_dir))
                    for img in embedded_imgs
                ]
                page_descs.extend([d for d in descs if d])
                desc_iter = iter(f"\n[BILD-INFO: {d}]\n" if d else "" for d in descs)
                if any(descs):
                    cleaned_text = re.sub(r"!\[.*?\]\(.*?\)",
                                           lambda m: next(desc_iter, ""), cleaned_text)
                else:
                    cleaned_text = re.sub(r"!\[.*?\]\(.*?\)", "", cleaned_text)
            else:
                cleaned_text = re.sub(r"!\[.*?\]\(.*?\)", "", cleaned_text)

            if use_full_page_scan:
                desc = _describe_image_content(
                    _render_full_page(page_obj), cleaned_text, seen_hashes,
                    is_full_page=True, images_dir=str(images_dir),
                )
                if desc:
                    page_descs.append(desc)
                    cleaned_text += f"\n\n[SEITEN-GRAFIK-ANALYSE: {desc}]\n"

            page_to_image_descs[page_num] = page_descs
            full_parts.append(f"### Seite {page_num}\n{cleaned_text}")

        chunks = semantic_chunking_with_overlap("\n\n".join(full_parts))
        page_mapping = _infer_chunk_page_mapping(chunks)
        if len(page_mapping) < len(chunks):
            logging.warning(
                f"Page-Mapping nur für {len(page_mapping)}/{len(chunks)} Chunks "
                f"ableitbar - H3-Faithfulness kann für die Lücken nicht gegen "
                f"den kanonischen Quelltext gemessen werden."
            )
        else:
            logging.info(
                f"Page-Mapping vollständig: {len(page_mapping)}/{len(chunks)} Chunks "
                f"einer Seite zugeordnet (Voraussetzung für H3-Fairness)."
            )
        _build_vector_db_and_save(chunks, config.FILE_INGESTION_SEMANTIC,
                                   "vorlesung_skript_semantic",
                                   page_mapping=page_mapping)
        _enrich_pages_with_image_descriptions(page_to_image_descs)

    elif mode == "raw":
        for p in page_indices:
            full_parts.append(str(doc[p].get_text("text")))
        chunks = raw_chunking("\n\n".join(full_parts))
        _build_vector_db_and_save(chunks, config.FILE_INGESTION_RAW,
                                   "vorlesung_skript_raw")
    else:
        raise ValueError(f"Unbekannter Modus: {mode}")

    doc.close()
    logging.info(f"Ingestion ({mode}) abgeschlossen - {len(chunks)} Chunks.")


def main() -> None:
    """CLI-Einstieg für einen einzelnen Ingestion-Lauf (--mode semantic oder raw)."""
    parser = argparse.ArgumentParser(description="PDF-Ingestion in Vektordatenbank.")
    parser.add_argument("--mode", type=str, default="semantic",
                        choices=["semantic", "raw"],
                        help="Chunking-Strategie (default: semantic).")
    args = parser.parse_args()

    config.ensure_output_dirs()
    setup_logger(config.DIR_INGESTION / "1_ingestion.log")
    ingest(config.PDF_INPUT_FILE, args.mode)


if __name__ == "__main__":
    main()
