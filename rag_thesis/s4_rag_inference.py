"""Schritt 4 - RAG-Inferenz für semantic und raw Modus.

Der Embedding-Cache wird zwischen den beiden Modi geteilt (Frage-Embedding 
einmal berechnet, in beiden Modi genutzt). Reduziert Embedding-Calls um ~50 %.
"""

from __future__ import annotations

import argparse
import logging
import threading
from typing import Dict, List, Optional

import chromadb

from . import config, prompts
from .io_utils import (
    load_checkpoint, load_json, save_json, setup_logger,
)
from .llm_client import chat_complete, embed_texts
from .parallel import run_parallel_with_checkpoint


_MAX_TOKENS = 2000

_chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
_query_lock = threading.Lock()
_embed_cache: Dict[str, List[float]] = {}
_embed_lock = threading.Lock()


# ─── Methodischer Hinweis (Konstruktvalidität) ────────────────────────────────
# Der Transfer-Prompt erlaubt explizit externes Logikwissen zur Brückenbildung,
# der Baseline-Prompt nicht. Diese Asymmetrie ist eine bewusste Limitation.
# Prompt-Texte liegen zentral in prompts.py.
_PROMPTS_BY_TYPE: Dict[str, str] = prompts.RAG_PROMPTS_BY_TYPE
_DEFAULT_PROMPT = prompts.RAG_DEFAULT_PROMPT


def _question_embedding(question: str) -> List[float]:
    """Cached Frage-Embedding (geteilt zwischen semantic/raw Modus)."""
    with _embed_lock:
        if question in _embed_cache:
            return _embed_cache[question]
    emb = embed_texts([question])[0]
    with _embed_lock:
        _embed_cache[question] = emb
    return emb


def _deduplicate(docs: List[str]) -> List[str]:
    """Entferne Whitespace-normalisierte Duplikate aus der Retrieval-Trefferliste.

    Greift nur bei exakter Identität nach Whitespace-Normalisierung. Quasi-
    identische Chunks (OCR-Rauschen) bleiben bestehen. Eine semantische Variante
    (Embedding-Ähnlichkeit/MMR) ist eine bekannte Limitation.
    """
    seen = set()
    unique = []
    for doc in docs:
        normalized = " ".join(doc.split())
        if normalized not in seen:
            seen.add(normalized)
            unique.append(doc)
    return unique


def _retrieve_context(question: str, collection, *, top_k: int = config.TOP_K) -> str:
    """Hole die top_k ähnlichsten Chunks und gib sie als separierten Block zurück."""
    try:
        q_emb = _question_embedding(question)
        with _query_lock:
            results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = results.get("documents")
        if not docs or not docs[0]:
            return ""
        return "\n\n---\n\n".join(_deduplicate([str(d) for d in docs[0] if d is not None]))
    except Exception as e:
        logging.error(f"Context-Retrieval fehlgeschlagen: {e}")
        return ""


def _answer_with_rag(question: str, context: str, qtype: str) -> str:
    """Beantworte question mit context als primärer Wissensquelle."""
    return chat_complete(
        messages=[
            {"role": "system", "content": _PROMPTS_BY_TYPE.get(qtype, _DEFAULT_PROMPT)},
            {"role": "user", "content": f"KONTEXT:\n{context}\n\nFRAGE:\n{question}"},
        ],
        model=config.MODEL_TEXT,
        max_tokens=_MAX_TOKENS,
        temperature=config.TEMPERATURE,
    )


def _process_item(item: Dict, collection) -> Optional[Dict]:
    """Wandle ein Golden-Dataset-Item in einen RAG-Antwort-Record (Retrieval + Antwort).

    Wirft, wenn die Retrieval-Suche leer bleibt. Sonst würde der Datensatz mit
    einer kontextlosen Antwort die spätere Evaluation verzerren.
    """
    question = item.get("question")
    if not isinstance(question, str):
        return None
    context = _retrieve_context(question, collection)
    if not context:
        raise RuntimeError("Kein Kontext gefunden")
    answer = _answer_with_rag(question, context, item.get("question_type", ""))
    return {
        "question": question,
        "ground_truth": item.get("answer"),
        "rag_answer": answer,
        "question_type": item.get("question_type", ""),
        "source_reference": item.get("source_reference", ""),
        "requires_context": item.get("requires_context", True),
        "retrieved_context": context,
    }


def build_rag_answers(mode: str) -> None:
    """Beantworte alle Golden-Dataset-Fragen via RAG für den gewählten Modus.

    mode: "semantic" oder "raw", bestimmt die Vektor-Collection.
    """
    logging.info("=== EXPERIMENT-CONFIG (Schritt 4) ===")
    logging.info(f"Modell: {config.MODEL_TEXT} | T={config.TEMPERATURE} | Mode={mode} "
                 f"| Top-K={config.TOP_K} | Workers={config.MAX_WORKERS}")
    logging.info("=" * 40)

    collection_name = f"vorlesung_skript_{mode}"
    output_file = config.DIR_RAG / f"4_rag_answers_{mode}.json"
    checkpoint_file = config.DIR_RAG / f"4_checkpoint_{mode}.json"

    questions = load_json(config.FILE_GOLDEN_DATASET)
    results = load_checkpoint(checkpoint_file)
    done = {r["question"] for r in results}
    pending = [item for item in questions
                if isinstance(item.get("question"), str) and item["question"] not in done]
    logging.info(f"Verarbeite {len(pending)} Fragen mit {config.MAX_WORKERS} Workern.")

    collection = _chroma.get_collection(name=collection_name)
    run_parallel_with_checkpoint(
        lambda item: _process_item(item, collection),
        pending, checkpoint_file, results,
        max_workers=config.MAX_WORKERS, label="question",
    )

    save_json(output_file, results)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    logging.info(f"RAG-Inferenz ({mode}) fertig. Total: {len(results)}")


def main() -> None:
    """CLI-Einstieg für einen alleinstehenden RAG-Lauf (--mode)."""
    parser = argparse.ArgumentParser(description="RAG-Inferenz mit Vektor-Retrieval.")
    parser.add_argument("--mode", type=str, default="semantic",
                        choices=["semantic", "raw"])
    args = parser.parse_args()

    config.ensure_output_dirs()
    setup_logger(config.DIR_RAG / "4_rag_pipeline.log")
    build_rag_answers(args.mode)


if __name__ == "__main__":
    main()
