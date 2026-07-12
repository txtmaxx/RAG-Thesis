"""Schritt 3 - Baseline-Inferenz (LLM ohne Dokumentenzugriff).

Liefert die Vergleichs-Antworten für H1: dasselbe Modell beantwortet die
Ground-Truth-Fragen ausschließlich aus seinem internen Wissen.
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, Optional

from . import config, prompts
from .io_utils import (
    load_checkpoint, load_json, save_json, setup_logger,
)
from .llm_client import chat_complete
from .parallel import run_parallel_with_checkpoint


_OUTPUT_FILE = config.FILE_BASELINE_ANSWERS
_CHECKPOINT_FILE = config.DIR_BASELINE / "3_checkpoint.json"
_MAX_TOKENS = 2000


# ─── Methodischer Hinweis (Konstruktvalidität) ────────────────────────────────
# Der Transfer-Prompt ist bewusst nicht symmetrisch zum RAG-Pendant (dort ist
# externes Logikwissen erlaubt), eine bewusste Limitation der
# Konstruktvalidität. Prompt-Texte liegen zentral in prompts.py.
_PROMPTS_BY_TYPE: Dict[str, str] = prompts.BASELINE_PROMPTS_BY_TYPE
_DEFAULT_PROMPT = prompts.BASELINE_DEFAULT_PROMPT


def _answer_question(question: str, qtype: str) -> str:
    """Beantworte question ohne Dokumenten-Kontext (typ-spezifischer Prompt)."""
    return chat_complete(
        messages=[
            {"role": "system", "content": _PROMPTS_BY_TYPE.get(qtype, _DEFAULT_PROMPT)},
            {"role": "user", "content": f"FRAGE:\n{question}"},
        ],
        model=config.MODEL_TEXT,
        max_tokens=_MAX_TOKENS,
        temperature=config.TEMPERATURE,
    )


def _process_item(item: Dict) -> Optional[Dict]:
    """Wandle ein Golden-Dataset-Item in einen Baseline-Antwort-Record um."""
    question = item.get("question")
    if not isinstance(question, str):
        return None
    answer = _answer_question(question, item.get("question_type", ""))
    return {
        "question": question,
        "ground_truth": item.get("answer"),
        "baseline_answer": answer,
        "question_type": item.get("question_type", ""),
        "source_reference": item.get("source_reference"),
        "requires_context": item.get("requires_context", True),
    }


def build_baseline_answers() -> None:
    """Beantworte alle Golden-Dataset-Fragen ohne Retrieval, parallel + resumable."""
    logging.info("=== EXPERIMENT-CONFIG (Schritt 3) ===")
    logging.info(f"Modell: {config.MODEL_TEXT} | T={config.TEMPERATURE} "
                 f"| Workers={config.MAX_WORKERS}")
    logging.info("=" * 40)

    dataset = load_json(config.FILE_GOLDEN_DATASET)
    results = load_checkpoint(_CHECKPOINT_FILE)
    done = {r["question"] for r in results}
    pending = [item for item in dataset
                if isinstance(item.get("question"), str) and item["question"] not in done]
    logging.info(f"Verarbeite {len(pending)} Fragen mit {config.MAX_WORKERS} Workern.")

    run_parallel_with_checkpoint(
        _process_item, pending, _CHECKPOINT_FILE, results,
        max_workers=config.MAX_WORKERS, label="question",
    )

    save_json(_OUTPUT_FILE, results)
    if _CHECKPOINT_FILE.exists():
        _CHECKPOINT_FILE.unlink()
    logging.info(f"Baseline-Inferenz fertig. Total: {len(results)}")


def main() -> None:
    """CLI-Einstieg für den alleinstehenden Baseline-Lauf."""
    parser = argparse.ArgumentParser(description="Baseline-Inferenz (kein RAG).")
    parser.parse_args()

    config.ensure_output_dirs()
    setup_logger(config.DIR_BASELINE / "3_baseline_pipeline.log")
    build_baseline_answers()


if __name__ == "__main__":
    main()
