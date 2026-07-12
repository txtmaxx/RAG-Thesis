"""Zentrale Projekt-Konfiguration.

Alle Modell-Namen, Pfade und Hyperparameter werden hier gebündelt, damit
Reproduzierbarkeit und Konfigurations-Audit über die Bachelorarbeit hinweg
sichergestellt sind. Werte können über Umgebungsvariablen überschrieben
werden, sodass die Experimente nicht-invasiv parametriert werden können.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Lies eine Umgebungsvariable als int, mit Fallback auf default."""
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    """Lies eine Umgebungsvariable als float, mit Fallback auf default."""
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw.strip() else default


# ─── API ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ─── Modelle ──────────────────────────────────────────────────────────────────
# Routing-Strategie: günstig fürs Schreiben, stark fürs Bewerten.
MODEL_TEXT: str = os.getenv("MODEL_TEXT", "gpt-4o-mini")
MODEL_TEXT_ADVANCED: str = os.getenv("MODEL_TEXT_ADVANCED", "gpt-4o")
MODEL_VISION: str = os.getenv("MODEL_VISION", "gpt-4o-mini")
MODEL_JUDGE: str = os.getenv("MODEL_JUDGE", "gpt-4o")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ─── LLM-Hyperparameter ───────────────────────────────────────────────────────
TEMPERATURE: float = _env_float("TEMPERATURE", 0.0)
TEMPERATURE_GENERATION: float = _env_float("TEMPERATURE_GENERATION", 0.3)
RANDOM_SEED: int = _env_int("RANDOM_SEED", 42)

# ─── PDF-Ingestion ────────────────────────────────────────────────────────────
START_PAGE: int = _env_int("START_PAGE", 1)
_END = os.getenv("END_PAGE", "")
END_PAGE: Optional[int] = int(_END) if _END.strip().isdigit() else None
MIN_IMAGE_SIZE: int = _env_int("MIN_IMAGE_SIZE", 100)
MAX_IMAGE_DIM: int = _env_int("MAX_IMAGE_DIM", 1024)
FOOTER_CUTOFF_RATIO: float = _env_float("FOOTER_CUTOFF_RATIO", 0.1)
HEADER_CUTOFF_RATIO: float = _env_float("HEADER_CUTOFF_RATIO", 0.0)
CHUNK_TARGET_SIZE: int = _env_int("CHUNK_TARGET_SIZE", 1200)
CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 150)
IMAGE_DETAIL: str = os.getenv("IMAGE_DETAIL", "high")

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K: int = _env_int("TOP_K", 8)

# ─── Ground-Truth-Generierung ─────────────────────────────────────────────────
WINDOW_SIZE: int = _env_int("WINDOW_SIZE", 7)
WINDOW_STRIDE: int = _env_int("WINDOW_STRIDE", 3)
QUESTION_TYPES: List[str] = ["Definition", "Anwendung", "Transfer"]

# ─── Parallelisierung ─────────────────────────────────────────────────────────
MAX_WORKERS: int = _env_int("MAX_WORKERS", 12)
MAX_WORKERS_JUDGE: int = _env_int("MAX_WORKERS_JUDGE", 5)

# ─── Rate-Limiting (clientseitig) ────────────────────────────────────────────
JUDGE_TPM_LIMIT: int = _env_int("JUDGE_TPM_LIMIT", 25000)
OPENAI_REQUEST_TIMEOUT: float = _env_float("OPENAI_REQUEST_TIMEOUT", 60.0)

# ─── Persistenz ───────────────────────────────────────────────────────────────
SAVE_RAW_RESPONSES: bool = os.getenv("SAVE_RAW_RESPONSES", "0").lower() in ("1", "true", "yes")

# ─── Pfade ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

PDF_INPUT_FILE: Path = DATA_DIR / os.getenv("PDF_FILENAME", "vorlesung.pdf")

DIR_ORCHESTRATOR: Path = OUTPUTS_DIR / "0_orchestrator"
DIR_INGESTION: Path = OUTPUTS_DIR / "1_ingestion"
DIR_GROUND_TRUTH: Path = OUTPUTS_DIR / "2_ground_truth"
DIR_BASELINE: Path = OUTPUTS_DIR / "3_baseline"
DIR_RAG: Path = OUTPUTS_DIR / "4_rag"
DIR_EVALUATION: Path = OUTPUTS_DIR / "5_evaluation"
DIR_ANALYSIS: Path = OUTPUTS_DIR / "6_analysis"

CHROMA_DIR: Path = DIR_INGESTION / "chroma_db"

FILE_INGESTION_SEMANTIC: Path = DIR_INGESTION / "1_pdf_ingestion_semantic.json"
FILE_INGESTION_RAW: Path = DIR_INGESTION / "1_pdf_ingestion_raw.json"
FILE_PDF_PAGES_RAW: Path = DIR_INGESTION / "1_pdf_pages_raw.json"
FILE_GOLDEN_DATASET: Path = DIR_GROUND_TRUTH / "2_golden_dataset.json"
FILE_BASELINE_ANSWERS: Path = DIR_BASELINE / "3_baseline_answers.json"
FILE_RAG_SEMANTIC: Path = DIR_RAG / "4_rag_answers_semantic.json"
FILE_RAG_RAW: Path = DIR_RAG / "4_rag_answers_raw.json"
FILE_EVALUATION: Path = DIR_EVALUATION / "5_evaluation_results.json"
FILE_MANUAL_REVIEW_CSV: Path = DIR_EVALUATION / "5_manual_review_sample.csv"


def ensure_output_dirs() -> None:
    """Lege alle Output-Verzeichnisse an."""
    for path in (
        DATA_DIR, OUTPUTS_DIR, DIR_INGESTION, DIR_GROUND_TRUTH,
        DIR_BASELINE, DIR_RAG, DIR_EVALUATION, DIR_ANALYSIS,
        DIR_ORCHESTRATOR, CHROMA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
