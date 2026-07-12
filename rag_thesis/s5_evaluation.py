"""Schritt 5 - Evaluation per LLM-as-a-Judge.

Methodische Designentscheidungen:

1. Proportionale Correctness (Primärmetrik für H1): Die Musterlösung wird
   in atomare GT-Aussagen zerlegt. Für jede prüft der Judge, ob sie in der
   Kandidatenantwort korrekt vorhanden ist. Score = belegte / gesamt. Vermeidet
   den Decken-Effekt der 1–5-Likert-Skala. Likert läuft parallel weiter, weil
   nur darüber die Inter-Rater-Reliabilität (Cohen's κ) berechnet werden kann.

2. Proportionale Faithfulness (Ragas-Stil): Die Antwort wird in atomare
   Aussagen zerlegt. Für jede Aussage prüft der Judge, ob sie aus dem Kontext
   ableitbar ist. Score = unterstützte Aussagen / Gesamt-Aussagen. Liefert
   einen kontinuierlichen 0…1-Wert ohne den Decken-Effekt der 1–5-Likert-Skala.

3. Faithfulness gegen angereicherten kanonischen Kontext: Der Vergleichs-
   kontext ist der unveränderte PDF-Text plus die Bildbeschreibungen des
   semantischen Ingestion-Laufs derselben Seite. Beide RAG-Varianten werden so
   gegen dieselbe Referenz gemessen, ohne den Semantic-Mode zu bestrafen,
   nur weil er Bildinformationen einbezieht.

4. Differenzierung Fakten vs. Zwischenschritte: Bei Anwendungsaufgaben
   sind rechnerische Zwischenschritte nicht wörtlich im Skript, folgen aber
   notwendig aus belegten Fakten. Der Verify-Prompt unterscheidet explizit
   zwischen faktischen Behauptungen (Kontextbindung Pflicht) und logisch
   abgeleiteten Rechenschritten (zulässig, wenn aus Kontext-Fakten folgend).

5. Inter-Rater-Reliability (IRR): Die Position-Swap-Doppelbewertung der
   Likert-Variante liefert beide Einzelscores + Differenz. 
   Skript 6 berichtet daraus die Judge-Reliabilität.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
from typing import Callable, Dict, List, Literal, Optional

from pydantic import BaseModel

from . import config, prompts
from .io_utils import (
    load_checkpoint, load_json, rel, save_json, setup_logger,
)
from .llm_client import structured_complete
from .parallel import run_parallel_with_checkpoint
from .text_utils import ensure_str, sanitize_latex


_CHECKPOINT_FILE = config.DIR_EVALUATION / "5_checkpoint.json"
_CSV_OUTPUT_FILE = config.FILE_MANUAL_REVIEW_CSV

_MAX_TOKENS_JUDGE = 1200
_MAX_TOKENS_DECOMPOSE = 1500
_MAX_TOKENS_VERIFY = 1500  # erweiterter Verify-Prompt + Frage machen den Output länger

_SOURCE_REF_RE = re.compile(r"(chunk_\d+)_to_(chunk_\d+)")
_PAGE_MARKER_RE = re.compile(r"### Seite (\d+)")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class JudgeScore(BaseModel):
    """1–5-Likert-Bewertung mit Begründung."""
    justification: str
    score: int


class ClaimList(BaseModel):
    """Atomare Aussagen einer Antwort (Ragas-Schritt 1)."""
    claims: List[str]


class ClaimVerdicts(BaseModel):
    """Pro Aussage: kontextuell belegt oder nicht (Ragas-Schritt 2).

    Das categories-Feld zwingt den Judge, die in _VERIFY_PROMPT
    vorgesehene Kategorisierung (A faktische Behauptung / B rechnerischer
    Zwischenschritt / C Setup-Wiederholung) strukturiert auszugeben statt nur
    intern in reasoning zu vermerken. So wird die Kategorie-Verteilung
    programmatisch auswertbar (vgl. s7_extract_categories).
    """
    verdicts: List[bool]
    categories: List[Literal["A", "B", "C"]]
    reasoning: str


class GroundTruthClaimVerdicts(BaseModel):
    """Pro Ground-Truth-Aussage: ob sie in der Kandidatenantwort korrekt wiedergegeben ist."""
    verdicts: List[bool]
    reasoning: str


# ─── Correctness (Likert, position-swapped, IRR-fähig) ────────────────────────

# Prompt-Texte zentral in prompts.py (Prompt Engineering an einer Stelle).
_CORRECTNESS_PROMPT = prompts.CORRECTNESS_PROMPT


def _judge_likert(system_prompt: str, user_content: str) -> Optional[JudgeScore]:
    """Einzel-Call an den Likert-Judge (1–5 + Begründung)."""
    completion = structured_complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        model=config.MODEL_JUDGE,
        response_format=JudgeScore,
        max_tokens=_MAX_TOKENS_JUDGE,
        temperature=config.TEMPERATURE,
    )
    return completion.choices[0].message.parsed


def evaluate_correctness(question: str, ground_truth: str, answer: str) -> Dict:
    """Likert 1–5 mit Position-Swap, normalisiert auf 0…1.

    Beide Einzelscores + Differenz werden gespeichert, damit Schritt 6 die
    Inter-Rater-Reliability des Judges berichten kann.

    Sequenzielle Ausführung (nicht parallel): Die TPM-Last der zwei Swap-Calls
    summiert sich auf demselben Worker. Parallelität würde das gpt-4o-Limit
    zu schnell reißen. Die äußere run_parallel_with_checkpoint-Schleife
    sorgt für die Wallclock-Beschleunigung.
    """
    q, gt, ans = ensure_str(question), ensure_str(ground_truth), ensure_str(answer)

    r1 = _judge_likert(_CORRECTNESS_PROMPT,
                        f"FRAGE:\n{q}\n\nGROUND TRUTH:\n{gt}\n\nANTWORT:\n{ans}")
    r2 = _judge_likert(_CORRECTNESS_PROMPT,
                        f"FRAGE:\n{q}\n\nANTWORT:\n{ans}\n\nGROUND TRUTH:\n{gt}")

    valid = [r for r in (r1, r2) if r is not None]
    if not valid:
        return {"score": None, "raw_scores": [], "score_disagreement": None,
                "justifications": [], "justification": "Evaluation fehlgeschlagen"}

    raw_scores = [r.score for r in valid]
    avg_norm = sum((s - 1) / 4 for s in raw_scores) / len(raw_scores)
    disagreement = abs(raw_scores[0] - raw_scores[1]) if len(raw_scores) == 2 else 0
    justifications = [sanitize_latex(r.justification) for r in valid]
    return {
        "score": avg_norm,
        "raw_score_avg": sum(raw_scores) / len(raw_scores),
        "raw_scores": raw_scores,
        "score_disagreement": disagreement,
        "justifications": justifications,
        "justification": justifications[0],
    }


# ─── Correctness (proportional, claim-based - primärer H1-Score) ─────────────
# Zerlegt die Ground Truth in atomare Fakten und zählt die belegten > 0…1 ohne
# Decken-Effekt (anders als Likert). Likert bleibt parallel, da nur dessen
# Position-Swap das Cohen's κ liefert (s6_analysis.compute_irr).

_GT_DECOMPOSE_PROMPT = prompts.GT_DECOMPOSE_PROMPT


_GT_VERIFY_PROMPT = prompts.GT_VERIFY_PROMPT


def _decompose_ground_truth(ground_truth: str) -> List[str]:
    """Zerlege die Musterlösung in atomare GT-Aussagen (analog Claim-Decomposition)."""
    completion = structured_complete(
        messages=[
            {"role": "system", "content": _GT_DECOMPOSE_PROMPT},
            {"role": "user", "content": f"MUSTERLÖSUNG:\n{ensure_str(ground_truth)}"},
        ],
        model=config.MODEL_JUDGE,
        response_format=ClaimList,
        max_tokens=_MAX_TOKENS_DECOMPOSE,
        temperature=config.TEMPERATURE,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return []
    return [c.strip() for c in parsed.claims if c and c.strip()]


def _verify_gt_claims_in_answer(gt_claims: List[str], answer: str,
                                  question: str = "") -> List[bool]:
    """Prüfe für jede GT-Aussage, ob sie in answer (explizit oder logisch) vorkommt."""
    if not gt_claims:
        return []
    formatted = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(gt_claims))
    q_block = f"AUFGABENSTELLUNG:\n{ensure_str(question)}\n\n" if question else ""
    user_content = (
        f"{q_block}KANDIDATEN-ANTWORT:\n{ensure_str(answer)}\n\n"
        f"GT-AUSSAGEN ({len(gt_claims)} Stück):\n{formatted}"
    )
    completion = structured_complete(
        messages=[
            {"role": "system", "content": _GT_VERIFY_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model=config.MODEL_JUDGE,
        response_format=GroundTruthClaimVerdicts,
        max_tokens=_MAX_TOKENS_VERIFY,
        temperature=config.TEMPERATURE,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return []
    verdicts = list(parsed.verdicts)
    # Längen-Robustheit (analog Faithfulness-Verifier).
    if len(verdicts) < len(gt_claims):
        verdicts.extend([False] * (len(gt_claims) - len(verdicts)))
    elif len(verdicts) > len(gt_claims):
        verdicts = verdicts[:len(gt_claims)]
    return verdicts


def evaluate_correctness_proportional(question: str, ground_truth: str,
                                       answer: str) -> Dict:
    """Proportionale Correctness: Anteil der in der Antwort belegten GT-Aussagen.

    Zwei-Pass-Verfahren analog zur Faithfulness:
    1. _decompose_ground_truth zerlegt die Musterlösung in N atomare Aussagen.
    2. _verify_gt_claims_in_answer prüft jede gegen die Kandidatenantwort.

    Score = n_supported / n_total. Liefert None (mit Begründung), wenn die
    Musterlösung keine extrahierbaren Aussagen enthält oder ein API-Call scheitert.

    Hinweis zur Position-Swap-Validierung
    -------------------------------------
    Diese proportionale Primärmetrik liefert pro Antwort einen einzelnen
    Skalar. Eine Position-Swap-Doppelbewertung (wie bei der 1-5-Likert-
    Variante in evaluate_correctness) ist hier nicht definiert. Die
    Inter-Rater-Reliabilität (Cohen's κ) wird deshalb auf der
    Likert-Variante berechnet und dient nur als Stabilitäts-Indikator des
    Judges. Der Validitätsanker für die Primärmetrik ist die
    Spearman-Korrelation gegen eine manuelle Stichprobe
    """
    try:
        gt_claims = _decompose_ground_truth(ground_truth)
    except Exception as e:
        logging.error(f"GT-Decomposition fehlgeschlagen: {e}")
        return {"score": None, "n_claims": 0, "n_supported": 0,
                "claims": [], "verdicts": [],
                "justification": "GT-Decomposition fehlgeschlagen"}

    if not gt_claims:
        return {"score": None, "n_claims": 0, "n_supported": 0,
                "claims": [], "verdicts": [],
                "justification": "Keine GT-Aussagen extrahierbar"}

    try:
        verdicts = _verify_gt_claims_in_answer(gt_claims, answer, question)
    except Exception as e:
        logging.error(f"GT-Verification fehlgeschlagen: {e}")
        return {"score": None, "n_claims": len(gt_claims), "n_supported": 0,
                "claims": gt_claims, "verdicts": [],
                "justification": "GT-Verification fehlgeschlagen"}

    n_supported = sum(1 for v in verdicts if v)
    score = n_supported / len(gt_claims)
    missing = [c for c, v in zip(gt_claims, verdicts) if not v]
    justification = (
        f"{n_supported} von {len(gt_claims)} GT-Aussagen in der Antwort belegt. "
        f"Fehlende Aussagen: {len(missing)}."
    )
    return {
        "score": score,
        "n_claims": len(gt_claims),
        "n_supported": n_supported,
        "claims": gt_claims,
        "verdicts": verdicts,
        "justification": justification,
    }


# ─── Faithfulness (Ragas-Stil, proportional) ──────────────────────────────────

_DECOMPOSE_PROMPT = prompts.DECOMPOSE_PROMPT


def _decompose_into_claims(answer: str) -> List[str]:
    """Zerlege answer in eine Liste atomarer faktischer Aussagen (Ragas-Schritt 1)."""
    completion = structured_complete(
        messages=[
            {"role": "system", "content": _DECOMPOSE_PROMPT},
            {"role": "user", "content": f"ANTWORT:\n{ensure_str(answer)}"},
        ],
        model=config.MODEL_JUDGE,
        response_format=ClaimList,
        max_tokens=_MAX_TOKENS_DECOMPOSE,
        temperature=config.TEMPERATURE,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return []
    return [c.strip() for c in parsed.claims if c and c.strip()]


_VERIFY_PROMPT = prompts.VERIFY_PROMPT


def _verify_claims_against_context(claims: List[str], context: str,
                                    question: str = "") -> tuple:
    """Prüfe jede Aussage gegen Skript-Kontext + Aufgabenstellung (Ragas-Schritt 2).

    Padded oder kürzt sowohl Verdicts als auch Categories auf len(claims),
    damit der Caller pro Claim genau einen Bool und einen Kategorie-Buchstaben
    erhält, selbst wenn der Judge eine abweichende Liste liefert. Fehlende
    Kategorien werden mit "A" aufgefüllt (konservative Default-Annahme:
    faktische Behauptung, der strengste Bewertungsmaßstab).

    Returns
    -------
    tuple[List[bool], List[str]]
        (verdicts, categories) jeweils der Länge len(claims).
    """
    if not claims:
        return [], []
    formatted = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    # Frage mitliefern, damit der Judge Setup-Wiederholungen (Kat. C) nicht als
    # Halluzination wertet (siehe _VERIFY_PROMPT).
    q_block = f"AUFGABENSTELLUNG:\n{ensure_str(question)}\n\n" if question else ""
    user_content = (
        f"{q_block}SKRIPT-KONTEXT:\n{ensure_str(context)}\n\n"
        f"AUSSAGEN ({len(claims)} Stück):\n{formatted}"
    )
    completion = structured_complete(
        messages=[
            {"role": "system", "content": _VERIFY_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model=config.MODEL_JUDGE,
        response_format=ClaimVerdicts,
        max_tokens=_MAX_TOKENS_VERIFY,
        temperature=config.TEMPERATURE,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return [], []
    verdicts = list(parsed.verdicts)
    categories = list(getattr(parsed, "categories", []) or [])
    n = len(claims)
    # Verdicts auf n bringen.
    if len(verdicts) < n:
        verdicts.extend([False] * (n - len(verdicts)))
    elif len(verdicts) > n:
        verdicts = verdicts[:n]
    # Categories analog auf n bringen, Default "A" (strengste Bewertung).
    if len(categories) < n:
        categories.extend(["A"] * (n - len(categories)))
    elif len(categories) > n:
        categories = categories[:n]
    return verdicts, categories


def evaluate_faithfulness_proportional(answer: str, canonical_context: str,
                                        question: str = "") -> Dict:
    """Ragas-Style proportionale Faithfulness gegen den kanonischen Quelltext.

    question ist optional, sollte aber für Transfer-/Anwendungsaufgaben gesetzt
    werden: Der Verify-Prompt akzeptiert wörtliche/paraphrasierte Wiederholungen
    des Aufgaben-Setups als kontextuell belegt (Kategorie C im Prompt). Ohne diesen
    Hinweis bewertete der Judge Aufgaben-Setups als unbelegt und bestrafte damit
    systematisch Transfer-/Anwendungsfragen, deren Inputs in der Frage selbst definiert sind.
    """
    try:
        claims = _decompose_into_claims(answer)
    except Exception as e:
        logging.error(f"Claim-Decomposition fehlgeschlagen: {e}")
        return {"score": None, "n_claims": 0, "n_supported": 0,
                "claims": [], "verdicts": [], "justification": "Decomposition fehlgeschlagen"}

    if not claims:
        return {"score": None, "n_claims": 0, "n_supported": 0,
                "claims": [], "verdicts": [], "justification": "Keine Aussagen extrahierbar"}

    try:
        verdicts, categories = _verify_claims_against_context(
            claims, canonical_context, question)
    except Exception as e:
        logging.error(f"Claim-Verification fehlgeschlagen: {e}")
        return {"score": None, "n_claims": len(claims), "n_supported": 0,
                "claims": claims, "verdicts": [], "categories": [],
                "justification": "Verification fehlgeschlagen"}

    n_supported = sum(1 for v in verdicts if v)
    score = n_supported / len(claims) if claims else None
    unsupported = [c for c, v in zip(claims, verdicts) if not v]
    # Verteilung der Claim-Kategorien (A/B/C) mitloggen für die Bias-Auswertung.
    cat_counts = {k: categories.count(k) for k in ("A", "B", "C")}
    justification = (
        f"{n_supported} von {len(claims)} Aussagen kontextuell belegt. "
        f"Nicht belegte Aussagen: {len(unsupported)}. "
        f"Kategorien (A/B/C): {cat_counts['A']}/{cat_counts['B']}/{cat_counts['C']}."
    )
    return {
        "score": score,
        "n_claims": len(claims),
        "n_supported": n_supported,
        "claims": claims,
        "verdicts": verdicts,
        "categories": categories,
        "category_counts": cat_counts,
        "justification": justification,
    }


# ─── Kanonischer Kontext (PDF-Original) ───────────────────────────────────────

CanonicalLookup = Callable[[str], str]


def _build_canonical_context_lookup() -> Optional[CanonicalLookup]:
    """Erzeuge einen Look-up source_reference -> kanonischer Vergleichs-Text.

    Der Vergleichs-Text pro Seite ist der unveränderte PDF-Text plus die
    Bildbeschreibungen aus dem semantischen Ingestion-Lauf, damit beide
    RAG-Modi gegen dieselbe Informationsbasis gemessen werden (siehe Modul-
    Docstring, Punkt 2).

    Returns None, falls die Eingabedaten fehlen. Der Aufrufer entscheidet
    dann, ob ein Fallback auf den retrieved_context angemessen ist.
    """
    if not config.FILE_PDF_PAGES_RAW.exists() or not config.FILE_INGESTION_SEMANTIC.exists():
        logging.warning(
            "Kanonische Kontext-Daten fehlen (1_pdf_pages_raw.json bzw. "
            "1_pdf_ingestion_semantic.json). Faithfulness wird auf den "
            "retrieved_context zurückfallen - methodisch suboptimal."
        )
        return None

    pages_data = load_json(config.FILE_PDF_PAGES_RAW)
    pages: Dict[int, str] = {}
    n_with_descs = 0
    for entry in pages_data:
        page_num = int(entry["page_number"])
        text = str(entry.get("text", ""))
        descs = entry.get("image_descriptions") or []
        if descs:
            n_with_descs += 1
            descs_block = "\n\n".join(f"[BILD-INFO: {d}]" for d in descs if d)
            text = f"{text}\n\n{descs_block}" if text else descs_block
        pages[page_num] = text

    logging.info(
        f"Kanonischer Kontext: {len(pages)} Seiten geladen, "
        f"davon {n_with_descs} mit Bildbeschreibungen angereichert."
    )

    chunks = load_json(config.FILE_INGESTION_SEMANTIC)
    chunk_idx: Dict[str, Dict] = {c["chunk_id"]: c for c in chunks if "chunk_id" in c}

    def page_of(chunk_id: str) -> Optional[int]:
        c = chunk_idx.get(chunk_id, {})
        if "page_number" in c:
            return int(c["page_number"])
        m = _PAGE_MARKER_RE.search(c.get("content", ""))
        return int(m.group(1)) if m else None

    return _LookupBuilder(pages, page_of).build


class _LookupBuilder:
    """Lazy Look-up des kanonischen Kontexts pro source_reference."""
    def __init__(self, pages: Dict[int, str], page_of: Callable[[str], Optional[int]]):
        self._pages = pages
        self._page_of = page_of

    def build(self, source_reference: str) -> str:
        m = _SOURCE_REF_RE.match(source_reference or "")
        if not m:
            return ""
        start_page = self._page_of(m.group(1))
        if start_page is None:
            return ""
        end_page = self._page_of(m.group(2)) or start_page
        # ±1 Seite Puffer, da Chunks über Seitengrenzen reichen können.
        s = max(1, start_page - 1)
        e = end_page + 1
        return "\n\n".join(self._pages[p] for p in range(s, e + 1) if p in self._pages)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def _process_pair(item: Dict, canonical_context_for: Optional[CanonicalLookup]) -> Dict:
    """Evaluiere Baseline + beide RAG-Antworten eines aligned Items vollständig.

    Berechnet Correctness (alle drei Systeme) und Faithfulness (beide RAG-Modi)
    in einem Schritt. Das Flag canonical_context_used macht den Fallback
    auf retrieved_context in der späteren Analyse stratifizierbar.
    """
    canonical = canonical_context_for(item["source_reference"]) if canonical_context_for else ""
    canonical_used = bool(canonical)
    # Fallback ohne kanonischen Kontext: jeder Mode misst gegen seinen eigenen
    # retrieved_context (schwächer für H3), über canonical_context_used markiert.
    sem_reference = canonical or item.get("rag_semantic_context", "")
    raw_reference = canonical or item.get("rag_raw_context", "")
    if not canonical_used:
        logging.warning(
            f"Kein kanonischer Kontext für source_reference="
            f"{item.get('source_reference', '?')} - Fallback auf retrieved_context."
        )

    # Primäre Correctness-Metrik (proportional, claim-based).
    base_corr = evaluate_correctness_proportional(
        item["question"], item["ground_truth"], item["baseline_answer"])
    sem_corr = evaluate_correctness_proportional(
        item["question"], item["ground_truth"], item["rag_semantic_answer"])
    raw_corr = evaluate_correctness_proportional(
        item["question"], item["ground_truth"], item["rag_raw_answer"])

    # Sekundär: Likert mit Position-Swap liefert das Cohen's κ.
    base_corr_likert = evaluate_correctness(
        item["question"], item["ground_truth"], item["baseline_answer"])
    sem_corr_likert = evaluate_correctness(
        item["question"], item["ground_truth"], item["rag_semantic_answer"])
    raw_corr_likert = evaluate_correctness(
        item["question"], item["ground_truth"], item["rag_raw_answer"])

    sem_faith = evaluate_faithfulness_proportional(
        item["rag_semantic_answer"], sem_reference, item["question"])
    raw_faith = evaluate_faithfulness_proportional(
        item["rag_raw_answer"], raw_reference, item["question"])

    return {
        "question": item["question"],
        "question_type": item["question_type"],
        "source_reference": item["source_reference"],
        "requires_context": item.get("requires_context", True),
        "ground_truth": ensure_str(item["ground_truth"]),
        "canonical_context_used": canonical_used,
        "baseline": {
            "answer": ensure_str(item["baseline_answer"]),
            "correctness": base_corr,
            "correctness_likert": base_corr_likert,
        },
        "rag_semantic": {
            "answer": ensure_str(item["rag_semantic_answer"]),
            "correctness": sem_corr,
            "correctness_likert": sem_corr_likert,
            "faithfulness": sem_faith,
            "retrieved_context": ensure_str(item["rag_semantic_context"]),
        },
        "rag_raw": {
            "answer": ensure_str(item["rag_raw_answer"]),
            "correctness": raw_corr,
            "correctness_likert": raw_corr_likert,
            "faithfulness": raw_faith,
            "retrieved_context": ensure_str(item["rag_raw_context"]),
        },
    }


def _align_datasets(baseline: List[Dict], rag_sem: List[Dict],
                    rag_raw: List[Dict]) -> List[Dict]:
    """Joine die drei Inferenz-Outputs über die Frage zu evaluierbaren Tripeln.

    Items, die in einem der drei Datensätze fehlen, werden ohne Warnung
    übersprungen. Die paarweisen Hypothesentests in Schritt 6 sind sonst nicht definiert.
    """
    s_map = {item["question"]: item for item in rag_sem}
    r_map = {item["question"]: item for item in rag_raw}
    return [
        {
            "question": b["question"],
            "question_type": b.get("question_type"),
            "source_reference": b.get("source_reference"),
            "requires_context": b.get("requires_context", True),
            "ground_truth": b.get("ground_truth", ""),
            "baseline_answer": b.get("baseline_answer", ""),
            "rag_semantic_answer": s_map[b["question"]].get("rag_answer", ""),
            "rag_semantic_context": s_map[b["question"]].get("retrieved_context", ""),
            "rag_raw_answer": r_map[b["question"]].get("rag_answer", ""),
            "rag_raw_context": r_map[b["question"]].get("retrieved_context", ""),
        }
        for b in baseline
        if b["question"] in s_map and b["question"] in r_map
    ]


def export_manual_review_sample(
    results: List[Dict], *,
    sample_ratio: float = 0.3,
    min_items: int = 15,
) -> None:
    """Exportiere eine zufällige Stichprobe als CSV für die manuelle Validierung
    des LLM-Judges (siehe Schritt 6, Korrelation Human↔LLM).

    Größe: max(min_items, ceil(N · sample_ratio)), gedeckelt auf N.
    Hintergrund: n<15 liefert für die Spearman-Korrelation ein so weites CI
    (±0.6 bei n=5), dass die Judge-Validierung methodisch wertlos wird -
    die manuelle Stichprobe darf den Rest der Auswertung nicht entwerten.
    """
    if not results:
        return
    random.seed(config.RANDOM_SEED)
    target = max(min_items, int(len(results) * sample_ratio + 0.999))
    n = min(target, len(results))
    sample = random.sample(results, n)
    with open(_CSV_OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow([
            "Question", "Type",
            "Ground Truth",
            "Baseline Answer", "Baseline Correctness (LLM)",
            "RAG Semantic Answer", "RAG Sem Correctness (LLM)", "RAG Sem Faithfulness (LLM)",
            "RAG Raw Answer", "RAG Raw Correctness (LLM)", "RAG Raw Faithfulness (LLM)",
            # Spalten für die manuelle Eintragung (alle Werte 0.0–1.0):
            "Human Baseline Correctness", "Human Sem Correctness", "Human Sem Faithfulness",
            "Human Raw Correctness", "Human Raw Faithfulness",
        ])
        for item in sample:
            writer.writerow([
                item["question"], item["question_type"], item["ground_truth"],
                item["baseline"]["answer"],
                item["baseline"]["correctness"]["score"],
                item["rag_semantic"]["answer"],
                item["rag_semantic"]["correctness"]["score"],
                item["rag_semantic"]["faithfulness"]["score"],
                item["rag_raw"]["answer"],
                item["rag_raw"]["correctness"]["score"],
                item["rag_raw"]["faithfulness"]["score"],
                "", "", "", "", "",
            ])
    logging.info(f"Manual-Review-Sample geschrieben ({n} Items): {rel(_CSV_OUTPUT_FILE)}")


def run_evaluation() -> None:
    """Führe die vollständige LLM-as-a-Judge-Evaluation aus (resumable, parallel)."""
    logging.info("=== EXPERIMENT-CONFIG (Schritt 5 - Evaluation) ===")
    logging.info(f"Judge: {config.MODEL_JUDGE} | T={config.TEMPERATURE} "
                 f"| Workers={config.MAX_WORKERS_JUDGE}")
    logging.info("Correctness (primär): proportional (Anteil belegter GT-Aussagen)")
    logging.info("Correctness (sekundär, für IRR): 1–5 Likert, position-swapped -> "
                 "rohe Scores für Cohen's κ")
    logging.info("Faithfulness: Ragas-style proportional (claims supported / total), "
                 "Referenz: kanonischer PDF-Text (Fallback: retrieved_context)")
    logging.info("=" * 50)

    pairs = _align_datasets(
        load_json(config.FILE_BASELINE_ANSWERS),
        load_json(config.FILE_RAG_SEMANTIC),
        load_json(config.FILE_RAG_RAW),
    )

    canonical_for: Optional[CanonicalLookup] = _build_canonical_context_lookup()

    results = load_checkpoint(_CHECKPOINT_FILE)
    done = {r["question"] for r in results}
    pending = [p for p in pairs if p["question"] not in done]
    logging.info(f"Evaluiere {len(pending)} Paare mit {config.MAX_WORKERS_JUDGE} Workern.")

    run_parallel_with_checkpoint(
        lambda item: _process_pair(item, canonical_for),
        pending, _CHECKPOINT_FILE, results,
        max_workers=config.MAX_WORKERS_JUDGE, label="pair",
    )

    save_json(config.FILE_EVALUATION, results)
    if _CHECKPOINT_FILE.exists():
        _CHECKPOINT_FILE.unlink()
    export_manual_review_sample(results)
    logging.info(f"Evaluation fertig. Total: {len(results)}")


def main() -> None:
    """CLI-Einstieg für die alleinstehende Evaluation."""
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge-Evaluation.")
    parser.parse_args()

    config.ensure_output_dirs()
    setup_logger(config.DIR_EVALUATION / "5_evaluation.log")
    run_evaluation()


if __name__ == "__main__":
    main()
