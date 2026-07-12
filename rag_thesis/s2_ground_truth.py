"""Schritt 2 - Ground-Truth-Generierung.

Erzeugt Prüfungsfragen + Musterlösungen aus den semantischen Chunks. Für
Hypothese H1 wird optional gefiltert (requires_context), sodass nur
Fragen behalten werden, die ohne den Kontext nicht aus Allgemeinwissen
beantwortbar sind. Beide Werte (True/False) werden gespeichert,
damit in Schritt 6 eine Sensitivitätsanalyse möglich ist.

Der Filterungs-Marker bleibt bewusst im Datensatz. Nichts wird 
stillschweigend verworfen, sondern als requires_context-Feld weitergegeben. 
So ist exakt nachvollziehbar, was selektiert wurde.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re as _re
from typing import Dict, List, Optional


from pydantic import BaseModel

from . import config, prompts
from .io_utils import (
    load_checkpoint, load_json, save_checkpoint, save_json, setup_logger,
)
from .llm_client import chat_complete, structured_complete
from .text_utils import normalize_whitespace, sanitize_latex


_OUTPUT_FILE = config.FILE_GOLDEN_DATASET
_CHECKPOINT_FILE = config.DIR_GROUND_TRUTH / "2_checkpoint.json"
_MAX_TOKENS = 2500


# ─── Skript-Referenz-Filter ─────────────────────────────────────────────────
# Verwirft vor dem teuren LLM-Verifier Fragen, die aufs Skript verweisen. Solche
# Meta-Verweise sind in einer Klausur ohne Skript unfair gegenüber der Baseline.
# Stufe A: explizite Skript-/Vorlesungs-/Seitenverweise.
_FORBIDDEN_SCRIPT_REFS_RE = _re.compile(
    r"\b("
    r"(?:im|laut|gemäß|nach|aus dem|im bereitgestellten) Skript"
    r"|(?:in|laut|gemäß|nach) der Vorlesung"
    r"|(?:auf|gemäß|laut) Seite\s*\d+"
    r"|im (?:Quelltext|Quellmaterial|Originaltext|bereitgestellten Text)"
    r"|wie im Skript"
    r"|des Skripts?"
    r"|Skript-(?:spezifisch|spezifische[rnms]?|Notation|Methode|Konvention|Inhalt|Vorlage)"
    r")\b",
    flags=_re.IGNORECASE,
)

# Stufe B: implizite Verweise auf nicht mitgeliefertes Material ("anhand der
# gegebenen Tabelle"). Löst nur Resampling aus, daher sind seltene False Positives
# unkritisch. Materialwörter inkl. deutscher Komposita (Wahrheits-/Übergangstabelle …).
# 'Reihenfolge' bewusst ausgeklammert, da es in der Frage selbst stehen kann.
_MAT = (
    r"(?:[A-Za-zÄÖÜäöüß]+-?)?"   # optionales Präfix-Kompositum
    r"(?:Tabelle|Formel|Abbildung|Diagramm|Analyse|Methode|Notation|"
    r"Schaltung|Darstellung|Grafik|Übersicht|Beschreibung)"
)
_IMPLICIT_MATERIAL_REFS_RE = _re.compile(
    r"\b(?:"
    # "in der (gegebenen|angegebenen|dargestellten|gezeigten) Tabelle/Formel/…"
    r"(?:in|laut|gemäß|nach|anhand|mit|aus) der (?:gegebenen |angegebenen |"
    r"dargestellten |gezeigten |verwendeten |beschriebenen |spezifizierten |"
    r"oben(?:stehenden|genannten)? |unten(?:stehenden|genannten)? )?"
    + _MAT +
    r"|"
    # "wie sie in der Tabelle dargestellt ist"
    r"wie (?:sie |er |es |dies )?(?:in der|im) " + _MAT +
    r"|"
    # "in der Form, die in der Analyse … verwendet wird"
    r"in der Form,? die in der (?:Analyse|Tabelle|Beschreibung)"
    r")\b",
    flags=_re.IGNORECASE,
)


def _find_script_references(question: str) -> List[str]:
    """Liefere alle Meta-Verweise auf nicht-mitgeliefertes Material.

    Vereinigt Stufe A (explizite Skript/Vorlesungs-Erwähnungen) und Stufe B
    (implizite Tabellen-/Formel-/Abbildungs-Verweise). Deterministisch und
    seiteneffektfrei. Eingabe nur die Fragestellung, nicht die Antwort.
    """
    if not question:
        return []
    return (
        [m.group(0) for m in _FORBIDDEN_SCRIPT_REFS_RE.finditer(question)]
        + [m.group(0) for m in _IMPLICIT_MATERIAL_REFS_RE.finditer(question)]
    )


class QAItem(BaseModel):
    question: str
    answer: str
    requires_context: bool


class VerifyVerdict(BaseModel):
    passes: bool
    issues: List[str]


# Prompt-Texte zentral in prompts.py. Aliase erhalten Lesbarkeit/Kompatibilität.
_TYPE_INSTRUCTIONS: Dict[str, str] = prompts.GROUND_TRUTH_TYPE_INSTRUCTIONS


def _truncate_at_topic_break(block: List[Dict]) -> List[Dict]:
    """Schneide block vor dem ersten neuen H1/H2-Heading ab.

    Verhindert, dass eine Generation-Window über einen Themenwechsel hinweg
    Kontext zusammenzieht. Sonst entstehen Fragen, deren Antwort über zwei
    fachlich unverbundene Bereiche springt.
    """
    import re
    result = [block[0]]
    for chunk in block[1:]:
        if re.match(r"^#{1,2}\s", chunk.get("content", "").lstrip()):
            break
        result.append(chunk)
    return result


def _build_windows(chunks: List[Dict]) -> List[Dict]:
    """Erzeuge gleitende Fenster über die Chunks (Größe/Stride aus config)."""
    windows: List[Dict] = []
    for i in range(0, len(chunks) - config.WINDOW_SIZE + 1, config.WINDOW_STRIDE):
        block = _truncate_at_topic_break(chunks[i:i + config.WINDOW_SIZE])
        if len(block) < 2:
            continue
        start_id = block[0].get("chunk_id", f"chunk_{i}")
        end_id = block[-1].get("chunk_id", f"chunk_{i + len(block) - 1}")
        windows.append({
            "text": normalize_whitespace("\n\n".join(p.get("content", "") for p in block)),
            "source_reference": f"{start_id}_to_{end_id}",
        })
    return windows


def _heuristic_valid(text: str) -> bool:
    """Billige Vorab-Filter gegen offensichtlich untaugliche Windows.

    Lehnt zu kurze Fenster (<500 Zeichen) und Fenster mit zu vielen langen
    Zahlen ab (typisch für Inhaltsverzeichnis/Seitenzahlblöcke). Spart Calls
    am teuren Relevance-Check und am Generator.
    """
    import re
    return len(text) >= 500 and len(re.findall(r"\d{4,}", text)) <= 25


def _relevance_check(text: str, cache: Dict[str, bool]) -> bool:
    """LLM-Vorfilter: enthält das Fenster prüfungsrelevanten Informatik-Inhalt?"""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key]
    system_prompt = prompts.RELEVANCE_CHECK_SYSTEM
    resp = chat_complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text[:1500]},
        ],
        model=config.MODEL_TEXT,
        max_tokens=10,
        temperature=config.TEMPERATURE,
    )
    result = bool(resp and "TRUE" in resp)
    cache[cache_key] = result
    return result


def _generate_qa(text: str, source_reference: str, qtype: str) -> Optional[Dict]:
    """Generiere genau EIN Q/A-Paar zum gegebenen Quelltext und Fragetyp.

    Nutzt das stärkere Modell (MODEL_TEXT_ADVANCED), weil die Qualität
    der Ground Truth alle nachgelagerten Hypothesentests dominiert.
    """
    system_prompt = prompts.ground_truth_generation_system(qtype)
    completion = structured_complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model=config.MODEL_TEXT_ADVANCED,
        response_format=QAItem,
        max_tokens=_MAX_TOKENS,
        temperature=config.TEMPERATURE_GENERATION,
    )
    item = completion.choices[0].message.parsed
    if item is None:
        raise RuntimeError(
            f"Structured Output lieferte None: {completion.choices[0].message.refusal}"
        )
    return {
        "question": sanitize_latex(item.question),
        "answer": sanitize_latex(item.answer),
        "source_reference": source_reference,
        "question_type": qtype,
        "requires_context": bool(item.requires_context),
    }


def _verify_qa(qa: Dict, source_text: str) -> VerifyVerdict:
    """Zweiter LLM-Pass: prüft den Q/A-Block auf interne und Quell-Konsistenz
    (Binär-Dezimal-Mismatch, falsche KV-Gruppen, Modulo-Inkonsistenz,
    Zustände↔Flipflops, fabrizierte Strukturen).

    Erfolgreich, wenn passes=True und issues=[]. Andernfalls liefert der
    Verdict eine begründete Fehlerliste, die der Caller zum Resampling nutzen kann.
    """
    system_prompt = prompts.GROUND_TRUTH_VERIFY_SYSTEM
    user_content = (
        f"=== QUELLTEXT (chunked) ===\n{source_text[:6000]}\n\n"
        f"=== FRAGE ===\n{qa['question']}\n\n"
        f"=== MUSTERLÖSUNG ===\n{qa['answer']}"
    )
    completion = structured_complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        model=config.MODEL_TEXT_ADVANCED,
        response_format=VerifyVerdict,
        max_tokens=500,
        temperature=config.TEMPERATURE,  # 0.0 für deterministische Prüfung
    )
    verdict = completion.choices[0].message.parsed
    if verdict is None:
        # Verifier-Refusal soll Generierung NICHT blockieren, markieren und durchlassen
        return VerifyVerdict(passes=True, issues=["verifier_refusal"])
    return verdict


def build_dataset(target_questions: int, *, keep_general: bool = False) -> None:
    """Erzeuge target_questions viele QA-Items, balanciert über die Typen.

    Parameters
    ----------
    keep_general
        Wenn True, werden auch Fragen mit requires_context=False behalten,
        nötig für die Sensitivitätsanalyse in Schritt 6.
    """
    logging.info("=== EXPERIMENT-CONFIG (Schritt 2) ===")
    logging.info(f"Generierung: {config.MODEL_TEXT_ADVANCED} "
                 f"| Relevance-Check: {config.MODEL_TEXT} "
                 f"| T_gen={config.TEMPERATURE_GENERATION} | Seed={config.RANDOM_SEED}")
    logging.info(f"Fragetypen: {config.QUESTION_TYPES} | Window: "
                 f"{config.WINDOW_SIZE}/{config.WINDOW_STRIDE}")
    logging.info(f"Ziel: {target_questions} Fragen | keep_general={keep_general}")
    logging.info("=" * 40)

    random.seed(config.RANDOM_SEED)
    chunks = load_json(config.FILE_INGESTION_SEMANTIC)
    windows = _build_windows(chunks)
    random.shuffle(windows)

    dataset = load_checkpoint(_CHECKPOINT_FILE)
    if len(dataset) >= target_questions:
        logging.info("Ziel bereits aus Checkpoint erreicht.")
        save_json(_OUTPUT_FILE, dataset)
        return

    target_per_type = target_questions // len(config.QUESTION_TYPES)
    type_counts: Dict[str, int] = {qt: 0 for qt in config.QUESTION_TYPES}
    for item in dataset:
        if item.get("requires_context", True) is False and not keep_general:
            continue
        qt = item.get("question_type", "")
        if qt in type_counts:
            type_counts[qt] += 1

    relevance_cache: Dict[str, bool] = {}

    for idx, w in enumerate(windows):
        if len(dataset) >= target_questions:
            break
        text, source_reference = w["text"], w["source_reference"]
        if not _heuristic_valid(text) or not _relevance_check(text, relevance_cache):
            continue
        available = [qt for qt, c in type_counts.items() if c < target_per_type] \
                     or config.QUESTION_TYPES
        qtype = random.choice(available)
        try:
            # Generieren + bis zu 2 Resamples bei negativer Verifikation
            qa = None
            verify_attempts = 0
            while verify_attempts < 3:
                qa = _generate_qa(text, source_reference, qtype)
                if qa is None:
                    break
                if not qa["requires_context"] and not keep_general:
                    logging.info(f"QA gefiltert (requires_context=False) Window {idx}")
                    qa = None
                    break
                # Billiger Regex-Vorfilter vor dem teuren LLM-Verifier (s. Filter oben).
                script_refs = _find_script_references(qa["question"])
                if script_refs:
                    logging.info(
                        f"Window {idx} Versuch {verify_attempts + 1}: "
                        f"Skript-Referenz in Frage - verworfen, refs={script_refs}"
                    )
                    verify_attempts += 1
                    qa = None
                    continue
                verdict = _verify_qa(qa, text)
                if verdict.passes:
                    if verify_attempts > 0:
                        logging.info(f"Window {idx}: Verifikation bestanden nach "
                                     f"{verify_attempts + 1} Versuchen.")
                    break
                logging.info(f"Window {idx} Versuch {verify_attempts + 1}: "
                             f"Verifikation fehlgeschlagen - issues={verdict.issues}")
                verify_attempts += 1
                qa = None
            if qa is None:
                continue
            dataset.append(qa)
            type_counts[qtype] += 1
            save_checkpoint(_CHECKPOINT_FILE, dataset)
            logging.info(f"Generiert {len(dataset)}/{target_questions} ({qtype}) "
                         f"| Verteilung: {type_counts}")
        except Exception as e:
            logging.warning(f"Generierung übersprungen Window {idx}: {e}")

    save_json(_OUTPUT_FILE, dataset)
    if _CHECKPOINT_FILE.exists():
        _CHECKPOINT_FILE.unlink()
    logging.info(f"Ground-Truth-Generierung fertig. Total: {len(dataset)}")


def main() -> None:
    """CLI-Einstieg für die alleinstehende Ground-Truth-Generierung."""
    parser = argparse.ArgumentParser(description="Ground-Truth-Generierung.")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--keep-general", action="store_true",
                        help="Auch Fragen mit requires_context=False behalten "
                             "(für Sensitivitätsanalyse).")
    args = parser.parse_args()

    config.ensure_output_dirs()
    setup_logger(config.DIR_GROUND_TRUTH / "2_dataset_generator.log")
    build_dataset(args.samples, keep_general=args.keep_general)


if __name__ == "__main__":
    main()
