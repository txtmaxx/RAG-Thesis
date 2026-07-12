"""Post-hoc-Analyse (B) - Validierung der retroaktiven Claim-Kategorisierung.

Explorative Auswertung NACH der Pipeline, nicht Teil der konfirmatorischen sechs
Schritte. Läuft nur mit der Option --with-posthoc oder als eigenständiges Modul.

Hintergrund
-----------
Die Kategorisierung (s7_extract_categories) klassifiziert alle ~1.500 Faithfulness-
Claims mit gpt-4o-mini in A/B/C. Diese Kategorien tragen den
Haupt-Befund (Setup-Wiederholungs-Anteil je Fragetyp). Eine post-hoc-
Klassifikation durch ein einzelnes günstiges Modell ist ohne Gegenprobe aber
nur eine unbelegte Behauptung.

Dieses Skript zieht deshalb eine stratifizierte Zufallsstichprobe (10 Claims
je Fragetyp = 30) mit festem config.RANDOM_SEED und lässt sie unabhängig von
einem stärkeren Modell (gpt-4o, = config.MODEL_JUDGE) erneut
klassifizieren, mit exakt demselben Prompt wie in der Kategorisierung (s7). Aus beiden Label-
Reihen werden berechnet:

  - Gesamt-Übereinstimmung (Accuracy)
  - Precision/Recall je Kategorie (gpt-4o als Referenz)
  - ungewichteter Cohen's κ über A/B/C

Methodische Einschränkung
-------------------------
Dies ist eine KI-gestützte Zweitklassifikation, kein menschliches Rating.
Gemessen wird also die *Stabilität der Klassifikation über zwei Modell-Tiers
hinweg* (mini -> 4o), nicht die Übereinstimmung mit einem menschlichen
Goldstandard. Die Notes-Spalte der CSV markiert jede Zeile entsprechend mit
[AI-assisted, gpt-4o]. Diese Limitation wird offen ausgewiesen.

Outputs
-------
  outputs/5_evaluation/5_categories_validation_sample.csv
  outputs/6_analysis/6_category_validation.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config
from .io_utils import rel, setup_logger
from .llm_client import structured_complete
from .stats_utils import cohens_kappa_unweighted
# Identischer Prompt + Schema wie in der Kategorisierung (s7). Sonst wäre es nicht fair.
from .s7_extract_categories import _CATEGORIZE_PROMPT, _CategoryDecision

_SOURCE_JSON = config.DIR_EVALUATION / "5_categories_retroactive.json"
_OUTPUT_CSV = config.DIR_EVALUATION / "5_categories_validation_sample.csv"
_OUTPUT_REPORT = config.DIR_ANALYSIS / "6_category_validation.txt"

_PER_TYPE = 10
_QTYPES = ("Definition", "Anwendung", "Transfer")
_CATS = ("A", "B", "C")
_AGREEMENT_THRESHOLD = 0.70  # < 70 % Übereinstimmung -> Kategorie-Befunde abschwächen
_VALIDATION_MODEL = config.MODEL_JUDGE  # gpt-4o, unabhängig vom mini-Klassifikator


# ─── Stichprobe ─────────────────────────────────────────────────────────────

def stratified_sample(
    records: List[Dict], *, per_type: int = _PER_TYPE, seed: int = config.RANDOM_SEED,
) -> List[Dict]:
    """Ziehe per_type Claims je Fragetyp (reproduzierbar über seed).

    Reine Funktion ohne IO/Netz, direkt testbar. Fragetypen mit weniger als
    per_type Claims werden vollständig übernommen.
    """
    rng = random.Random(seed)
    out: List[Dict] = []
    by_type: Dict[str, List[Dict]] = {q: [] for q in _QTYPES}
    for r in records:
        if r.get("question_type") in by_type:
            by_type[r["question_type"]].append(r)
    for qtype in _QTYPES:
        pool = sorted(by_type[qtype], key=lambda r: r["claim_id"]) 
        k = min(per_type, len(pool))
        out.extend(rng.sample(pool, k))
    return out


# ─── Unabhängige Klassifikation (gpt-4o) ─────────────────────────────────────

def _classify_independent(claim: str, question: str) -> Optional[str]:
    """Zweitklassifikation mit gpt-4o, identischer Prompt wie in der Kategorisierung (s7)."""
    user_content = f"AUFGABENSTELLUNG:\n{question}\n\nAUSSAGE:\n{claim}"
    try:
        completion = structured_complete(
            messages=[
                {"role": "system", "content": _CATEGORIZE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            model=_VALIDATION_MODEL,
            response_format=_CategoryDecision,
            max_tokens=200,
            temperature=config.TEMPERATURE,
        )
        parsed = completion.choices[0].message.parsed
        return parsed.category if parsed else None
    except Exception as e:
        logging.error(f"Zweitklassifikation fehlgeschlagen: {e}")
        return None


# ─── Metriken ────────────────────────────────────────────────────────────────

def compute_metrics(machine: List[str], human: List[str]) -> Dict:
    """Accuracy, per-Kategorie Precision/Recall und ungewichteter κ.

    machine = Labels aus der Kategorisierung (s7, gpt-4o-mini), human = Referenz aus
    der gpt-4o-Zweitklassifikation. Reine Funktion (testbar).
    """
    n = len(machine)
    assert n == len(human), "Label-Reihen müssen gleich lang sein"
    agree = sum(1 for m, h in zip(machine, human) if m == h)
    accuracy = agree / n if n else float("nan")

    per_cat: Dict[str, Dict[str, float]] = {}
    for cat in _CATS:
        tp = sum(1 for m, h in zip(machine, human) if m == cat and h == cat)
        machine_pos = sum(1 for m in machine if m == cat)
        human_pos = sum(1 for h in human if h == cat)
        precision = tp / machine_pos if machine_pos else float("nan")
        recall = tp / human_pos if human_pos else float("nan")
        per_cat[cat] = {
            "tp": tp, "machine_n": machine_pos, "human_n": human_pos,
            "precision": precision, "recall": recall,
        }

    kappa = cohens_kappa_unweighted(machine, human, categories=_CATS)
    return {
        "n": n, "agree": agree, "accuracy": accuracy,
        "per_cat": per_cat, "kappa": kappa,
    }


# ─── IO ──────────────────────────────────────────────────────────────────────

def write_csv(rows: List[Dict]) -> None:
    """Schreibe die Validierungs-Stichprobe als CSV (eine Zeile je Claim)."""
    _OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow([
            "claim_id", "question_type", "system", "claim",
            "machine_category", "human_category", "agreement", "notes",
        ])
        for r in rows:
            writer.writerow([
                r["claim_id"], r["question_type"], r["system"],
                " ".join(str(r["claim"]).split()),  # Whitespace normalisieren
                r["machine_category"], r["human_category"],
                "1" if r["machine_category"] == r["human_category"] else "0",
                "[AI-assisted, gpt-4o]",
            ])
    logging.info(f"Validierungs-Stichprobe geschrieben ({len(rows)} Claims): "
                 f"{rel(_OUTPUT_CSV)}")


def write_report(rows: List[Dict], metrics: Dict) -> None:
    """Schreibe den Validierungs-Report mit Accuracy/Precision/Recall/κ."""
    by_type = Counter(r["question_type"] for r in rows)
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  VALIDIERUNG DER RETROAKTIVEN CLAIM-KATEGORISIERUNG")
    lines.append("  Erst-Klassifikator:  gpt-4o-mini (s7)")
    lines.append(f"  Zweit-Klassifikator: {_VALIDATION_MODEL} (unabhängig, gleicher Prompt)")
    lines.append("  Art: KI-gestützte Zweitklassifikation, KEIN menschliches Rating")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Stichprobe: {metrics['n']} Claims, stratifiziert "
                 f"({', '.join(f'{q}={by_type.get(q,0)}' for q in _QTYPES)}), "
                 f"Seed={config.RANDOM_SEED}")
    lines.append("")
    lines.append("(1) GESAMT-ÜBEREINSTIMMUNG")
    lines.append(f"    Accuracy: {100*metrics['accuracy']:.1f} %  "
                 f"({metrics['agree']}/{metrics['n']} Claims gleich klassifiziert)")
    lines.append(f"    Cohen's κ (ungewichtet, A/B/C): {metrics['kappa']:.3f}")
    lines.append("")
    lines.append("(2) PRECISION / RECALL JE KATEGORIE (gpt-4o als Referenz)")
    lines.append("    Kat   TP   mini=Kat   4o=Kat   Precision   Recall")
    for cat in _CATS:
        pc = metrics["per_cat"][cat]
        prec = "  n/a " if pc["precision"] != pc["precision"] else f"{100*pc['precision']:5.1f}%"
        rec = "  n/a " if pc["recall"] != pc["recall"] else f"{100*pc['recall']:5.1f}%"
        lines.append(f"    {cat:<4s}{pc['tp']:4d}{pc['machine_n']:10d}{pc['human_n']:9d}"
                     f"     {prec}    {rec}")
    lines.append("")
    lines.append("(3) DISSENS-FÄLLE (mini ≠ 4o)")
    dissent = [r for r in rows if r["machine_category"] != r["human_category"]]
    if not dissent:
        lines.append("    keine")
    for r in dissent:
        claim_short = " ".join(str(r["claim"]).split())[:90]
        lines.append(f"    [{r['question_type'][:4]}] mini={r['machine_category']} "
                     f"4o={r['human_category']}: {claim_short}")
    lines.append("")
    lines.append("(4) EINORDNUNG")
    acc = metrics["accuracy"]
    kappa = metrics["kappa"]
    if acc >= _AGREEMENT_THRESHOLD and (kappa != kappa or kappa >= 0.6):
        lines.append(f"    Accuracy ≥ {int(100*_AGREEMENT_THRESHOLD)} % und κ ausreichend:")
        lines.append("    Die retroaktive Klassifikation ist über zwei Modell-Tiers")
        lines.append("    stabil. Die Kategorie-Anteile werden gestützt.")
    else:
        lines.append(f"    Accuracy < {int(100*_AGREEMENT_THRESHOLD)} % bzw. κ < 0,6:")
        lines.append("    Die Klassifikation ist NICHT robust. Die Kategorie-Anteile")
        lines.append("    sind als grobe Indikatoren zu lesen, nicht als exakte Werte.")
    lines.append("")
    lines.append("    Hinweis: Gemessen wird die Modell-übergreifende Stabilität der")
    lines.append("    Klassifikation, nicht die Übereinstimmung mit einem menschlichen")
    lines.append("    Goldstandard. Eine menschliche Annotation bleibt im Ausblick.")
    lines.append("")
    lines.append("=" * 72)

    _OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"Validierungs-Report geschrieben: {rel(_OUTPUT_REPORT)}")


# ─── Orchestrierung ───────────────────────────────────────────────────────────

def run_validation() -> Tuple[List[Dict], Dict]:
    """Vollständiger Validierungslauf. Liefert (rows, metrics)."""
    config.ensure_output_dirs()
    records = json.loads(Path(_SOURCE_JSON).read_text(encoding="utf-8"))
    sample = stratified_sample(records)
    logging.info(f"Stratifizierte Stichprobe: {len(sample)} Claims "
                 f"({_PER_TYPE} je Fragetyp), Seed={config.RANDOM_SEED}")

    rows: List[Dict] = []
    for i, rec in enumerate(sample, 1):
        human_cat = _classify_independent(rec["claim"], rec["question"])
        if human_cat is None:
            logging.warning(f"Claim {rec['claim_id']} ohne Zweit-Label - übersprungen.")
            continue
        rows.append({
            "claim_id": rec["claim_id"],
            "question_type": rec["question_type"],
            "system": rec["system"],
            "claim": rec["claim"],
            "machine_category": rec["category"],
            "human_category": human_cat,
        })
        logging.info(f"  {i}/{len(sample)} klassifiziert "
                     f"(mini={rec['category']} / 4o={human_cat})")

    machine = [r["machine_category"] for r in rows]
    human = [r["human_category"] for r in rows]
    metrics = compute_metrics(machine, human)

    write_csv(rows)
    write_report(rows, metrics)

    logging.info(f"Validierung fertig: Accuracy={100*metrics['accuracy']:.1f} % | "
                 f"κ={metrics['kappa']:.3f}")
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validierung der retroaktiven Claim-Kategorisierung (gpt-4o-Gegenprobe).")
    parser.parse_args()
    config.ensure_output_dirs()
    setup_logger(config.DIR_ANALYSIS / "8_category_validation.log")
    run_validation()


if __name__ == "__main__":
    main()
