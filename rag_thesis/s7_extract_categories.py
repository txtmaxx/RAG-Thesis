"""Post-hoc-Analyse (A) - Retroaktive Kategorisierung der Faithfulness-Claims.

Explorative Auswertung NACH der Pipeline, nicht Teil der konfirmatorischen sechs
Schritte. Läuft nur mit der Option --with-posthoc oder als eigenständiges Modul.

Quantifiziert für den Datensatz unter outputs/5_evaluation/5_evaluation_results.json,
welcher Rolle jede Claim-Aussage zukommt: (A) faktische Behauptung,
(B) rechnerischer Zwischenschritt oder (C) Setup-Wiederholung aus der
Aufgabenstellung. Ruft pro Claim einen gpt-4o-mini-Call und schreibt:

  outputs/5_evaluation/5_categories_retroactive.json
  outputs/6_analysis/6_category_distribution.txt
  outputs/6_analysis/6_category_distribution.png   (falls matplotlib verfügbar)
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from . import config, prompts
from .io_utils import load_checkpoint, save_json, setup_logger
from .llm_client import structured_complete
from .parallel import run_parallel_with_checkpoint
from .stats_utils import wilson_ci

_CHECKPOINT_FILE = config.DIR_EVALUATION / "5_categories_retroactive_checkpoint.json"
_OUTPUT_JSON = config.DIR_EVALUATION / "5_categories_retroactive.json"
_OUTPUT_REPORT = config.DIR_ANALYSIS / "6_category_distribution.txt"
_OUTPUT_PLOT = config.DIR_ANALYSIS / "6_category_distribution.png"


# ─── Pydantic-Schema ──────────────────────────────────────────────────────────

class _CategoryDecision(BaseModel):
    """Einzelne Kategorie-Entscheidung mit kurzer Begründung."""
    category: Literal["A", "B", "C"]
    reasoning: str


# ─── Prompt ───────────────────────────────────────────────────────────────────

# Prompt-Text zentral in prompts.py (von s8 wiederverwendet).
_CATEGORIZE_PROMPT = prompts.CATEGORIZE_PROMPT


def _categorize_claim(claim: str, question: str) -> Optional[str]:
    """Klassifiziert eine einzelne Aussage als A/B/C."""
    user_content = (
        f"AUFGABENSTELLUNG:\n{question}\n\n"
        f"AUSSAGE:\n{claim}"
    )
    try:
        completion = structured_complete(
            messages=[
                {"role": "system", "content": _CATEGORIZE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            model=config.MODEL_TEXT,  # gpt-4o-mini ist hier ausreichend
            response_format=_CategoryDecision,
            max_tokens=200,
            temperature=config.TEMPERATURE,
        )
        parsed = completion.choices[0].message.parsed
        return parsed.category if parsed else None
    except Exception as e:
        logging.error(f"Kategorisierung fehlgeschlagen für Claim {claim[:80]!r}: {e}")
        return None


# ─── Datenfluss ───────────────────────────────────────────────────────────────

def _flatten_claims(evaluation_results: List[Dict]) -> List[Dict]:
    """Erzeugt eine flache Liste aller Claims aus beiden RAG-Modi.

    Jedes Element hat einen stabilen claim_id (Frage-Hash + System + Index)
    für die Checkpoint-Resume-Logik in run_parallel_with_checkpoint.

    Hinweis zur Reproduzierbarkeit: hash() auf str ist pro Prozess gesalzen,
    die claim_id ist also nur innerhalb eines Laufs stabil. Für prozess-
    übergreifend reproduzierbare IDs (z.B. wenn ein bestehendes Checkpoint-File
    aus einem anderen Lauf gelesen werden soll) entweder PYTHONHASHSEED=0
    setzen oder den Hash auf hashlib.sha1 umstellen. Für den einmaligen
    Lauf dieser Arbeit ist die Prozess-Stabilität
    ausreichend, da Checkpoint und Auswertung im selben Prozess entstehen.
    """
    items = []
    for d in evaluation_results:
        q = d["question"]
        qhash = abs(hash(q)) % (10 ** 12)
        qtype = d["question_type"]
        for system_key in ("rag_semantic", "rag_raw"):
            faith = d.get(system_key, {}).get("faithfulness", {}) or {}
            claims = faith.get("claims") or []
            verdicts = faith.get("verdicts") or []
            for i, claim in enumerate(claims):
                items.append({
                    "claim_id": f"{qhash}_{system_key}_{i}",
                    "question": q,
                    "question_type": qtype,
                    "system": system_key,
                    "claim_index": i,
                    "claim": claim,
                    "verdict": (verdicts[i] if i < len(verdicts) else None),
                })
    return items


def _process_one(item: Dict) -> Optional[Dict]:
    """Wird vom ThreadPool aufgerufen."""
    cat = _categorize_claim(item["claim"], item["question"])
    if cat is None:
        return None
    return {**item, "category": cat}


def run_categorization() -> List[Dict]:
    """Hauptlauf: lade Evaluation, klassifiziere alle Claims, persistiere."""
    config.ensure_output_dirs()
    eval_results = json.loads(Path(config.FILE_EVALUATION).read_text())
    claims = _flatten_claims(eval_results)
    logging.info(f"Zu klassifizierende Claims: {len(claims)} "
                 f"(aus {len(eval_results)} evaluierten Frage-Antwort-Paaren)")

    # Resume-fähig
    done_records = load_checkpoint(_CHECKPOINT_FILE)
    done_ids = {r["claim_id"] for r in done_records}
    pending = [c for c in claims if c["claim_id"] not in done_ids]
    logging.info(f"Bereits klassifiziert: {len(done_records)} | Offen: {len(pending)}")

    if pending:
        run_parallel_with_checkpoint(
            _process_one, pending, _CHECKPOINT_FILE, done_records,
            max_workers=config.MAX_WORKERS, label="claim",
        )

    save_json(_OUTPUT_JSON, done_records)
    if _CHECKPOINT_FILE.exists():
        _CHECKPOINT_FILE.unlink()
    logging.info(f"Geschrieben: {_OUTPUT_JSON} ({len(done_records)} Claims)")
    return done_records


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(records: List[Dict]) -> Tuple[Dict, Dict]:
    """Erzeugt den Verteilungs-Report nach System und Fragetyp."""
    # Aggregation: (system, qtype) -> Counter über Kategorien
    agg = defaultdict(Counter)
    agg_unsupported = defaultdict(Counter)  # nur Claims mit verdict=False
    totals_by_qtype = Counter()
    cat_by_qtype = defaultdict(Counter)
    for r in records:
        key = (r["system"], r["question_type"])
        agg[key][r["category"]] += 1
        totals_by_qtype[r["question_type"]] += 1
        cat_by_qtype[r["question_type"]][r["category"]] += 1
        if r.get("verdict") is False:
            agg_unsupported[key][r["category"]] += 1

    lines = []
    lines.append("=" * 72)
    lines.append("  CLAIM-KATEGORIE-VERTEILUNG (retroaktiv klassifiziert)")
    lines.append("  Datenquelle: outputs/5_evaluation/5_categories_retroactive.json")
    lines.append("  Klassifikator: gpt-4o-mini (post-hoc, T=0,0)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Gesamtzahl klassifizierter Claims: {len(records)}")
    lines.append("")
    lines.append("(1) VERTEILUNG NACH SYSTEM × FRAGETYP")
    for system_key in ("rag_semantic", "rag_raw"):
        lines.append(f"  System: {system_key}")
        for qtype in ("Definition", "Anwendung", "Transfer"):
            counts = agg.get((system_key, qtype), Counter())
            total = sum(counts.values())
            if total == 0:
                continue
            a, b, c = counts.get("A", 0), counts.get("B", 0), counts.get("C", 0)
            lines.append(
                f"    {qtype:<11s} (n={total:3d}): "
                f"A={a:3d} ({100*a/total:4.1f} %)  "
                f"B={b:3d} ({100*b/total:4.1f} %)  "
                f"C={c:3d} ({100*c/total:4.1f} %)"
            )
        lines.append("")

    lines.append("(2) AGGREGIERT ÜBER BEIDE SYSTEME, NACH FRAGETYP")
    for qtype in ("Definition", "Anwendung", "Transfer"):
        counts = cat_by_qtype[qtype]
        total = sum(counts.values())
        if total == 0:
            continue
        a, b, c = counts.get("A", 0), counts.get("B", 0), counts.get("C", 0)
        lines.append(
            f"  {qtype:<11s} (n={total:3d}): "
            f"A={a:3d} ({100*a/total:4.1f} %)  "
            f"B={b:3d} ({100*b/total:4.1f} %)  "
            f"C={c:3d} ({100*c/total:4.1f} %)"
        )
    lines.append("")

    # Setup-Wiederholungs-Bias: Anteil C nach Fragetyp
    lines.append("(3) HAUPT-BEFUND: ANTEIL KATEGORIE C (SETUP-WIEDERHOLUNG)")
    lines.append("    Hypothese: Anwendung/Transfer haben deutlich mehr")
    lines.append("    Setup-Wiederholungen als Definition.")
    lines.append("    CI = Wilson-Score-Intervall, 95 %.")
    lines.append("")
    for qtype in ("Definition", "Anwendung", "Transfer"):
        counts = cat_by_qtype[qtype]
        total = sum(counts.values())
        if total == 0:
            continue
        c = counts.get("C", 0)
        lo, hi = wilson_ci(c, total)
        lines.append(
            f"    {qtype:<11s}: {100*c/total:5.1f} %  ({c}/{total} Claims)  "
            f"95%-CI [{100*lo:.1f} %, {100*hi:.1f} %]"
        )
    lines.append("")
    lines.append("=" * 72)

    report_text = "\n".join(lines)
    _OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_REPORT.write_text(report_text, encoding="utf-8")
    logging.info(f"Report geschrieben: {_OUTPUT_REPORT}")
    return dict(agg), dict(cat_by_qtype)


def write_plot(cat_by_qtype: Dict) -> None:
    """Optional: Balkengrafik der Kategorien-Verteilung pro Fragetyp."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning(f"matplotlib nicht verfügbar, Plot übersprungen: {e}")
        return

    qtypes = ["Definition", "Anwendung", "Transfer"]
    cats = ["A", "B", "C"]
    data = []
    for q in qtypes:
        counts = cat_by_qtype.get(q, Counter())
        total = sum(counts.values()) or 1
        data.append([100 * counts.get(c, 0) / total for c in cats])

    fig, ax = plt.subplots(figsize=(9, 5))
    import numpy as np
    x = np.arange(len(qtypes))
    width = 0.27
    colors = ["#3b6cb6", "#7f9c4a", "#c25b5b"]
    labels = ["A - faktisch", "B - Rechnung", "C - Setup-Wiederholung"]
    for i, cat in enumerate(cats):
        vals = [data[j][i] for j in range(len(qtypes))]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i])
        # Wertelabel über jeden Balken (auch für die kleinen Anteile).
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(qtypes)
    ax.set_ylabel("Anteil der Claims (%)")
    ax.set_title("Claim-Kategorien nach Fragetyp (retroaktiv klassifiziert)")
    # Legende oben rechts (über den niedrigen Transfer-Balken), DPI wie s6.
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(_OUTPUT_PLOT, dpi=300)
    plt.close(fig)
    logging.info(f"Plot geschrieben: {_OUTPUT_PLOT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retroaktive Claim-Kategorisierung.")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Balkengrafik überspringen.")
    args = parser.parse_args()

    config.ensure_output_dirs()
    setup_logger(config.DIR_ANALYSIS / "7_categories.log")
    records = run_categorization()
    _, cat_by_qtype = write_report(records)
    if not args.skip_plot:
        write_plot(cat_by_qtype)


if __name__ == "__main__":
    main()
