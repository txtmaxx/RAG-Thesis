"""Konsolen-Entry-Points für die Pipeline.

Stellt einen Top-Level run_pipeline-Befehl bereit, der die sechs Pipeline-Schritte
sequenziell ausführt. Die beiden Post-hoc-Analysen (Kategorisierung + Validierung)
gehören NICHT zur Pipeline und laufen nur mit --with-posthoc oder als eigenständige
Module. Einzelne Schritte sind als Submodule mit eigenen main()-Funktionen direkt
ausführbar (siehe README).
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import (
    config,
    s1_ingestion,
    s2_ground_truth,
    s3_baseline,
    s4_rag_inference,
    s5_evaluation,
    s6_analysis,
    s7_extract_categories,
    s8_validate_categories,
)
from .io_utils import setup_logger


def run_pipeline() -> None:
    """Parse die CLI-Argumente und führe die ausgewählten Pipeline-Schritte aus.

    Greift bei KeyboardInterrupt Checkpoint-basiert und liefert Exit-Code 130
    (Unix-Standard für SIGINT), sodass ein Rerun nahtlos fortsetzen kann.
    """
    parser = argparse.ArgumentParser(
        prog="rag-pipeline",
        description="RAG-Pipeline (6 Schritte): Ingestion -> Ground Truth -> "
                    "Baseline & RAG -> Evaluation -> Statistik. Post-hoc-Analysen "
                    "(Kategorisierung + Validierung) optional via --with-posthoc.",
    )
    parser.add_argument("--samples", type=int, default=60,
                        help="Anzahl Ground-Truth-Fragen (default: 60, balanciert 20 je Fragetyp).")
    parser.add_argument("--keep-general", action="store_true",
                        help="Auch Fragen mit requires_context=False behalten "
                             "(für Sensitivitätsanalyse in Schritt 6).")
    parser.add_argument("--skip-review", action="store_true",
                        help="Manuellen Review-Stopp nach Schritt 2 überspringen.")
    parser.add_argument("--from-step", type=int, default=1, choices=range(1, 7),
                        help="Pipeline ab Schritt N starten (für teilweise Reruns).")
    parser.add_argument("--to-step", type=int, default=6, choices=range(1, 7),
                        help="Pipeline nach Schritt N beenden (default: 6). "
                             "Beispiel: --to-step 2 bricht nach der Ground-Truth-"
                             "Generierung sauber ab - geeignet für Nachtläufe, "
                             "die anschließend manuell reviewt werden.")
    parser.add_argument("--with-posthoc", action="store_true",
                        help="Nach der Pipeline zusätzlich die Post-hoc-Analysen "
                             "ausführen (Claim-Kategorisierung + deren Validierung). "
                             "Nicht Teil der konfirmatorischen Pipeline.")
    args = parser.parse_args()
    if args.to_step < args.from_step:
        parser.error(
            f"--to-step ({args.to_step}) muss ≥ --from-step ({args.from_step}) sein."
        )

    config.ensure_output_dirs()
    setup_logger(config.DIR_ORCHESTRATOR / "0_orchestrator.log")
    logging.info("=" * 72)
    # samples/keep_general gelten nur für Schritt 2, bei --from-step > 2 weglassen.
    if args.from_step <= 2 <= args.to_step:
        logging.info(
            f"Starte Pipeline | samples={args.samples} | from_step={args.from_step} "
            f"| to_step={args.to_step} | keep_general={args.keep_general}"
        )
    else:
        logging.info(
            f"Starte Pipeline | from_step={args.from_step} | to_step={args.to_step} "
            "(samples/keep_general nur für Schritt 2 relevant - übersprungen)"
        )
    logging.info("=" * 72)

    def _runs(step: int) -> bool:
        return args.from_step <= step <= args.to_step

    try:
        if _runs(1):
            logging.info(">>> Schritt 1a: Ingestion (semantic)")
            s1_ingestion.ingest(config.PDF_INPUT_FILE, "semantic")
            logging.info(">>> Schritt 1b: Ingestion (raw)")
            s1_ingestion.ingest(config.PDF_INPUT_FILE, "raw")
        if _runs(2):
            logging.info(">>> Schritt 2: Ground-Truth-Generierung")
            s2_ground_truth.build_dataset(args.samples, keep_general=args.keep_general)
            # Review-Stopp nur, wenn nach Schritt 2 weitere Schritte folgen
            # (bei --to-step 2 endet die Pipeline ohne ENTER-Wartepunkt).
            if not args.skip_review and args.to_step > 2:
                input(
                    "\n[PAUSE] Bitte führen Sie eine manuelle Qualitätsprüfung der "
                    f"Datei '{config.FILE_GOLDEN_DATASET}' durch.\n"
                    "Drücken Sie ENTER, um fortzufahren …"
                )
        if _runs(3):
            logging.info(">>> Schritt 3: Baseline-Inferenz")
            s3_baseline.build_baseline_answers()
        if _runs(4):
            logging.info(">>> Schritt 4a: RAG-Inferenz (semantic)")
            s4_rag_inference.build_rag_answers("semantic")
            logging.info(">>> Schritt 4b: RAG-Inferenz (raw)")
            s4_rag_inference.build_rag_answers("raw")
        if _runs(5):
            logging.info(">>> Schritt 5: LLM-as-a-Judge-Evaluation")
            s5_evaluation.run_evaluation()
        if _runs(6):
            logging.info(">>> Schritt 6: Statistische Analyse")
            s6_analysis.run_analysis()
        # Post-hoc-Analysen: NICHT Teil der Pipeline, nur auf ausdrückliche Anforderung.
        if args.with_posthoc:
            logging.info(">>> Post-hoc-Analyse A: Retroaktive Claim-Kategorisierung")
            records = s7_extract_categories.run_categorization()
            _, cat_by_qtype = s7_extract_categories.write_report(records)
            s7_extract_categories.write_plot(cat_by_qtype)
            logging.info(">>> Post-hoc-Analyse B: Validierung der Kategorisierung")
            s8_validate_categories.run_validation()

        if args.to_step < 6:
            msg = (f"\nPipeline bis Schritt {args.to_step} sauber beendet. "
                   f"Weiter mit `rag-pipeline --from-step {args.to_step + 1}`.")
        else:
            msg = (f"\nPipeline erfolgreich abgeschlossen. Ergebnisse in "
                   f"{config.OUTPUTS_DIR}/.")
        print(msg)
        logging.info("PIPELINE BEENDET.")
    except KeyboardInterrupt:
        # Abbruch durch Anwender: Checkpoints sind geschrieben, ein Rerun setzt fort.
        print("\nAbgebrochen. Resume mit demselben Befehl - bereits "
              "verarbeitete Items werden übersprungen.")
        sys.exit(130)
    except Exception as e:
        logging.exception(f"Pipeline-Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
