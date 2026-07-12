"""Threadpool-Runner mit Checkpoint-Persistenz.

Die teuren Pipeline-Schritte (Baseline, RAG, Evaluation, Kategorisierung)
bestehen aus vielen unabhängigen, I/O-gebundenen API-Calls. run_parallel_with_checkpoint 
führt sie nebenläufig aus und schreibt den Fortschritt regelmäßig
in eine Checkpoint-Datei. Bricht ein Lauf ab (Fehler, Strg+C, Rate-Limit),
setzt derselbe Befehl exakt am letzten Stand wieder auf.

Vertrag mit den Aufrufern (vgl. s3–s5, s7):
- results ist eine bereits aus dem Checkpoint geladene Liste. Neue
  Ergebnisse werden in-place angehängt.
- pending enthält nur die noch offenen Items.
- fn(item) liefert einen Ergebnis-Record oder None (Item überspringen).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional

from .io_utils import PathLike, save_checkpoint

# Wie oft (in abgeschlossenen Items) der Zwischenstand persistiert wird.
_CHECKPOINT_EVERY = 5


def run_parallel_with_checkpoint(
    fn: Callable[[Any], Optional[dict]],
    pending: List[Any],
    checkpoint_file: PathLike,
    results: List[dict],
    *,
    max_workers: int,
    label: str = "item",
) -> List[dict]:
    """Verarbeite pending nebenläufig und sichere den Fortschritt periodisch.

    results wird in-place erweitert und zusätzlich zurückgegeben. Ergebnisse
    werden in Abschluss-Reihenfolge angehängt (die Reihenfolge ist für die
    nachgelagerten, schlüssel-basierten Auswertungen ohne Belang).

    Bei KeyboardInterrupt wird der aktuelle Stand noch als Checkpoint
    geschrieben und der Abbruch weitergereicht. Der Aufrufer (cli.py)
    beendet dann mit Exit-Code 130, ein Rerun greift den Checkpoint auf.
    """
    if not pending:
        logging.info(f"Keine offenen {label}s - nichts zu verarbeiten.")
        return results

    total = len(results) + len(pending)
    done = len(results)
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fn, item) for item in pending]
        try:
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as exc:  # einzelner Item-Fehler bricht den Lauf nicht ab
                    logging.error(f"{label}-Verarbeitung fehlgeschlagen: {exc}")
                    record = None
                with lock:
                    if record is not None:
                        results.append(record)
                    done += 1
                    if done % _CHECKPOINT_EVERY == 0:
                        save_checkpoint(checkpoint_file, results)
                        logging.info(f"Verarbeitet {done}/{total} {label}s "
                                     f"(Checkpoint gesichert)")
            with lock:
                save_checkpoint(checkpoint_file, results)
        except KeyboardInterrupt:
            # Laufende Futures abbrechen und den erreichten Stand sichern.
            for future in futures:
                future.cancel()
            with lock:
                save_checkpoint(checkpoint_file, results)
            logging.warning(f"Abbruch - Checkpoint mit {len(results)} {label}s gesichert.")
            raise

    logging.info(f"Fertig: {len(results)}/{total} {label}s verarbeitet.")
    return results
