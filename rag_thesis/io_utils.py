"""I/O-Helfer: JSON laden/speichern, Checkpoint-Resume, Logging-Setup.

Alle Pipeline-Schritte sind so implementiert, dass sie nach einem Abbruch
über die Checkpoint-Datei fortgesetzt werden können, wichtig für
kostenrelevante Langläufer (Skripte 1, 2, 4, 5).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, Path]


def rel(path: PathLike) -> str:
    """Stelle path relativ zum Projekt-Root dar (für Logs & Reports).

    Hält die Logs portabel: ein Lauf auf einer fremden Maschine erzeugt
    dieselben relativen Pfade wie der Lauf auf dem Entwicklungsrechner.
    Bei Pfaden außerhalb des Projekt-Baums fällt die Funktion auf die
    Originaldarstellung zurück.
    """
    p = Path(path).resolve()
    # Späte Importe vermeiden Zirkular-Importe: config zieht io_utils mit hoch.
    try:
        from . import config
        return str(p.relative_to(config.PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: PathLike) -> Any:
    """JSON-Datei laden (UTF-8). Wirft, wenn Datei fehlt."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: PathLike, data: Any) -> None:
    """JSON-Datei speichern (UTF-8, eingerückt, no-ascii für deutsche Umlaute)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: PathLike) -> List[Dict]:
    """Checkpoint laden. Leere Liste, falls Datei fehlt oder beschädigt ist."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                logging.info(f"Checkpoint geladen: {p} ({len(data)} Einträge)")
                return data
    except Exception as e:
        logging.warning(f"Checkpoint {p} beschädigt, starte neu: {e}")
    return []


def save_checkpoint(path: PathLike, data: List[Dict]) -> None:
    """Checkpoint atomar schreiben (Temp + Rename, um halbgeschriebene Dateien zu vermeiden)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
    except Exception as e:
        logging.error(f"Checkpoint-Schreiben fehlgeschlagen ({p}): {e}")


def setup_logger(log_file: PathLike, level: int = logging.INFO, *, also_stdout: bool = True) -> None:
    """Konfiguriere Root-Logger: Datei + optional stdout, mit Zeit-Präfix.

    Bei wiederholtem Aufruf werden bestehende Handler entfernt, damit jedes
    Skript sauber in seine eigene Log-Datei schreibt.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if also_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # OpenAI/HTTPX-Logs auf WARNING dämpfen, sonst flutet HTTP-Request-Output das Log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
