"""Text-Hilfsfunktionen: robuste String-Coercion, Whitespace- und LaTeX-Normalisierung.

Reine, seiteneffektfreie Stringoperationen. Bewusst ohne Abhängigkeiten, damit sie überall 
(Ingestion, Generierung, Evaluation) wiederverwendbar und leicht testbar sind.
"""

from __future__ import annotations

import re
from typing import Any

# Steuerzeichen außer Tab (\x09), LF (\x0a) und CR (\x0d). Aus PDF-/LLM-Text
# tauchen vereinzelt nicht-druckbare Zeichen auf, die JSON/Anzeige stören.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def ensure_str(value: Any) -> str:
    """Erzwinge einen String. None -> "", alles andere via str().

    Schützt nachgelagerte f-Strings und Prompts davor, an None oder
    Nicht-Strings (z.B. fehlende Felder, abgebrochene Modellantworten) zu
    scheitern.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_whitespace(text: str) -> str:
    """Vereinheitliche Whitespace ohne den Inhalt zu verändern.

    Kollabiert Folgen von Leerzeichen/Tabs zu einem Leerzeichen, entfernt
    Leerzeichen am Zeilenende und reduziert drei oder mehr Leerzeilen auf eine
    (= max. eine Leerzeile zwischen Absätzen). Trimmt Anfang und Ende.
    """
    s = ensure_str(text)
    if not s:
        return ""
    s = _CONTROL_CHARS_RE.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)        # mehrfache Spaces/Tabs -> ein Space
    s = re.sub(r"[ \t]+\n", "\n", s)     # trailing Whitespace pro Zeile
    s = re.sub(r"\n{3,}", "\n\n", s)     # max. eine Leerzeile
    return s.strip()


def sanitize_latex(text: str) -> str:
    """Vereinheitliche LaTeX-Mathematik auf $-/$$-Notation.

    PDF-Extraktion und LLM-Ausgaben mischen LaTeX-Delimiter (\\( … \\),
    \\[ … \\]) mit $-Notation. Für eine konsistente, vergleichbare
    Schreibweise werden die Klammer-Delimiter auf $/$$ vereinheitlicht. 
    Der mathematische Inhalt bleibt unverändert. Nur die Delimiter 
    und nicht-druckbare Steuerzeichen werden angefasst.
    """
    s = ensure_str(text)
    if not s:
        return ""
    s = _CONTROL_CHARS_RE.sub("", s)
    s = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", s, flags=re.DOTALL)  # \[ … \] -> $$ … $$
    s = re.sub(r"\\\((.+?)\\\)", r"$\1$", s, flags=re.DOTALL)    # \( … \) -> $ … $
    return s.strip()
