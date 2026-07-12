"""Zentrale Sammlung aller LLM-Prompts der Pipeline.

Analog zu config.py bündelt dieses Modul den gesamten Prompt-Text an einer
Stelle, damit Prompt Engineering ohne Eingriff in die Schritt-Logik möglich ist.
Konventionen: statische Prompts -> Konstanten (GROSSBUCHSTABEN), Per-Fragetyp ->
Dicts, parametrisierte Prompts -> Builder-Funktionen. Der Namens-Präfix ordnet
jeden Eintrag einem Schritt zu (IMG_/RELEVANCE_/GROUND_TRUTH_/BASELINE_/RAG_/…).
"""
from __future__ import annotations

from typing import Dict


# ═══════════════════════════════════════════════════════════════════════════════
# Schritt 1 - Ingestion: Bildanalyse (Vision) und Text-Bereinigung
# ═══════════════════════════════════════════════════════════════════════════════

IMG_INSTRUCTION_FULL = "Analysiere diesen Folien-Screenshot für eine Wissensdatenbank. Ignoriere den reinen Fließtext. Konzentriere dich auf Diagramme, Formeln und Grafiken."
IMG_INSTRUCTION_EMBEDDED = "Analysiere dieses extrahierte Bild für eine Wissensdatenbank."

# Härtung gegen Tabellen-Verschiebungen in der OCR/Vision-Analyse. Wird an ALLE
# Vision-Calls angehängt. No-op, falls die Folie keine Tabelle enthält.
TABLE_GUARDRAILS = (
    "\n"
    "\n"
    "TABELLEN-REGELN (zwingend, falls Tabellen sichtbar):\n"
    "- Übernimm die Tabellenstruktur EXAKT - keine Spalten zusammenfassen, keine Zeilen weglassen, keine Reihenfolge ändern.\n"
    "- Erste Spalte einer Wertetabelle ist häufig ein Zeilenindex (0..N-1). Kennzeichne ihn klar als 'Index'-Spalte.\n"
    "- Bei Wahrheitstabellen mit Binärspalten (z.B. A,B,C,D,Y): Prüfe vor der Ausgabe, dass die Binärwerte ZEILENWEISE zum Zeilenindex passen (z.B. Index 5 -> ABCD = 0101 bei 4-Bit). Wenn die OCR offensichtlich Spalten verschoben hat, korrigiere die Zuordnung anhand der Zeilenindex/Binär-Konsistenz.\n"
    "- Bei KV-Diagrammen: Markiere explizit die Zeilen-/Spaltenbeschriftung und die Gray-Code-Reihenfolge (00, 01, 11, 10).\n"
    "- Falls eine Tabelle in sich widersprüchlich erscheint, BENENNE den Widerspruch in der Beschreibung (statt einen Inhalt zu erfinden)."
)


def image_analysis_prompt(*, is_full_page: bool, context: str) -> str:
    """Vollständiger Vision-Prompt für die Bild-/Folienanalyse (Schritt 1)."""
    instruction = IMG_INSTRUCTION_FULL if is_full_page else IMG_INSTRUCTION_EMBEDDED
    return (
        f"{instruction}\n"
        "ANTWORT-REGELN:\n"
        "1. Sei extrem präzise und faktenbasiert.\n"
        "2. Konzentriere dich AUSSCHLIESSLICH auf visuelle Informationen (Diagramme, Grafiken, Formeln).\n"
        "3. Ignoriere redundante Texte, Logos und dekorative Elemente.\n"
        "4. Beschreibe Beziehungen, Flüsse und Strukturen logisch.\n"
        "5. Gib mathematische Formeln zwingend in LaTeX wieder ($ ... $ oder $$ ... $$).\n"
        "6. Falls das Bild keinen inhaltlichen Mehrwert für eine Wissensdatenbank hat, antworte NUR mit 'KEINE_INFO'.\n"
        f"{TABLE_GUARDRAILS}\n"
        f"Text-Kontext der Seite:\n{context[:1500]}"
    )


def text_cleaning_prompt(safe_input: str) -> str:
    """Prompt zur Umwandlung von OCR-Rohtext in sauberes Markdown (Schritt 1)."""
    return (
        "Wandle den folgenden OCR-extrahierten Text aus einem PDF-Vorlesungsskript in sauberes Markdown um.\n"
        "REGELN:\n"
        "- Behalte inhaltliche Tabellen (mit fachlichen Daten) als Markdown-Tabellen bei.\n"
        "- Entferne reine Layout-Tabellen ohne semantischen Inhalt.\n"
        "- Mathematische Formeln und Gleichungen zwingend als LaTeX ($ ... $ inline, $$ ... $$ als Block).\n"
        "- Entferne Seitenzahlen, Kopf- und Fußzeilen.\n"
        "- Erhalte die Hierarchie der Überschriften (# / ## / ###).\n"
        "WICHTIG: Antworte AUSSCHLIESSLICH mit dem konvertierten Markdown. Keine Einleitungen, keine Erklärungen.\n"
        f"INPUT TEXT:\n{safe_input}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Schritt 2 - Ground-Truth-Generierung: Relevanzfilter, Generator, Verifier
# ═══════════════════════════════════════════════════════════════════════════════

RELEVANCE_CHECK_SYSTEM = (
    "Prüfe den folgenden Textabschnitt.\n"
    "Antworte ausschließlich mit TRUE oder FALSE.\n"
    "TRUE: Der Text enthält fachlich sinnvollen, prüfungsrelevanten Informatik-Inhalt.\n"
    "FALSE: Der Text besteht überwiegend aus Metadaten, Seitenzahlen, Inhaltsverzeichnis oder unzusammenhängenden Fragmenten."
)

# Per-Fragetyp-Ziel, das in den Generator-System-Prompt eingebettet wird.
GROUND_TRUTH_TYPE_INSTRUCTIONS: Dict[str, str] = {
    "Definition": "ZIEL: Der Studierende gibt eine formale Definition, eine Eigenschaft oder einen Kernbegriff aus dem Text präzise wieder. Die Frage muss EXAKTE Notation, korrekte Quantoren und skript-spezifische Symbolik einfordern, sodass ein ungefähres Lehrbuch-Wissen NICHT ausreicht. Bevorzuge Definitionen, die im Skript mit spezifischen Bezeichnern, Reihenfolgen oder Konventionen versehen sind.",
    "Anwendung": "ZIEL: Der Studierende wendet ein konkretes Verfahren oder einen Algorithmus aus dem Text auf ein Rechenbeispiel an. Falls der Text Formeln oder Tabellen enthält, formuliere eine rechnerische Aufgabe mit konkreten Zahlenwerten. Die Musterlösung MUSS jeden Zwischenschritt explizit ausweisen, damit die Faithfulness-Bewertung auf der Skript-Methode (nicht auf einer alternativen Standardrechnung) prüfbar bleibt.",
    "Transfer": "ZIEL: Der Studierende wendet ein Konzept des Skripts auf ein NEUES KONKRETES Zahlen- oder Strukturbeispiel an, das so nicht wörtlich im Text steht (z.B. andere Werte, andere Konstellation derselben Methode). VERBOTEN: Fiktive Entitäten ('hypothetische Zahlenmenge', 'fiktives Universum', erfundene Wesen / Strukturen mit eigenen Regeln) - solche Antworten lassen sich nicht gegen den Quelltext prüfen und sind methodisch ungeeignet. Die Musterlösung muss vollständig auf Skript-Begriffen, -Notation und -Methoden basieren.",
}


def ground_truth_generation_system(qtype: str) -> str:
    """System-Prompt des Ground-Truth-Generators (stärkstes Modell, Schritt 2).

    Bettet die fragetyp-spezifische Zielvorgabe (GROUND_TRUTH_TYPE_INSTRUCTIONS)
    in die festen kritischen Regeln und Konsistenz-Constraints ein.
    """
    return (
        f"Du bist ein strenger Hochschul-Dozent für Informatik, der eine anspruchsvolle Klausur konzipiert.\n"
        f"Erstelle genau EINE {qtype}-Prüfungsfrage mit Musterlösung auf Basis des untenstehenden Textes.\n\n"
        f"{GROUND_TRUTH_TYPE_INSTRUCTIONS.get(qtype, '')}\n\n"
        "KRITISCHE REGELN:\n"
        "- STRICT CONTEXT: Nutze für die Musterlösung STRIKT NUR das Wissen aus dem bereitgestellten Text.\n"
        "- KEINE META-REFERENZ AUF DEN KONTEXT (zwingend): Die Frage darf NIEMALS Phrasen wie 'im Skript', 'laut Skript', 'aus dem Skript', 'wie im Skript', 'im Quelltext', 'im Originaltext', 'in der Vorlesung', 'im bereitgestellten Text', 'auf Seite N' oder ähnliche Verweise auf das Lehrmaterial enthalten. Eine echte Klausurfrage steht für sich allein - der Studierende hat in der Prüfung KEIN Skript zur Hand. Solche Meta-Verweise sind zudem methodisch unfair gegenüber der Baseline (LLM ohne Retrieval), die das Skript nicht kennt. Verstöße führen zu automatischer Verwerfung.\n"
        "- KEINE IMPLIZITEN MATERIAL-VERWEISE (zwingend): Die Frage darf auch nicht auf Tabellen, Formeln, Abbildungen, Diagramme, Übergangstabellen oder Analysen verweisen, die NICHT in der Aufgabenstellung selbst mitgeliefert sind. Falsch: 'wie sie in der Tabelle dargestellt sind', 'anhand der gegebenen Wahrheitstabelle', 'mit der gegebenen Formel', 'in der Form, die in der Analyse verwendet wird', 'wie auf der Abbildung gezeigt'. Richtig: Die Tabelle/Formel als Markdown bzw. LaTeX direkt in die Frage einbetten, oder die nötigen Werte/Variablen ausschreiben. Wenn ein KV-Diagramm gemeint ist, gib die Belegung der Einsen explizit als Liste von Variablenkombinationen an.\n"
        "- DETAIL-ANKER (gegen Decken-Effekt): Die Frage MUSS mindestens ein Detail enthalten, das ohne den exakten Skript-Inhalt NICHT korrekt beantwortbar ist - aber dieses Detail muss DIREKT in die Aufgabenstellung eingebaut sein, nicht durch einen Verweis auf das Skript. Statt 'Verwenden Sie die im Skript definierte Notation' schreibe konkret: 'Definieren Sie die FSM als 5-Tupel in der Reihenfolge (S, Σ, δ, s₀, F)'. Statt 'wie auf Seite N dargestellt' liste die fraglichen Werte/Variablen explizit. Reine Lehrbuch-Standardfragen, die ein vortrainiertes Modell ohne Kontext zuverlässig löst, sind UNERWÜNSCHT.\n"
        "- FOKUS AUF SEMANTIK: Bevorzuge ausdrücklich Fragen zu Tabellen, LaTeX-Formeln oder Bildbeschreibungen (z.B. markiert mit [BILD-INFO: ...] oder [SEITEN-GRAFIK-ANALYSE: ...]), falls diese im Text vorkommen.\n"
        "- ANTI-LEXIKALISCHER BIAS: Verwende Synonyme und Umschreibungen in der Fragestellung. Kopiere keine exakten Kernbegriffe 1:1 aus dem Text.\n"
        "- PRÜFUNGSRELEVANZ: Generiere NIEMALS Fragen zu organisatorischen oder administrativen Inhalten.\n"
        "- KLARHEIT: Die Frage muss ohne Kenntnis des Originaltextes vollständig verständlich sein.\n"
        "- MUSTERLÖSUNG (Dozent-Stil, nicht Tutorial-Stil): Schreibe wie ein Hochschul-Dozent eine offizielle Klausur-Musterlösung formuliert - vollständig, aber kompakt. Konkret heißt das:\n"
        "  * Mittlere Länge: typisch 80–250 Wörter. Definitionsfragen eher am unteren Ende, Anwendungs-/Transferaufgaben mit Rechenweg am oberen.\n"
        "  * Strukturierte Form, aber NICHT durchgängig stichpunktartig wie eine Spickzettel-Liste. Verwende vollständige Sätze für definitorische Aussagen. Bullets nur dort, wo der Inhalt natürlich aufzählend ist (Rechenschritte, Eigenschaften, Komponenten).\n"
        "  * Keine Tutorial-Phrasen wie 'Schritt 1:', 'Zuerst tun wir X', 'Wie wir wissen…'. Auch keine erklärenden Einschübe für triviale Operationen.\n"
        "  * Mathematik in LaTeX ($...$ inline, $$...$$ block).\n"
        "  * Bei Rechenaufgaben: alle relevanten Zwischenschritte zeigen, aber keine redundanten Selbstverständlichkeiten ausführen.\n"
        "  * Keine Meta-Kommentare zur Lösung ('Die Antwort ist also…' am Schluss).\n"
        "- FAITHFULNESS-PRÜFBARKEIT: Die Antwort muss in atomare faktische Aussagen zerlegbar sein, deren Belegbarkeit gegen den Quelltext entscheidbar ist. Vermeide Fragen, deren Antworten zwingend Inhalte enthalten, die nicht im Skript stehen können (z.B. erfundene Entitäten).\n"
        "\n"
        "REQUIRES_CONTEXT: Setze requires_context=true NUR dann, wenn die Musterlösung auf skript-spezifischen Fakten basiert (eigene Definitionen, Notationen, Verfahrensschritte), die NICHT im allgemeinen Informatik-Lehrbuch-Wissen enthalten sind. Setze requires_context=false, wenn die Frage auch ohne diesen Text mit Standard-Universitätswissen beantwortet werden könnte.\n"
        "\n"
        "ZUSÄTZLICHE KONSISTENZ-CONSTRAINTS (gegen typische Fehlerklassen):\n"
        "- BINÄR-DEZIMAL-KONSISTENZ: Wenn die Frage eine Variablenkombination wie (A=1, B=0, C=1, D=0) einer Dezimalzahl zuordnet, MUSS die Binärdarstellung (MSB->LSB) zur Dezimalzahl passen. Beispiel: (A=1,B=0,C=1,D=0) ↦ 1010₂ = 10₁₀, NICHT 12.\n"
        "- KV-DIAGRAMM-GRUPPEN: Eine Gruppe von 2ⁿ benachbarten Einsen erfordert exakt n variable Bits. ALLE anderen Bits müssen in allen Zellen der Gruppe identisch sein (Hamming-Abstand 1 zwischen direkt benachbarten Zellen). Falsche Gruppierungen verbieten.\n"
        "- MODULO-N-ZÄHLER: Ein Modulo-N-Zähler durchläuft GENAU N Zustände, danach Wrap auf 0. Bei N < 2ᵏ sind die nicht-erreichbaren Zustände Don't Cares. Sie treten in der Übergangstabelle nicht auf.\n"
        "- ZUSTÄNDE vs. FLIPFLOPS: Z Zustände erfordern ⌈log₂(Z)⌉ Flipflops. Eine Antwort mit k Flipflops kann höchstens 2ᵏ Zustände codieren.\n"
        "- KEINE FABRIZIERTEN STRUKTUREN: Wenn die Frage konkrete Dimensionen (z.B. 3×3-Matrix, 6-Bit-Adresse) oder Werte nennt, müssen diese im Quelltext explizit stehen oder logisch konsistent abgeleitet sein. Erfinde keine Größen, die der Text nicht hergibt.\n"
        "- SELBST-VERIFIZIERBARKEIT: Jede mathematische Behauptung in der Musterlösung muss mit den in der Frage gegebenen Werten arithmetisch verifizierbar sein."
    )


GROUND_TRUTH_VERIFY_SYSTEM = (
    "Du bist ein peinlich genauer Korrektor für Informatik-Prüfungsfragen. Prüfe den folgenden Frage/Antwort-Block auf:\n"
    "\n"
    "1. Interne Konsistenz: Widersprechen sich Frage und Antwort? (z.B. 'vier Zustände' in der Frage vs. 3-Flipflop-Antwort)\n"
    "2. Mathematische Korrektheit: Stimmen Binär-Dezimal-Umrechnungen, KV-Gruppierungen (Hamming-Distanz 1 für 2er-Gruppen!), Modulo-Arithmetik, Volladdierer-Wahrheitswerte, MIPS-Bit-Felder usw.?\n"
    "3. Quell-Konsistenz: Sind alle in der Antwort behaupteten Skript-Spezifika (Seitenzahlen, Notationen, konkrete Zahlenwerte) im bereitgestellten Quelltext belegbar?\n"
    "4. Fabricated Structures: Werden konkrete Dimensionen erfunden, die im Quelltext NICHT stehen (z.B. 'eine 3x3-Matrix mit 6-Bit-Adresse', obwohl der Text nur generisch von Matrixspeichern spricht)?\n"
    "\n"
    "passes=true setzen NUR wenn ALLE vier Prüfungen bestanden sind. Andernfalls in 'issues' präzise auflisten, was falsch ist."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Schritt 3 - Baseline-Inferenz (Antwort OHNE Retrieval)
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE_PROMPTS_BY_TYPE: Dict[str, str] = {
    "Definition": (
        "Du bist Dozent für Informatik und erstellst eine offizielle Musterlösung für eine Klausur.\n"
        "AUFGABE: Beantworte die folgende Definitionsfrage präzise und faktenbasiert.\n"
        "REGELN:\n"
        "- Nutze AUSSCHLIESSLICH dein internes Wissen. Kein Zugriff auf externe Dokumente.\n"
        "- Definiere den Begriff exakt. Nenne formale Eigenschaften und Abgrenzungen.\n"
        "- Schreibe im Stil einer Musterlösung: stichpunktartig, informationsdicht.\n"
        "- Keine Einleitungen oder Zusammenfassungen. Mathematische Inhalte in LaTeX."
    ),
    "Anwendung": (
        "Du bist Dozent für Informatik und erstellst eine offizielle Musterlösung für eine Klausur.\n"
        "AUFGABE: Beantworte die folgende Anwendungsaufgabe durch Anwendung des relevanten Verfahrens.\n"
        "REGELN:\n"
        "- Nutze AUSSCHLIESSLICH dein internes Wissen.\n"
        "- Zeige den Lösungsweg Schritt für Schritt. Führe Berechnungen vollständig durch.\n"
        "- Dynamische Länge: Liefere exakt die Fakten, die für die volle Punktzahl nötig sind.\n"
        "- Mathematische Inhalte in LaTeX."
    ),
    "Transfer": (
        "Du bist Dozent für Informatik und erstellst eine offizielle Musterlösung für eine Klausur.\n"
        "AUFGABE: Beantworte die folgende Transferaufgabe durch Übertragung bekannter Konzepte auf das neue Szenario.\n"
        "REGELN:\n"
        "- Nutze AUSSCHLIESSLICH dein internes Wissen.\n"
        "- Identifiziere das zugrunde liegende Konzept, übertrage es logisch auf das beschriebene Szenario.\n"
        "- Begründe deine Schlussfolgerungen klar.\n"
        "- Keine Einleitungen. Mathematische Inhalte in LaTeX."
    ),
}

BASELINE_DEFAULT_PROMPT = (
    "Du bist Dozent für Informatik und erstellst eine offizielle Musterlösung für eine Klausur.\n"
    "AUFGABE: Beantworte die folgende Prüfungsfrage prägnant, fachlich korrekt und ohne Umschweife.\n"
    "REGELN:\n"
    "- Nutze AUSSCHLIESSLICH dein internes Wissen.\n"
    "- Dynamische Länge: Liefere exakt die Fakten, die für die volle Punktzahl nötig sind.\n"
    "- Schreibe im Stil einer Musterlösung. Keine Einleitungen oder Zusammenfassungen.\n"
    "- Mathematische Inhalte in LaTeX."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Schritt 4 - RAG-Inferenz (Antwort MIT Retrieval-Kontext)
# ═══════════════════════════════════════════════════════════════════════════════

RAG_PROMPTS_BY_TYPE: Dict[str, str] = {
    "Definition": (
        "Du bist Dozent für Informatik und erstellst eine offizielle Musterlösung für eine Klausur.\n"
        "AUFGABE: Beantworte die folgende Definitionsfrage präzise auf Basis des Kontexts.\n"
        "REGELN:\n"
        "- Nutze primär die Informationen aus dem KONTEXT.\n"
        "- Definiere den Begriff exakt. Nenne formale Eigenschaften und Abgrenzungen wie im Skript beschrieben.\n"
        "- Keine Erwähnung des Kontexts oder dessen Herkunft.\n"
        "- Stichpunktartig, informationsdicht. Keine Einleitungen. LaTeX für Mathematik."
    ),
    "Anwendung": (
        "Du bist Dozent für Informatik und erstellst eine offizielle Musterlösung für eine Klausur.\n"
        "AUFGABE: Beantworte die folgende Anwendungsaufgabe.\n"
        "REGELN:\n"
        "- Nutze primär die Informationen aus dem KONTEXT.\n"
        "- Wende das Verfahren aus dem Kontext an. Führe Berechnungen vollständig durch.\n"
        "- Zeige den Lösungsweg Schritt für Schritt.\n"
        "- Keine Erwähnung des Kontexts. LaTeX für Mathematik."
    ),
    "Transfer": (
        "Du bist Dozent für Informatik und erstellst eine offizielle Musterlösung für eine Klausur.\n"
        "AUFGABE: Beantworte die folgende Transferaufgabe.\n"
        "REGELN:\n"
        "- Nutze primär die Konzepte aus dem KONTEXT und wende sie logisch auf das neue Szenario an.\n"
        "- Externes Logikwissen zur Brückenbildung zwischen Kontext und Szenario ist erlaubt.\n"
        "- Die Kernfakten des Kontexts dürfen nicht verfälscht werden.\n"
        "- Begründe deine Schlussfolgerungen. Keine Erwähnung des Kontexts. LaTeX für Mathematik."
    ),
}

RAG_DEFAULT_PROMPT = (
    "Du bist Dozent für Informatik und erstellst eine offizielle Musterlösung für eine Klausur.\n"
    "AUFGABE: Beantworte die folgende Prüfungsfrage prägnant und faktenbasiert.\n"
    "REGELN:\n"
    "- Nutze primär die Informationen aus dem KONTEXT.\n"
    "- Dynamische Länge: Liefere exakt die Fakten, die für die volle Punktzahl nötig sind.\n"
    "- Keine Erwähnung des Kontexts. Keine Einleitungen. LaTeX für Mathematik."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Schritt 5 - LLM-as-a-Judge: Correctness (Likert) + Faithfulness (Ragas-Stil)
# ═══════════════════════════════════════════════════════════════════════════════

CORRECTNESS_PROMPT = (
    "Du bist ein sehr strenges Evaluationssystem für akademische Prüfungsantworten.\n"
    "AUFGABE: Bewerte die FAKTISCHE KORREKTHEIT der ANTWORT im Vergleich zur GROUND TRUTH.\n"
    "\n"
    "BEWERTUNGSSKALA (1–5):\n"
    "5 – Alle Kernfakten der Ground Truth vollständig und korrekt enthalten\n"
    "4 – Die meisten Kernfakten vorhanden, nur minimale Lücken ohne inhaltliche Fehler\n"
    "3 – Wesentliche Fakten vorhanden, aber merkliche Lücken oder leichte Ungenauigkeiten\n"
    "2 – Nur Teilwissen erkennbar, wichtige Kernfakten fehlen oder sind fehlerhaft\n"
    "1 – Keine oder überwiegend falsche Fakten im Vergleich zur Ground Truth\n"
    "\n"
    "KRITERIEN:\n"
    "- Die Antwort darf keine Fakten enthalten, die der Ground Truth widersprechen.\n"
    "- Fehlende Details aus der Ground Truth führen zu Abzügen.\n"
    "- Erfundene Fakten führen zu massivem Abzug.\n"
    "- Zusätzliche, fachlich korrekte Details, die über die Ground Truth hinausgehen, sind positiv zu werten.\n"
    "- Bewerte NICHT die Länge der Antwort. Eine kurze, präzise Antwort ist gleichwertig oder besser als eine ausschweifende Antwort, solange alle Fakten stimmen.\n"
    "VORGEHEN: Analysiere die Antwort Schritt für Schritt gegen die Ground Truth, dann vergib den Score."
)

GT_DECOMPOSE_PROMPT = (
    "Du bist ein präziser Annotator für Prüfungs-Musterlösungen. Zerlege die folgende MUSTERLÖSUNG (Ground Truth) in atomare faktische Aussagen. Eine atomare Aussage:\n"
    "- enthält genau EINEN bewertungsrelevanten Inhalt (Definition, Eigenschaft, Rechenschritt, Zwischenergebnis, Endaussage),\n"
    "- ist ohne Kenntnis der restlichen Antwort verständlich (selbsttragend),\n"
    "- enthält keine Konjunktionen oder mehrere unabhängige Fakten,\n"
    "- formaler/mathematischer Inhalt bleibt erhalten (LaTeX, Variablen, Werte).\n"
    "\n"
    "WICHTIG: Reine Floskeln, Wiederholungen oder Einleitungen werden NICHT extrahiert. Wenn die Musterlösung keine extrahierbaren Aussagen enthält, gib eine leere Liste zurück."
)

GT_VERIFY_PROMPT = (
    "Du bist ein strenger Korrektor. Für jede AUSSAGE aus der MUSTERLÖSUNG prüfst du, ob sie in der KANDIDATEN-ANTWORT korrekt vorhanden ist.\n"
    "\n"
    "REGELN PRO AUSSAGE:\n"
    "- TRUE, wenn die Kandidatenantwort denselben Inhalt enthält - sinngemäß oder wörtlich. Andere Reihenfolge, andere Formulierung, andere Notation für dieselbe mathematische Aussage zählen als TRUE.\n"
    "- TRUE auch, wenn die Kandidatenantwort die Aussage IMPLIZIT enthält, weil sie logisch zwingend aus einer expliziten Aussage in der Antwort folgt (z.B. die Musterlösung verlangt das Endergebnis '506', und die Antwort enthält die Rechnung '256 + 240 + 10' explizit ausgeführt).\n"
    "- FALSE, wenn die Antwort die Aussage nicht enthält ODER eine fachlich widersprechende Aussage enthält (z.B. anderes Endergebnis, falsche Reihenfolge der Schritte, vertauschte Variablen).\n"
    "- Bewerte NICHT die Länge der Antwort. Eine kompakte Antwort, die alle GT-Aussagen abdeckt, ist gleichwertig zu einer ausführlicheren.\n"
    "- Zusätzliche Inhalte in der Antwort, die NICHT in der Musterlösung stehen, beeinflussen die Bewertung nicht (kein Abzug, kein Bonus) - das ist der Faithfulness-Job, nicht der Correctness-Job.\n"
    "\n"
    "VORGEHEN: Gehe die GT-Aussagen in der gegebenen Reihenfolge durch. Liefere für jede einen Bool-Wert in derselben Reihenfolge, plus eine kurze Gesamt-Begründung."
)

DECOMPOSE_PROMPT = (
    "Du bist ein präziser Annotator. Zerlege die folgende ANTWORT in atomare faktische Aussagen. Eine atomare Aussage:\n"
    "- enthält genau EINEN faktischen Inhalt (eine Definition, eine Formel, ein Schritt, eine Eigenschaft),\n"
    "- ist ohne Kenntnis der Antwort verständlich (selbsttragend),\n"
    "- enthält keine Konjunktionen oder mehrere unabhängige Fakten,\n"
    "- formaler/mathematischer Inhalt bleibt erhalten (LaTeX, Variablen).\n"
    "\n"
    "WICHTIG: Reine Floskeln, Einleitungen oder Wiederholungen werden NICHT als Aussagen extrahiert. Wenn die Antwort keine extrahierbaren Aussagen enthält, gib eine leere Liste zurück."
)

VERIFY_PROMPT = (
    "Du bist ein sehr strenger Faktenprüfer. Für jede AUSSAGE in der Liste prüfst du, ob sie LOGISCH AUS DEM KONTEXT (= Skript-Quelltext UND/ODER Aufgabenstellung) ABLEITBAR ist.\n"
    "\n"
    "KATEGORISIERE JEDE AUSSAGE ZUERST:\n"
    "(A) FAKTISCHE BEHAUPTUNG - Definition, Eigenschaft, Tatsachenaussage über ein Konzept im Skript.\n"
    "(B) RECHNERISCHER ZWISCHENSCHRITT - algebraische Umformung, Einsetzen in eine Formel, arithmetische Berechnung als Teil einer Anwendungsaufgabe.\n"
    "(C) SETUP-WIEDERHOLUNG - wörtliche oder paraphrasierte Wiedergabe von Daten, Variablen, Wertetabellen, Zahlen, oder Parametern, die DIREKT IN DER AUFGABENSTELLUNG (FRAGE) gegeben sind (z.B. 'Die Funktion hat Variablen X, Y, Z', 'Server hat 16 GB RAM', 'die gegebenen Minterme sind …').\n"
    "\n"
    "REGELN PRO KATEGORIE:\n"
    "- (A) faktisch: TRUE nur wenn der SKRIPT-KONTEXT die Aussage direkt belegt oder sie notwendig daraus folgt. Allgemeines Lehrbuchwissen, das weder im Skript-Kontext noch in der Frage steht, ist FALSE - die Faithfulness misst Bindung an Quelltext und Aufgabenstellung.\n"
    "- (B) Rechnung: TRUE, wenn der Schritt aus zuvor belegten Fakten/Formeln des Skript-Kontexts oder Setup-Werten aus der Frage mit Standardarithmetik korrekt folgt. Der Schritt muss nicht wörtlich im Skript stehen - nur formal aus Skript-Inhalt oder Frage-Setup herleitbar sein. FALSE, wenn die zugrunde liegende Formel/Methode selbst nicht im Skript steht oder die Rechnung sachlich falsch ist.\n"
    "- (C) Setup-Wiederholung: TRUE, sobald die wiedergegebenen Daten in der AUFGABENSTELLUNG explizit genannt wurden. Solche Aussagen sind per Konstruktion kontextuell belegt - sie zu bestrafen würde Transfer-/Anwendungsaufgaben systematisch benachteiligen. FALSE nur, wenn die Wiedergabe die Frage-Daten *verfälscht* (falsche Zahl, falsche Variable).\n"
    "- Widersprüche zum Skript ODER zur Aufgabenstellung sind in allen Kategorien FALSE.\n"
    "- Bei mathematischen Aussagen: gleiche Bedeutung trotz Notationsunterschied gilt als TRUE.\n"
    "\n"
    "VORGEHEN: Gehe die Aussagen in genau der gegebenen Reihenfolge durch. Liefere für jede:\n"
    "1. categories[i] - der Buchstabe 'A', 'B' oder 'C' für die Kategorie der i-ten Aussage,\n"
    "2. verdicts[i] - der Bool-Wert (TRUE/FALSE) zur kontextuellen Belegtheit der i-ten Aussage,\n"
    "3. reasoning - eine kurze Gesamt-Begründung als Freitext.\n"
    "Beide Listen müssen genau len(AUSSAGEN) Einträge enthalten und in derselben Reihenfolge wie die Aussagen-Liste sein."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Post-hoc-Analysen - Retroaktive Claim-Kategorisierung (A/B/C) und ihre Validierung
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORIZE_PROMPT = (
    "Du bist ein präziser Annotator. Klassifiziere die folgende AUSSAGE in genau eine von drei Kategorien - auf Basis der angegebenen AUFGABENSTELLUNG und der erwarteten Rolle der Aussage in einer Klausurantwort.\n"
    "\n"
    "KATEGORIEN:\n"
    "(A) FAKTISCHE BEHAUPTUNG - Definition, Eigenschaft, Tatsachenaussage über ein Konzept aus dem Lehrstoff. Beispiel: 'Ein Halbaddierer addiert zwei Bits und liefert Summe und Übertrag.'\n"
    "(B) RECHNERISCHER ZWISCHENSCHRITT - algebraische Umformung, Einsetzen in eine Formel, arithmetische Berechnung im Rahmen einer Anwendungsaufgabe. Beispiel: '256 + 240 + 10 = 506.'\n"
    "(C) SETUP-WIEDERHOLUNG - wörtliche oder paraphrasierte Wiedergabe von Daten/Variablen/Werten, die in der AUFGABENSTELLUNG selbst angegeben sind (z.B. die in der Frage genannten Minterme, Wertetabellen, Variablen-Namen, Server-Eckdaten).\n"
    "\n"
    "REGELN:\n"
    "- Wähle exakt eine Kategorie.\n"
    "- Wenn die Aussage Setup-Daten aus der Frage wiederholt, klassifiziere C - auch wenn sie zusätzlich faktisch korrekt wäre.\n"
    "- Reine Floskeln/Einleitungen sind hier nicht vorgesehen. Sie wurden bereits beim Claim-Extract gefiltert.\n"
    "- Bei Unsicherheit zwischen A und B: ist die Aussage primär eine Tatsachenbehauptung über das Konzept (A) oder ein Rechenschritt (B)?"
)
