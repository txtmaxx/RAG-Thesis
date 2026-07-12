# RAG-Pipeline zur Generierung faktentreuer Prüfungsfragen

Bachelorarbeit B.Sc. Informatik · Provadis Hochschule
Max Kail · 2026

Diese Pipeline liest ein PDF-Vorlesungsskript ein. Daraus erzeugt sie automatisch
Prüfungsfragen mit Musterlösungen. Anschließend vergleicht sie zwei Antwortsysteme
quantitativ miteinander. Das erste System ist ein reines Sprachmodell ohne
Dokumentenzugriff (Baseline). Das zweite System ist eine Variante mit Retrieval
Augmented Generation (RAG). Das Vorgehen folgt dem Paradigma des Design Science
Research.

## Forschungsfrage

> Wie muss eine RAG-Softwarearchitektur konzipiert sein, um aus unstrukturierten
> Vorlesungsskripten automatisch faktentreue Prüfungsfragen zu generieren, und bei
> welchen Fragetypen lässt sich die Faktentreue der generierten Antworten
> signifikant steigern?

## Hypothesen

Drei Hypothesen werden statistisch geprüft:

- **H1:** Die Bereitstellung von Kontext durch das RAG-System führt zu einer
  signifikant höheren Korrektheit der generierten Antworten als die Nutzung eines
  Basis-Sprachmodells ohne Dokumentenzugriff.
- **H2:** Die Faktentreue ist bei Definitionsfragen signifikant höher als bei
  Transfer- und Anwendungsfragen.
- **H3:** Eine semantische Vorverarbeitung der Skripte (LLM-Bereinigung,
  Bildanalyse, strukturbasiertes Chunking) führt zu einer signifikant höheren
  Faktentreue als die Verwendung von unverarbeitetem Rohtext.

## Ergebnis in Kürze

In der finalen Iteration (n = 60 Fragen, balanciert 20/20/20 über die drei
Fragetypen) konnte **keine der drei Hypothesen** statistisch bestätigt werden.
RAG und Baseline erreichen praktisch identische Korrektheits-Mittelwerte
(M = 0,852 vs. 0,854; p = 0,746). Die Verlässlichkeit der Bewertung ist zweifach
abgesichert. Die Position-Swap-Doppelbewertung des Judges erreicht ein quadratisch
gewichtetes Cohens κ von 0,847 (180 Bewertungspaare). Die manuelle Validierung auf
18 Items zeigt Spearman-Korrelationen zwischen 0,918 und 0,989 (MAE höchstens 0,086).
 
## Aufbau des Projekts

```
rag_thesis/                 Pipeline-Paket
├─ __init__.py              Paket-Marker und Version
├─ config.py                Zentrale Konfiguration (per .env überschreibbar)
├─ prompts.py               Zentrale Sammlung ALLER LLM-Prompts (Prompt Engineering)
├─ llm_client.py            OpenAI-Wrapper (Token-Bucket, Retry, Per-Call-Log)
├─ io_utils.py              JSON-IO, Checkpoint-Resume, Logging
├─ parallel.py              Threadpool-Runner mit Checkpoint-Persistenz
├─ text_utils.py            String-Coercion, LaTeX- und Whitespace-Normalisierung
├─ chunking.py              Semantisches und rohes Chunking (rein, testbar)
├─ stats_utils.py           Bootstrap-CIs, Effektstärken, Cohens κ, Wilson-CI
├─ s1_ingestion.py          Schritt 1: PDF in Vektordatenbank
├─ s2_ground_truth.py       Schritt 2: Frage-Antwort-Generierung
├─ s3_baseline.py           Schritt 3: Inferenz ohne Kontext
├─ s4_rag_inference.py      Schritt 4: Inferenz mit Retrieval
├─ s5_evaluation.py         Schritt 5: LLM-as-a-Judge (proportional und Likert)
├─ s6_analysis.py           Schritt 6: Hypothesentests und Plots
├─ s7_extract_categories.py Post-hoc-Analyse A: Claim-Kategorisierung A/B/C
├─ s8_validate_categories.py Post-hoc-Analyse B: Validierung der Kategorisierung
└─ cli.py                   Top-Level-Befehl (rag-pipeline)

tests/                      Unit-Tests (pytest)
data/                       Eingaben. Das Vorlesungsskript hier ablegen.
outputs/                    Ergebnisse pro Lauf
├─ 0_orchestrator/          Pipeline-Log und Per-Call-Audit
├─ 1_ingestion/             Chunks, Bilder, ChromaDB
├─ 2_ground_truth/          Golden Dataset (Frage-Antwort-Paare)
├─ 3_baseline/              Antworten ohne Retrieval
├─ 4_rag/                   Antworten mit Retrieval (semantisch und roh)
├─ 5_evaluation/            Judge-Bewertungen und Stichproben-CSV
└─ 6_analysis/              Statistikreport und Plots
```

Die Schritte 1 bis 6 bilden die Pipeline. Sie beantworten die Forschungsfrage und
prüfen die drei Hypothesen.

Die Claim-Kategorisierung und ihre Validierung (`s7`/`s8`) sind nachgelagerte
**Post-hoc-Analysen**, kein Teil der Pipeline. Sie laufen nur mit
`rag-pipeline --with-posthoc` oder als eigenständige Module.

Jeder Schritt ist zusätzlich einzeln ausführbar.

## Einrichtung

### 1. Python-Umgebung

Benötigt wird Python 3.9 oder neuer. Empfohlen ist ein virtuelles Environment:

```bash
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install -e ".[dev]"
```

Die Option `-e` installiert das Paket im Editier-Modus. Änderungen am Code wirken
dann sofort. Der Zusatz `[dev]` installiert zusätzlich pytest.

### 2. API-Key konfigurieren

```bash
cp .env.example .env
# .env öffnen und OPENAI_API_KEY eintragen
```

### 3. PDF bereitstellen

Lege das Vorlesungsskript unter `data/vorlesung.pdf` ab. Alternativ passt du den
Namen über `PDF_FILENAME` in der `.env` an.

## Verwendung

### Komplette Pipeline (Schritte 1 bis 6)

```bash
# Top-Level-Befehl nach pip install -e . (Schritte 1-6)
rag-pipeline --samples 60

# zusätzlich die Post-hoc-Analysen (Kategorisierung + Validierung):
rag-pipeline --samples 60 --with-posthoc

# oder ohne Installation direkt aus dem Repo:
python -m rag_thesis.cli --samples 60
```

Die wichtigsten Flags:

| Flag             | Wirkung                                                                      |
| ---------------- | ---------------------------------------------------------------------------- |
| `--samples N`    | Anzahl der Ground-Truth-Fragen. Default ist 60 (balanciert mit je 20 Definitions-, Anwendungs- und Transferfragen). |
| `--keep-general` | Behält auch Fragen mit `requires_context=False` (für die Sensitivitätsanalyse). |
| `--skip-review`  | Überspringt den manuellen Stopp nach Schritt 2.                              |
| `--from-step N`  | Startet ab Schritt N. Erlaubt sind 1 bis 6. Default ist 1.                   |
| `--to-step N`    | Beendet nach Schritt N. Erlaubt sind 1 bis 6. Default ist 6.                 |
| `--with-posthoc` | Führt nach der Pipeline die Post-hoc-Analysen aus (Kategorisierung + Validierung). |

Die wichtigsten Umgebungsvariablen (in `.env` oder als Shell-Variable):

| Variable                 | Wirkung                                                                       |
| ------------------------ | ----------------------------------------------------------------------------- |
| `OPENAI_API_KEY`         | Pflicht. Ohne Key bricht der erste API-Aufruf ab.                             |
| `PDF_FILENAME`           | Name des Eingabe-PDF in `data/`. Default ist `vorlesung.pdf`.                 |
| `WINDOW_SIZE`            | Chunks pro Fenster bei der Generierung. Default ist 7.                        |
| `WINDOW_STRIDE`          | Schrittweite zwischen den Fenstern. Default ist 3.                            |
| `TOP_K`                  | Anzahl der Treffer aus ChromaDB pro RAG-Anfrage. Default ist 8.              |
| `FOOTER_CUTOFF_RATIO`    | Anteil am unteren Seitenrand, der vor der Verarbeitung abgeschnitten wird. Default ist 0,1, das heißt die unteren 10 % der Seite werden entfernt (etwa Fußzeilen und Seitenzahlen). |
| `HEADER_CUTOFF_RATIO`    | Anteil am oberen Seitenrand, der abgeschnitten wird. Default ist 0,0, das heißt es wird nichts entfernt. |

### Einzelne Schritte

```bash
python -m rag_thesis.s1_ingestion --mode semantic
python -m rag_thesis.s1_ingestion --mode raw
python -m rag_thesis.s2_ground_truth --samples 60
python -m rag_thesis.s3_baseline
python -m rag_thesis.s4_rag_inference --mode semantic
python -m rag_thesis.s4_rag_inference --mode raw
python -m rag_thesis.s5_evaluation
python -m rag_thesis.s6_analysis
python -m rag_thesis.s7_extract_categories
python -m rag_thesis.s8_validate_categories
```

### Tests

```bash
pytest -v
```

### Abbruch und Fortsetzen

Die teuren Schritte (Generierung, Inferenz, Bewertung) schreiben regelmäßig
Checkpoints. Wird ein Lauf unterbrochen, sei es durch Strg+C, einen Absturz oder
ein API-Rate-Limit, geht kein Fortschritt verloren. Starte einfach denselben
Befehl noch einmal. Der Lauf setzt am letzten Checkpoint fort. Bereits
verarbeitete Items werden übersprungen und nicht erneut berechnet.

Das macht lange Läufe robust. Du kannst die Pipeline über Nacht laufen lassen und
nach einem Abbruch am nächsten Tag einfach fortsetzen. Die Checkpoint-Dateien
liegen als `*_checkpoint.json` in den jeweiligen Ausgabe-Ordnern. Nach dem
erfolgreichen Abschluss eines Schritts werden sie automatisch gelöscht.

## Methodisches Vorgehen

Die wichtigsten methodischen Punkte:

1. **Proportionale Korrektheit als Primärmetrik.** Die Musterlösung wird in einzelne
   Aussagen zerlegt. Der Judge zählt, wie viele davon in der Antwort belegt sind. Der
   Score ist der Anteil belegter Aussagen. Das vermeidet den Decken-Effekt der
   1-bis-5-Likert-Skala. Die Likert-Variante läuft sekundär weiter. Nur sie liefert
   über die Position-Swap-Bewertung das Cohens κ.

2. **Proportionale Faktentreue im Ragas-Stil.** Die Antwort wird ebenfalls in
   Aussagen zerlegt. Für jede Aussage wird geprüft, ob sie im PDF-Kontext belegt ist.

3. **Kanonischer Vergleichskontext.** Die Faktentreue wird gegen den unveränderten
   PDF-Text der Quellseite gemessen. Dazu kommen die Bildbeschreibungen derselben
   Seite. So messen beide RAG-Modi gegen dieselbe Informationsbasis.

4. **Filter gegen Skript-Verweise.** Ein Verifier verhindert Fragen wie „Welche
   Beispiele zeigt die Tabelle auf Seite 386?". Klausurfragen müssen ohne das Skript
   verständlich sein.

5. **Balanciertes Sampling 20/20/20.** Definition, Anwendung und Transfer sind gleich
   stark vertreten. Das vermeidet Verzerrungen bei H2.

6. **Inter-Rater-Reliabilität des Judges.** Das quadratisch gewichtete Cohens κ
   stammt aus den beiden Position-Swap-Bewertungen. Es zeigt, wie stabil der Judge
   bewertet.

7. **Validierung gegen manuelle Annotation.** Rund 18 Items
   werden als CSV exportiert. Ein Mensch trägt die `Human …`-Spalten nach; erst dann
   berechnet Schritt 6 die Spearman-Korrelation und den mittleren absoluten Fehler
   zwischen Mensch und Judge und erzeugt die `6_judge_validation_*.png`-Streudiagramme.
   Ohne ausgefüllte Spalten überspringt Schritt 6 diesen Teil automatisch.

8. **Bootstrap-Konfidenzintervalle (95 %)** für alle Mittelwerte und gepaarten
   Differenzen. Es werden 2000 Wiederholungen gezogen.

9. **Bonferroni-Korrektur** für die drei Hypothesen. Das Niveau ist α′ = 0,05 / 3 ≈
   0,0167.

10. **Post-hoc- und Robustheits-Analysen.** Schritt 6 erzeugt zusätzlich
    `6_interaction_analysis.txt`: Friedman-Test und Scheirer-Ray-Hare (Interaktion
    System × Fragetyp), Per-Typ-Paartests, eine Power-Analyse für H1 und H3 sowie die
    Robustheit gegen alternative Korrekturen (Holm, Benjamini-Hochberg) und die
    Spearman-Konfidenzintervalle der Judge-Validierung. Separat ausführbar über
    `python -m rag_thesis.interaction_analysis`.

11. **Per-Call-Audit.** Jeder OpenAI-Aufruf wird in `outputs/0_orchestrator/api_calls.jsonl`
    protokolliert. So sind Kosten und Laufzeit nachvollziehbar.

12. **Reproduzierbarkeit.** Alle Modelle, Seeds und Hyperparameter stehen zentral in
    `config.py`. Der Seed 42 wird an OpenAI mitgegeben. Die Schnittstelle behandelt
    ihn als Best-Effort-Maßnahme. Voller Determinismus ist nicht garantiert.

## Eingesetzte Modelle

| Rolle              | Modell                   |
| ------------------ | ------------------------ |
| Baseline und RAG   | `gpt-4o-mini`            |
| Bildanalyse        | `gpt-4o-mini`            |
| Ground Truth       | `gpt-4o`                 |
| Judge (Bewertung)  | `gpt-4o`                 |
| Embeddings         | `text-embedding-3-small` |

## Ergebnisdateien

Nach einem vollständigen Lauf liegen unter `outputs/`:

| Pfad                                              | Inhalt                                               |
| ------------------------------------------------- | ---------------------------------------------------- |
| `0_orchestrator/0_orchestrator.log`               | Gesamter Pipeline-Lauf                               |
| `0_orchestrator/api_calls.jsonl`                  | Per-Call-Audit aller OpenAI-Aufrufe                  |
| `1_ingestion/1_pdf_ingestion_semantic.json`       | Bereinigte semantische Chunks                        |
| `1_ingestion/1_pdf_ingestion_raw.json`            | Rohtext-Chunks (Vergleichsbasis)                     |
| `1_ingestion/1_pdf_pages_raw.json`                | Original-PDF-Text pro Seite (kanonisch)              |
| `1_ingestion/chroma_db/`                          | Persistente Vektor-Collections                       |
| `1_ingestion/images/`                             | Analysierte Folien-Bilder                            |
| `2_ground_truth/2_golden_dataset.json`            | Ground-Truth-Fragen mit Musterlösungen               |
| `3_baseline/3_baseline_answers.json`              | Antworten ohne Retrieval                             |
| `4_rag/4_rag_answers_semantic.json`               | RAG-Antworten im semantischen Modus                  |
| `4_rag/4_rag_answers_raw.json`                    | RAG-Antworten im Rohtext-Modus                       |
| `5_evaluation/5_evaluation_results.json`          | Judge-Bewertungen (Korrektheit und Faktentreue)      |
| `5_evaluation/5_manual_review_sample.csv`         | Stichprobe für die manuelle Judge-Validierung        |
| `6_analysis/6_statistical_report.txt`             | Vollständiger Hypothesen-Bericht                     |
| `6_analysis/6_interaction_analysis.txt`           | Post-hoc-Analysen (Friedman, Scheirer-Ray-Hare, Power, Holm/BH, Spearman-CIs) |
| `6_analysis/6_h1_correctness_boxplot.png`         | Boxplot zu H1                                        |
| `6_analysis/6_h2_faithfulness_by_type.png`        | Balkendiagramm zu H2                                 |
| `6_analysis/6_h3_faithfulness_boxplot.png`        | Boxplot zu H3                                        |
| `6_analysis/6_judge_validation_*.png`             | Streudiagramme Mensch gegen Judge (nur wenn die manuelle Stichprobe `5_manual_review_sample.csv` ausgefüllt ist) |

Die Post-hoc-Analysen (`s7`/`s8`) erzeugen zusätzlich:

| Pfad                                                | Inhalt                                              |
| --------------------------------------------------- | --------------------------------------------------- |
| `5_evaluation/5_categories_retroactive.json`        | A/B/C-Kategorie je Claim (s7)                        |
| `5_evaluation/5_categories_validation_sample.csv`   | 30er-Validierungsstichprobe (s8)                    |
| `6_analysis/6_category_distribution.txt`            | Kategorie-Verteilung nach Fragetyp mit Wilson-CIs   |
| `6_analysis/6_category_distribution.png`            | Balkendiagramm der Kategorie-Verteilung (s7)        |
| `6_analysis/6_category_validation.txt`              | Validierungs-Report (Accuracy und κ)                |

## Übertragbarkeit auf andere Fächer

Pipeline und Prompts wurden an einem Skript der **Technischen Informatik**
entwickelt und evaluiert. Insbesondere die Prompts in `rag_thesis/prompts.py`
sind **fachspezifisch**: Sie enthalten fest verdrahtete Korrektheits-Regeln für
digitale Logik und Rechnerarchitektur (KV-Diagramme/Hamming-Abstand,
Wahrheitstabellen, Boolesche Algebra, Binär-Dezimal-Umrechnung, Modulo-N-Zähler,
Flipflops/FSM, Volladdierer, MIPS-Bit-Felder).

Wer die Pipeline auf ein **anderes Fach** (oder Teilgebiet) anwenden möchte, muss
diese Anteile anpassen — vor allem die Generierungs- und Verifikations-Guardrails
(`TABLE_GUARDRAILS`, `GROUND_TRUTH_*`, `*_VERIFY_*`) in `prompts.py` sowie ggf. die
Fragetypen (Definition/Anwendung/Transfer) und deren Anteile.

Der übrige Code (Ingestion, Chunking, Retrieval, Evaluation, Statistik) ist
fachneutral und bleibt unverändert nutzbar.

## Lizenz

MIT. Siehe `LICENSE`.
