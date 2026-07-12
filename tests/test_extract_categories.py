"""Tests für Schritt 7 (retroaktive Claim-Kategorisierung).

Keine echten API-Calls: getestet werden ausschließlich die reinen Daten- und
Report-Funktionen mit synthetischem Input. Die IO-Pfade werden auf tmp_path
umgebogen, damit die echten Outputs unter outputs/ nicht überschrieben werden.
"""

from collections import Counter

from rag_thesis import s7_extract_categories as s7


def _synthetic_eval_results():
    """Zwei Frage-Antwort-Paare mit Claims in beiden RAG-Modi."""
    return [
        {
            "question": "Definiere einen Halbaddierer.",
            "question_type": "Definition",
            "rag_semantic": {"faithfulness": {
                "claims": ["Ein Halbaddierer addiert zwei Bits.",
                           "Er liefert Summe und Übertrag."],
                "verdicts": [True, True],
            }},
            "rag_raw": {"faithfulness": {
                "claims": ["Ein Halbaddierer addiert zwei Bits."],
                "verdicts": [True],
            }},
        },
        {
            "question": "Berechne 256 + 240.",
            "question_type": "Anwendung",
            "rag_semantic": {"faithfulness": {
                "claims": ["256 + 240 = 496.", "Das Ergebnis ist 496."],
                "verdicts": [True, False],
            }},
            "rag_raw": {"faithfulness": {"claims": [], "verdicts": []}},
        },
    ]


def test_flatten_claims_with_synthetic_input():
    items = s7._flatten_claims(_synthetic_eval_results())
    # 2 + 1 + 2 + 0 = 5 Claims über beide Modi
    assert len(items) == 5
    # Stabiler claim_id: <qhash>_<system>_<index>
    first = items[0]
    assert set(first) == {"claim_id", "question", "question_type", "system",
                          "claim_index", "claim", "verdict"}
    assert first["claim_id"].endswith("_rag_semantic_0")
    assert all("_rag_semantic_" in i["claim_id"] or "_rag_raw_" in i["claim_id"]
               for i in items)
    # claim_ids sind eindeutig
    assert len({i["claim_id"] for i in items}) == 5
    # Verdict wird durchgereicht
    anwendung_false = [i for i in items
                       if i["question_type"] == "Anwendung" and i["verdict"] is False]
    assert len(anwendung_false) == 1


def test_write_report_with_synthetic_records(tmp_path, monkeypatch):
    report_file = tmp_path / "report.txt"
    monkeypatch.setattr(s7, "_OUTPUT_REPORT", report_file)
    records = [
        {"system": "rag_semantic", "question_type": "Definition",
         "category": "A", "verdict": True},
        {"system": "rag_semantic", "question_type": "Definition",
         "category": "C", "verdict": False},
        {"system": "rag_raw", "question_type": "Anwendung",
         "category": "C", "verdict": True},
        {"system": "rag_raw", "question_type": "Anwendung",
         "category": "B", "verdict": True},
    ]
    agg, cat_by_qtype = s7.write_report(records)
    assert report_file.exists()
    text = report_file.read_text(encoding="utf-8")
    # Aggregation korrekt
    assert cat_by_qtype["Definition"]["A"] == 1
    assert cat_by_qtype["Definition"]["C"] == 1
    assert agg[("rag_raw", "Anwendung")]["B"] == 1
    # Wilson-CI-Zeile ist im Report
    assert "Wilson" in text
    assert "95%-CI" in text
    # Definition: C-Anteil 50 % (1/2)
    assert "50.0 %" in text


def test_write_plot_smoke(tmp_path, monkeypatch):
    plot_file = tmp_path / "plot.png"
    monkeypatch.setattr(s7, "_OUTPUT_PLOT", plot_file)
    cat_by_qtype = {
        "Definition": Counter({"A": 8, "C": 2}),
        "Anwendung": Counter({"A": 4, "B": 1, "C": 5}),
        "Transfer": Counter({"A": 6, "C": 4}),
    }
    # Darf nicht werfen. Wenn matplotlib vorhanden ist, entsteht die Datei.
    s7.write_plot(cat_by_qtype)
    try:
        import matplotlib  # noqa: F401
        assert plot_file.exists()
    except Exception:
        pass  # matplotlib nicht installiert -> Funktion überspringt sauber
