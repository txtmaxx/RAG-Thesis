"""Tests für die Analyse-Pipeline (Schema-Robustheit)."""

import pandas as pd

from rag_thesis.s6_analysis import _extract_score, compute_irr, to_dataframe


def test_extract_score_handles_new_schema():
    entry = {
        "baseline": {"correctness": {"score": 0.75, "raw_scores": [4, 4]}},
        "rag_semantic": {"faithfulness": {"score": 0.92}},
    }
    assert _extract_score(entry, "baseline", "correctness") == 0.75
    assert _extract_score(entry, "rag_semantic", "faithfulness") == 0.92


def test_extract_score_handles_legacy_flat_schema():
    entry = {
        "baseline": {"correctness_score": 0.5},
        "rag_raw": {"faithfulness_score": 0.8},
    }
    assert _extract_score(entry, "baseline", "correctness") == 0.5
    assert _extract_score(entry, "rag_raw", "faithfulness") == 0.8


def test_extract_score_returns_none_on_missing():
    assert _extract_score({}, "baseline", "correctness") is None
    assert _extract_score({"baseline": {}}, "baseline", "correctness") is None


def test_to_dataframe_constructs_columns():
    data = [
        {
            "question_type": "Definition",
            "requires_context": True,
            "baseline":     {"correctness":  {"score": 0.7}},
            "rag_semantic": {"correctness":  {"score": 0.9},
                              "faithfulness": {"score": 1.0}},
            "rag_raw":      {"correctness":  {"score": 0.8},
                              "faithfulness": {"score": 0.9}},
        }
    ]
    df = to_dataframe(data)
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["base_corr"] == 0.7
    assert df.iloc[0]["sem_faith"] == 1.0
    assert df.iloc[0]["raw_faith"] == 0.9


def test_compute_irr_extracts_paired_scores():
    data = [
        {"baseline":     {"correctness": {"raw_scores": [4, 4]}},
         "rag_semantic": {"correctness": {"raw_scores": [5, 5]}},
         "rag_raw":      {"correctness": {"raw_scores": [3, 4]}}},
        {"baseline":     {"correctness": {"raw_scores": [5, 5]}},
         "rag_semantic": {"correctness": {"raw_scores": [4, 5]}},
         "rag_raw":      {"correctness": {"raw_scores": [2, 2]}}},
    ]
    irr = compute_irr(data)
    assert irr["n"] == 6   # 2 Items × 3 Systeme
    assert 0 <= irr["exact_agreement_pct"] <= 100
    assert irr["mean_abs_diff"] >= 0
