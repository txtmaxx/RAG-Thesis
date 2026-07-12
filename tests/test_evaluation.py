"""Tests für Evaluations-Helfer (ohne LLM-Calls)."""

from unittest.mock import patch

from rag_thesis.s5_evaluation import (
    _align_datasets, evaluate_correctness_proportional,
)


def test_align_datasets_drops_unmatched():
    base = [
        {"question": "Q1", "ground_truth": "A1", "baseline_answer": "B1",
         "question_type": "Definition", "source_reference": "chunk_0_to_chunk_4",
         "requires_context": True},
        {"question": "Q2", "ground_truth": "A2", "baseline_answer": "B2",
         "question_type": "Anwendung", "source_reference": "chunk_5_to_chunk_9",
         "requires_context": True},
    ]
    sem = [{"question": "Q1", "rag_answer": "S1", "retrieved_context": "ctx_s1"}]
    raw = [{"question": "Q1", "rag_answer": "R1", "retrieved_context": "ctx_r1"}]
    aligned = _align_datasets(base, sem, raw)
    assert len(aligned) == 1
    assert aligned[0]["question"] == "Q1"
    assert aligned[0]["rag_semantic_answer"] == "S1"
    assert aligned[0]["rag_raw_context"] == "ctx_r1"
    assert aligned[0]["requires_context"] is True


def test_align_datasets_preserves_metadata():
    base = [{"question": "Q", "ground_truth": "GT", "baseline_answer": "B",
             "question_type": "Transfer", "source_reference": "chunk_1_to_chunk_2",
             "requires_context": False}]
    sem = [{"question": "Q", "rag_answer": "S", "retrieved_context": "CS"}]
    raw = [{"question": "Q", "rag_answer": "R", "retrieved_context": "CR"}]
    out = _align_datasets(base, sem, raw)[0]
    assert out["question_type"] == "Transfer"
    assert out["source_reference"] == "chunk_1_to_chunk_2"
    assert out["requires_context"] is False


# ─── Proportionale Correctness: Score-Berechnung ──────────────────────────────
#
# API-Wrapper patch, damit der Test keine OpenAI-Calls braucht.
# Geprüft wird die deterministische Score-Berechnung + Verdict-Aggregation.

def _make_correctness(gt_claims, verdicts):
    """Stub-Helfer: decompose liefert gt_claims, verify liefert verdicts."""
    with patch("rag_thesis.s5_evaluation._decompose_ground_truth", return_value=gt_claims), \
         patch("rag_thesis.s5_evaluation._verify_gt_claims_in_answer", return_value=verdicts):
        return evaluate_correctness_proportional("Q", "GT", "A")


def test_correctness_proportional_perfect_match():
    out = _make_correctness(["c1", "c2", "c3"], [True, True, True])
    assert out["score"] == 1.0
    assert out["n_claims"] == 3
    assert out["n_supported"] == 3


def test_correctness_proportional_partial_match():
    out = _make_correctness(["c1", "c2", "c3", "c4"], [True, False, True, False])
    assert out["score"] == 0.5
    assert out["n_supported"] == 2


def test_correctness_proportional_no_match():
    out = _make_correctness(["c1", "c2"], [False, False])
    assert out["score"] == 0.0


def test_correctness_proportional_empty_groundtruth():
    out = _make_correctness([], [])
    assert out["score"] is None
    assert out["n_claims"] == 0
    assert "extrahierbar" in out["justification"].lower()
