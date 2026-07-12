"""Tests für die zentrale Prompt-Sammlung (rag_thesis.prompts).

Schützt die für das Prompt Engineering zentrale Datei vor versehentlichem
Kaputtmachen: Konstanten dürfen nicht leer werden, die Builder müssen
nicht-leere Strings liefern, und die Per-Fragetyp-Dicts müssen genau die drei
erwarteten Schlüssel enthalten. Keine API-Calls.
"""

from rag_thesis import prompts
from rag_thesis import s5_evaluation, s7_extract_categories, s8_validate_categories

QTYPES = {"Definition", "Anwendung", "Transfer"}

# Alle statischen System-Prompt-Konstanten, die nicht leer sein dürfen.
_STATIC_CONSTANTS = [
    "IMG_INSTRUCTION_FULL", "IMG_INSTRUCTION_EMBEDDED", "TABLE_GUARDRAILS",
    "RELEVANCE_CHECK_SYSTEM", "GROUND_TRUTH_VERIFY_SYSTEM",
    "BASELINE_DEFAULT_PROMPT", "RAG_DEFAULT_PROMPT",
    "CORRECTNESS_PROMPT", "GT_DECOMPOSE_PROMPT", "GT_VERIFY_PROMPT",
    "DECOMPOSE_PROMPT", "VERIFY_PROMPT", "CATEGORIZE_PROMPT",
]

_TYPE_DICTS = [
    "GROUND_TRUTH_TYPE_INSTRUCTIONS", "BASELINE_PROMPTS_BY_TYPE", "RAG_PROMPTS_BY_TYPE",
]


def test_static_constants_are_nonempty_strings():
    for name in _STATIC_CONSTANTS:
        val = getattr(prompts, name)
        assert isinstance(val, str), f"{name} ist kein str"
        assert val.strip(), f"{name} ist leer"


def test_per_type_dicts_have_exactly_three_question_types():
    for name in _TYPE_DICTS:
        d = getattr(prompts, name)
        assert set(d) == QTYPES, f"{name} hat Schlüssel {set(d)}, erwartet {QTYPES}"
        for qt, text in d.items():
            assert isinstance(text, str) and text.strip(), f"{name}[{qt}] leer"


def test_ground_truth_generation_builder_embeds_type_instruction():
    for qt in QTYPES:
        sysp = prompts.ground_truth_generation_system(qt)
        assert isinstance(sysp, str) and sysp.strip()
        # Die fragetyp-spezifische Zielvorgabe muss eingebettet sein …
        assert prompts.GROUND_TRUTH_TYPE_INSTRUCTIONS[qt] in sysp
        # … ebenso die festen Regelblöcke und der konkrete Fragetyp.
        assert "KRITISCHE REGELN:" in sysp
        assert qt in sysp


def test_unknown_question_type_falls_back_without_crashing():
    # .get(qtype, '') -> unbekannter Typ liefert weiterhin einen gültigen Prompt.
    sysp = prompts.ground_truth_generation_system("Unbekannt")
    assert "KRITISCHE REGELN:" in sysp


def test_image_analysis_prompt_variants_and_truncation():
    full = prompts.image_analysis_prompt(is_full_page=True, context="Kontext")
    emb = prompts.image_analysis_prompt(is_full_page=False, context="Kontext")
    assert prompts.IMG_INSTRUCTION_FULL in full
    assert prompts.IMG_INSTRUCTION_EMBEDDED in emb
    for p in (full, emb):
        assert "ANTWORT-REGELN:" in p
        assert prompts.TABLE_GUARDRAILS in p
    # Kontext wird auf 1500 Zeichen begrenzt.
    long_ctx = "x" * 5000
    assert ("x" * 1500) in prompts.image_analysis_prompt(is_full_page=True, context=long_ctx)
    assert ("x" * 1501) not in prompts.image_analysis_prompt(is_full_page=True, context=long_ctx)


def test_text_cleaning_prompt_contains_input():
    p = prompts.text_cleaning_prompt("ABC-INPUT")
    assert "ABC-INPUT" in p
    assert "Markdown" in p


def test_step_modules_reference_central_prompts():
    # Die Schritt-Module müssen DENSELBEN Prompt nutzen wie prompts.py (Aliase intakt).
    assert s5_evaluation._CORRECTNESS_PROMPT is prompts.CORRECTNESS_PROMPT
    assert s5_evaluation._VERIFY_PROMPT is prompts.VERIFY_PROMPT
    assert s7_extract_categories._CATEGORIZE_PROMPT is prompts.CATEGORIZE_PROMPT
    assert s8_validate_categories._CATEGORIZE_PROMPT is prompts.CATEGORIZE_PROMPT
