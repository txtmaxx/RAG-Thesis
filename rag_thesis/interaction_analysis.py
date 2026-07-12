"""Post-hoc- und Robustheits-Analysen (ergänzt Schritt 6).

Diese Analysen sind exploratorisch, nicht konfirmatorisch. Sie werden nach
Vorliegen aller Daten gerechnet und sichern die drei vorab registrierten
Hypothesen (H1/H2/H3) gegen alternative Lesarten und Methodenwahlen ab:

1. Friedman-Test über die drei Systeme (gepaart, gesamt und je Fragetyp).
2. Scheirer-Ray-Hare: nichtparametrische 2-Faktor-ANOVA auf Rängen
   (System x Fragetyp), prüft die Interaktion.
3. Per-Type-Paartests (Wilcoxon, RAG Semantic > Baseline je Fragetyp).
4. Post-hoc-Power-Analyse für H1 und H3 inkl. minimal detektierbarer Effektstärke.
5. Alternative Multiple-Testing-Korrekturen (Holm, Benjamini-Hochberg) als
   Robustheits-Check gegenüber dem konservativen Bonferroni.
6. Konfidenzintervalle der Spearman-Judge-Validierung (Fisher-z und Bootstrap).

Schreibt outputs/6_analysis/6_interaction_analysis.txt. Reine Statistik auf
bereits vorhandenen Daten, kein erneuter Pipeline-Lauf und keine API-Aufrufe.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, cast

import numpy as np
from scipy import stats

from . import config
from .io_utils import load_json, rel, setup_logger
from .stats_utils import (
    achieved_power_paired_t,
    benjamini_hochberg,
    bonferroni,
    fisher_z_ci,
    holm_bonferroni,
    mdes_paired_t,
    scheirer_ray_hare,
)

ALPHA = 0.05
BONFERRONI_N = 3
ALPHA_CORR = ALPHA / BONFERRONI_N
BOOT_N = 2000
_EPS = 1e-9
_TYPES = ("Definition", "Anwendung", "Transfer")
REPORT_FILE = config.DIR_ANALYSIS / "6_interaction_analysis.txt"


class _StatResult(Protocol):
    """Form der scipy.stats-Ergebnisobjekte (WilcoxonResult, SignificanceResult).

    Die zu scipy 1.13.1 gelieferten Stubs typisieren den Rückgabewert nur als
    anonyme Klasse, sodass Pyright `.statistic`/`.pvalue` nicht auflöst. Über
    diesen Protocol-`cast` erhält der Type-Checker die tatsächliche, zur
    Laufzeit vorhandene Struktur zurück – inkl. weiterhin aktiver Tippfehler-
    Prüfung auf den Attributnamen.
    """

    statistic: float
    pvalue: float


def _pvalue(result: object) -> float:
    return float(cast(_StatResult, result).pvalue)


def _statistic(result: object) -> float:
    return float(cast(_StatResult, result).statistic)


# ─── Datenextraktion ──────────────────────────────────────────────────────────

def _score(entry: Dict, system: str, metric: str) -> Optional[float]:
    """Hole den (proportionalen) Score eines Systems für eine Metrik."""
    sub = entry.get(system, {}) or {}
    nested = sub.get(metric)
    if isinstance(nested, dict):
        v = nested.get("score")
        return float(v) if isinstance(v, (int, float)) else None
    return None


def _rows(data: List[Dict]) -> List[Dict]:
    """Verflache die Evaluations-JSON auf die für die Analyse nötigen Felder."""
    out = []
    for e in data:
        out.append({
            "qtype": e.get("question_type"),
            "base_corr": _score(e, "baseline", "correctness"),
            "sem_corr": _score(e, "rag_semantic", "correctness"),
            "raw_corr": _score(e, "rag_raw", "correctness"),
            "sem_faith": _score(e, "rag_semantic", "faithfulness"),
            "raw_faith": _score(e, "rag_raw", "faithfulness"),
        })
    return out


def _paired(rows: List[Dict], key_a: str, key_b: str) -> Tuple[np.ndarray, np.ndarray]:
    """Ausgerichtete Wertepaare zweier Spalten, nur wo beide vorhanden sind."""
    a, b = [], []
    for r in rows:
        if r[key_a] is not None and r[key_b] is not None:
            a.append(r[key_a])
            b.append(r[key_b])
    return np.array(a, dtype=float), np.array(b, dtype=float)


# ─── (1) Friedman ─────────────────────────────────────────────────────────────

def _friedman_block(rows: List[Dict]) -> List[str]:
    out = ["(1) FRIEDMAN-TEST  - Correctness über 3 Systeme, gepaart pro Item"]

    def run(subset: List[Dict], label: str) -> None:
        trip = [(r["base_corr"], r["sem_corr"], r["raw_corr"]) for r in subset
                if None not in (r["base_corr"], r["sem_corr"], r["raw_corr"])]
        n = len(trip)
        if n < 3:
            out.append(f"    {label} (n={n}): zu wenige Items für den Test")
            return
        b, s, rw = zip(*trip)
        res = stats.friedmanchisquare(b, s, rw)
        w = res.statistic / (n * (3 - 1))  # Kendall's W
        out.append(f"    {label:11s} (n={n}): chi^2 = {res.statistic:.3f}, df=2, "
                   f"p = {res.pvalue:.4f}, Kendall W = {w:.3f}")

    run(rows, "Gesamt")
    for t in _TYPES:
        run([r for r in rows if r["qtype"] == t], t)
    return out


# ─── (2) Scheirer-Ray-Hare ────────────────────────────────────────────────────

def _srh_block(rows: List[Dict]) -> List[str]:
    vals, sysf, typf = [], [], []
    systems = (("Baseline", "base_corr"), ("RAG_Sem", "sem_corr"), ("RAG_Raw", "raw_corr"))
    for r in rows:
        if r["qtype"] is None:
            continue
        for name, key in systems:
            if r[key] is not None:
                vals.append(r[key])
                sysf.append(name)
                typf.append(r["qtype"])
    out = ["", "(2) SCHEIRER-RAY-HARE  - nichtparametrische 2-Faktor-ANOVA auf Rängen",
           f"    Faktoren: System (3) x Fragetyp (3), N = {len(vals)} Beobachtungen"]
    if len(vals) < 9:
        out.append("    zu wenige Beobachtungen für den Test")
        return out
    res = scheirer_ray_hare(vals, sysf, typf)
    h, df, p = res["b"]
    out.append(f"    Haupteffekt Fragetyp:    H = {h:.3f}, df = {df}, p = {p:.4f}")
    h, df, p = res["a"]
    out.append(f"    Haupteffekt System:      H = {h:.3f}, df = {df}, p = {p:.4f}")
    h, df, p = res["ab"]
    out.append(f"    Interaktion System x Typ: H = {h:.3f}, df = {df}, p = {p:.4f}")
    return out


# ─── (3) Per-Type-Paartests ───────────────────────────────────────────────────

def _per_type_block(rows: List[Dict]) -> List[str]:
    out = ["", "(3) PER-TYPE-PAARTESTS  - Wilcoxon (gepaart, einseitig RAG_Sem > Baseline)"]
    for t in _TYPES:
        sem, base = _paired([r for r in rows if r["qtype"] == t], "sem_corr", "base_corr")
        n = sem.size
        if n < 3:
            out.append(f"    {t}: zu wenige Items (n={n})")
            continue
        diff = sem - base
        ties = int(np.sum(np.abs(diff) <= _EPS))
        sd = float(diff.std(ddof=1))
        dz = float(diff.mean() / sd) if sd > 0 else 0.0
        try:
            p = _pvalue(stats.wilcoxon(sem, base, alternative="greater",
                                       zero_method="wilcox"))
        except ValueError:
            p = float("nan")
        out.append(f"    {t:11s}: Baseline M={base.mean():.3f} | RAG_Sem M={sem.mean():.3f} | "
                   f"d_z={dz:+.3f} | Ties={ties} | p(1-seitig)={p:.4f}")
    return out


# ─── (4) Post-hoc-Power ───────────────────────────────────────────────────────

def _power_block(rows: List[Dict]) -> List[str]:
    out = ["", f"(4) POST-HOC POWER-ANALYSE  (alpha' = {ALPHA_CORR:.4f} Bonferroni, einseitig)"]

    def run(a: np.ndarray, b: np.ndarray, label: str) -> None:
        diff = a - b
        n = diff.size
        ties = int(np.sum(np.abs(diff) <= _EPS))
        nz = diff[np.abs(diff) > _EPS]
        n_eff = int(nz.size)
        sd = float(nz.std(ddof=1)) if n_eff > 1 else 0.0
        dz = float(nz.mean() / sd) if sd > 0 else 0.0
        power = achieved_power_paired_t(dz, n_eff, ALPHA_CORR) if n_eff > 1 else float("nan")
        mdes = mdes_paired_t(n_eff, ALPHA_CORR, 0.8) if n_eff > 1 else float("nan")
        out.append(f"    {label}:")
        out.append(f"         Tie-Rate: {ties}/{n}  |  effektive n (ohne Ties): {n_eff}")
        out.append(f"         beobachtete Effektstärke d_z: {dz:+.3f}")
        out.append(f"         erreichte Power: {power:.3f}")
        out.append(f"         MDES (d_z, 80 % Power): ca. {mdes:.3f}")

    sem_c, base_c = _paired(rows, "sem_corr", "base_corr")
    run(sem_c, base_c, "H1 (RAG_Sem vs Baseline, Correctness)")
    sem_f, raw_f = _paired(rows, "sem_faith", "raw_faith")
    run(sem_f, raw_f, "H3 (RAG_Sem vs RAG_Raw, Faithfulness)")
    return out


# ─── (5) Alternative Multiple-Testing-Korrekturen ─────────────────────────────

def _raw_pvalues(rows: List[Dict]) -> Tuple[float, float, float]:
    """Die drei rohen Hypothesen-p-Werte (gleiche Tests wie in Schritt 6)."""
    sem, base = _paired(rows, "sem_corr", "base_corr")
    p1 = _pvalue(stats.wilcoxon(sem, base, alternative="greater", zero_method="wilcox"))

    d = [r["sem_faith"] for r in rows if r["qtype"] == "Definition" and r["sem_faith"] is not None]
    ta = [r["sem_faith"] for r in rows
          if r["qtype"] in ("Transfer", "Anwendung") and r["sem_faith"] is not None]
    p2 = _pvalue(stats.mannwhitneyu(d, ta, alternative="greater"))

    sem_f, raw_f = _paired(rows, "sem_faith", "raw_faith")
    p3 = _pvalue(stats.wilcoxon(sem_f, raw_f, alternative="greater", zero_method="wilcox"))
    return p1, p2, p3


def _correction_block(rows: List[Dict]) -> List[str]:
    out = ["", "(5) ALTERNATIVE MULTIPLE-TESTING-KORREKTUREN  (drei vorab registrierte H)"]
    raw = list(_raw_pvalues(rows))
    holm = holm_bonferroni(raw)
    bh = benjamini_hochberg(raw)
    for i, name in enumerate(("H1", "H2", "H3")):
        out.append(f"    {name}: p_roh = {raw[i]:.4f}  |  Bonferroni = {bonferroni(raw[i], 3):.4f}"
                   f"  |  Holm = {holm[i]:.4f}  |  Benjamini-Hochberg = {bh[i]:.4f}")
    return out


# ─── (6) Spearman-Konfidenzintervalle (Judge-Validierung) ────────────────────

_CSV_PAIRS = [
    ("Baseline Correctness (LLM)", "Human Baseline Correctness", "Baseline Correctness"),
    ("RAG Sem Correctness (LLM)", "Human Sem Correctness", "RAG Sem Correctness"),
    ("RAG Sem Faithfulness (LLM)", "Human Sem Faithfulness", "RAG Sem Faithfulness"),
    ("RAG Raw Correctness (LLM)", "Human Raw Correctness", "RAG Raw Correctness"),
    ("RAG Raw Faithfulness (LLM)", "Human Raw Faithfulness", "RAG Raw Faithfulness"),
]


def _spearman_block() -> List[str]:
    out = ["", "(6) SPEARMAN 95-%-KONFIDENZINTERVALLE  (Human vs. LLM-Judge)"]
    path = config.FILE_MANUAL_REVIEW_CSV
    if not path.exists():
        out.append("    Manuelle Review-CSV nicht vorhanden - übersprungen.")
        return out
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    rng = np.random.default_rng(config.RANDOM_SEED)
    any_done = False
    for llm_col, hum_col, label in _CSV_PAIRS:
        xs, ys = [], []
        for r in rows:
            try:
                xs.append(float(r[llm_col]))
                ys.append(float(r[hum_col]))
            except (KeyError, ValueError, TypeError):
                continue
        if len(xs) < 4:
            continue
        any_done = True
        x, y = np.array(xs), np.array(ys)
        rho = _statistic(stats.spearmanr(x, y))
        lo, hi = fisher_z_ci(rho, len(x))
        boot = []
        for _ in range(BOOT_N):
            idx = rng.integers(0, len(x), len(x))
            xi, yi = x[idx], y[idx]
            if np.ptp(xi) == 0 or np.ptp(yi) == 0:
                continue  # konstante Resample-Spalte -> Korrelation undefiniert
            r = _statistic(stats.spearmanr(xi, yi))
            if not np.isnan(r):
                boot.append(r)
        if boot:
            b_lo, b_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        else:
            b_lo, b_hi = float("nan"), float("nan")
        out.append(f"    {label}: rho = {rho:.3f}, n = {len(x)}")
        out.append(f"         Fisher-z 95%-CI:  [{lo:.3f}, {hi:.3f}]")
        out.append(f"         Bootstrap 95%-CI: [{b_lo:.3f}, {b_hi:.3f}]")
    if not any_done:
        out.append("    Keine ausgefüllten Human-Spalten gefunden - übersprungen.")
    return out


# ─── Orchestrierung ───────────────────────────────────────────────────────────

def run(input_path: Optional[str] = None) -> None:
    """Berechne alle Post-hoc-Analysen und schreibe den Report."""
    config.ensure_output_dirs()

    in_path = Path(input_path or config.FILE_EVALUATION)
    if not in_path.exists():
        raise FileNotFoundError(f"Evaluations-Datei nicht gefunden: {rel(in_path)}")
    rows = _rows(load_json(in_path))

    lines = ["=" * 72,
             "  INTERAKTIONS- UND POST-HOC-ANALYSE (kein Pipeline-Rerun)",
             f"  Datengrundlage: {rel(in_path)}, n = {len(rows)}",
             "=" * 72, ""]
    lines += _friedman_block(rows)
    lines += _srh_block(rows)
    lines += _per_type_block(rows)
    lines += _power_block(rows)
    lines += _correction_block(rows)
    lines += _spearman_block()

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(f"Interaktionsanalyse geschrieben: {rel(REPORT_FILE)}")
    print("\n".join(lines))


def main() -> None:
    """CLI-Einstieg für die alleinstehende Post-hoc-Analyse."""
    parser = argparse.ArgumentParser(description="Post-hoc- und Robustheits-Analysen.")
    parser.add_argument("--input", type=str, default=str(config.FILE_EVALUATION),
                        help="Pfad zur Evaluations-JSON.")
    args = parser.parse_args()
    config.ensure_output_dirs()
    setup_logger(config.DIR_ANALYSIS / "6_analysis.log")
    run(args.input)


if __name__ == "__main__":
    main()
