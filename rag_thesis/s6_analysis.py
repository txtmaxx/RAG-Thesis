"""Schritt 6 - Statistische Auswertung & Visualisierung.

Liefert:
- Hypothesentests (H1/H2/H3) mit Bonferroni-Korrektur, einseitig.
- Bootstrap-95-%-Konfidenzintervalle für alle Mittelwerte.
- Inter-Rater-Reliabilität des Judges (quadratisch gewichtetes Cohen's κ
  zwischen den beiden Position-Swap-Bewertungen).
- Sensitivitätsanalyse für H1: gestützt auf alle Items vs. nur
  requires_context=True-Items.
- Human↔LLM-Korrelation, sofern die Spalten der manuellen Review-CSV
  ausgefüllt wurden.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .io_utils import load_json, rel, setup_logger
from .stats_utils import (
    bonferroni,
    bootstrap_ci_mean,
    bootstrap_ci_paired_diff,
    cohens_kappa_quadratic_weighted,
    rank_biserial_mannwhitney,
    rank_biserial_wilcoxon,
    shapiro_is_normal,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 300
ALPHA = 0.05
BONFERRONI_N = 3
BOOT_N = 2000
# Fester Seed für die Bootstrap-Ziehungen -> die Konfidenzintervalle sind bei
# jedem Lauf identisch reproduzierbar (vgl. config.RANDOM_SEED).
BOOT_SEED = config.RANDOM_SEED

REPORT_FILE = config.DIR_ANALYSIS / "6_statistical_report.txt"


# ─── Robust score extractor (neue + alte Schema-Variante) ─────────────────────

def _extract_score(entry: Dict, system: str, metric: str) -> Optional[float]:
    """Extrahiere Score robust für altes (flach) und neues (genested) Schema."""
    sys_data = entry.get(system, {}) or {}
    nested = sys_data.get(metric)
    if isinstance(nested, dict):
        v = nested.get("score")
    else:
        v = sys_data.get(f"{metric}_score")
    return float(v) if isinstance(v, (int, float)) else None


def _extract_raw_scores(entry: Dict, system: str, metric: str) -> Optional[List[int]]:
    """Hole rohe Likert-Scores beider Position-Swap-Bewertungen (für IRR)."""
    sys_data = entry.get(system, {}) or {}
    nested = sys_data.get(metric)
    if isinstance(nested, dict):
        raw = nested.get("raw_scores")
        if isinstance(raw, list) and len(raw) >= 2:
            return [int(x) for x in raw[:2]]
    return None


def to_dataframe(data: List[Dict]) -> pd.DataFrame:
    """Verflache die genestete Evaluations-JSON in einen analytischen DataFrame."""
    rows = []
    for entry in data:
        rows.append({
            "question_type": entry.get("question_type"),
            "requires_context": entry.get("requires_context", True),
            "canonical_used": bool(entry.get("canonical_context_used", False)),
            "base_corr": _extract_score(entry, "baseline", "correctness"),
            "sem_corr":  _extract_score(entry, "rag_semantic", "correctness"),
            "raw_corr":  _extract_score(entry, "rag_raw", "correctness"),
            "sem_faith": _extract_score(entry, "rag_semantic", "faithfulness"),
            "raw_faith": _extract_score(entry, "rag_raw", "faithfulness"),
        })
    return pd.DataFrame(rows)


# ─── Plots ────────────────────────────────────────────────────────────────────

def _ci_label(values: pd.Series) -> str:
    """Formatiere Mittelwert + Bootstrap-95-%-CI als kompaktes Label für Plots."""
    arr = values.dropna().to_numpy()
    if arr.size < 2:
        return ""
    lo, hi = bootstrap_ci_mean(arr, n_iter=BOOT_N, seed=BOOT_SEED)
    return f"M={arr.mean():.3f} [95% CI {lo:.3f}, {hi:.3f}]"


def _legend_box_mean_median(ax) -> None:
    """Einheitliche Legende für die Boxplots: erklärt Median-Linie vs. Mittelwert-
    Punkt mit CI. Ohne sie ist nicht ersichtlich, was der schwarze Punkt bedeutet."""
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.6, label="Median"),
            Line2D([0], [0], marker="o", color="black", lw=0, markersize=6,
                   label="Mittelwert ± 95%-CI"),
        ],
        loc="lower center", ncol=2, fontsize=9, framealpha=0.9,
    )


def plot_correctness_boxplot(df: pd.DataFrame) -> None:
    """H1-Visualisierung: Correctness-Verteilungen Baseline vs. RAG-Modi."""
    valid = df[["base_corr", "sem_corr", "raw_corr"]].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        [valid["base_corr"], valid["sem_corr"], valid["raw_corr"]],
        showfliers=False, patch_artist=True,
    )
    colors = ["#94a3b8", "#3b82f6", "#f97316"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.6)
    means = [valid["base_corr"].mean(), valid["sem_corr"].mean(), valid["raw_corr"].mean()]
    cis = [bootstrap_ci_mean(valid[c], n_iter=BOOT_N, seed=BOOT_SEED)
           for c in ["base_corr", "sem_corr", "raw_corr"]]
    for i, (m, (lo, hi)) in enumerate(zip(means, cis), 1):
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o",
                    color="black", capsize=4, zorder=5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Baseline", "RAG (Semantic)", "RAG (Raw)"])
    ax.set_ylabel("Korrektheit (LLM-as-a-Judge)")
    ax.set_title("H1: Korrektheit – Baseline vs. RAG Semantic vs. RAG Raw\n"
                 "(Mittelwert ± Bootstrap-95-%-CI)")
    ax.set_ylim(-0.05, 1.1)
    _legend_box_mean_median(ax)
    plt.tight_layout()
    plt.savefig(config.DIR_ANALYSIS / "6_h1_correctness_boxplot.png", dpi=DPI)
    plt.close()


def plot_faithfulness_by_type(df: pd.DataFrame) -> None:
    """H2-Visualisierung: Faithfulness je Fragetyp (Definition/Anwendung/Transfer)."""
    valid = df.dropna(subset=["sem_faith"])
    types = sorted(valid["question_type"].dropna().unique())
    if not types:
        return
    x = np.arange(len(types))
    width = 0.3

    sem_means, sem_los, sem_his = [], [], []
    raw_means, raw_los, raw_his = [], [], []
    for t in types:
        s = cast(pd.DataFrame, valid[valid["question_type"] == t])["sem_faith"].dropna().to_numpy()
        r = df.dropna(subset=["raw_faith"]).query("question_type == @t")[
            "raw_faith"].dropna().to_numpy()
        sem_means.append(s.mean() if s.size else np.nan)
        raw_means.append(r.mean() if r.size else np.nan)
        s_ci = bootstrap_ci_mean(s, n_iter=BOOT_N, seed=BOOT_SEED) if s.size >= 2 else (np.nan, np.nan)
        r_ci = bootstrap_ci_mean(r, n_iter=BOOT_N, seed=BOOT_SEED) if r.size >= 2 else (np.nan, np.nan)
        sem_los.append(s_ci[0])
        sem_his.append(s_ci[1])
        raw_los.append(r_ci[0])
        raw_his.append(r_ci[1])

    fig, ax = plt.subplots(figsize=(9, 5))
    sem_err = [
        [m - lo if not np.isnan(lo) else 0 for m, lo in zip(sem_means, sem_los)],
        [hi - m if not np.isnan(hi) else 0 for m, hi in zip(sem_means, sem_his)],
    ]
    raw_err = [
        [m - lo if not np.isnan(lo) else 0 for m, lo in zip(raw_means, raw_los)],
        [hi - m if not np.isnan(hi) else 0 for m, hi in zip(raw_means, raw_his)],
    ]
    ax.bar(x - width / 2, sem_means, width, yerr=sem_err, label="RAG Semantic",
           color="#3b82f6", alpha=0.85, capsize=4)
    ax.bar(x + width / 2, raw_means, width, yerr=raw_err, label="RAG Raw",
           color="#f97316", alpha=0.85, capsize=4)
    # Wertelabels ÜBER die obere CI-Kappe setzen (sem_his/raw_his), sonst
    # schneidet der Fehlerbalken-Whisker die Ziffern
    for i, (s, r, sh, rh) in enumerate(zip(sem_means, raw_means, sem_his, raw_his)):
        if not np.isnan(s):
            sy = sh if not np.isnan(sh) else s
            ax.text(i - width / 2, sy + 0.03, f"{s:.2f}", ha="center", va="bottom", fontsize=9)
        if not np.isnan(r):
            ry = rh if not np.isnan(rh) else r
            ax.text(i + width / 2, ry + 0.03, f"{r:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel("Faktentreue (Anteil belegter Aussagen)")
    ax.set_title("H2: Faktentreue nach Fragetyp – RAG Semantic vs. RAG Raw\n"
                 "(Bootstrap-95-%-CI)")
    ax.set_ylim(0, 1.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(config.DIR_ANALYSIS / "6_h2_faithfulness_by_type.png", dpi=DPI)
    plt.close()


def plot_faithfulness_comparison(df: pd.DataFrame) -> None:
    """H3-Visualisierung: Faithfulness RAG Raw vs. RAG Semantic."""
    valid = df[["raw_faith", "sem_faith"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([valid["raw_faith"], valid["sem_faith"]],
                     showfliers=False, patch_artist=True)
    colors = ["#f97316", "#3b82f6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.6)
    means = [valid["raw_faith"].mean(), valid["sem_faith"].mean()]
    cis = [bootstrap_ci_mean(valid[c], n_iter=BOOT_N, seed=BOOT_SEED)
           for c in ["raw_faith", "sem_faith"]]
    for i, (m, (lo, hi)) in enumerate(zip(means, cis), 1):
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o",
                    color="black", capsize=4, zorder=5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["RAG (Raw)", "RAG (Semantic)"])
    ax.set_ylabel("Faktentreue (Anteil belegter Aussagen)")
    ax.set_title("H3: Faktentreue – RAG Raw vs. RAG Semantic\n"
                 "(Mittelwert ± Bootstrap-95-%-CI)")
    ax.set_ylim(-0.05, 1.1)
    _legend_box_mean_median(ax)
    plt.tight_layout()
    plt.savefig(config.DIR_ANALYSIS / "6_h3_faithfulness_boxplot.png", dpi=DPI)
    plt.close()


def plot_human_vs_llm(human: pd.DataFrame, llm: pd.DataFrame, *,
                       metric: str, system_label: str,
                       rho: Optional[float] = None, mae: Optional[float] = None,
                       n: Optional[int] = None) -> None:
    """Scatter zur Validierung: Human-Score vs. LLM-Judge-Score je Item.

    Blendet Spearman ρ / MAE / n direkt ins Bild ein, damit die 
    Abbildung in der Arbeit ohne Begleittext lesbar ist.
    """
    if human.empty or llm.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(llm[metric], human[metric], alpha=0.75, color="#3b82f6",
               s=70, edgecolors="white", linewidths=0.6, zorder=3)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="ideal")
    ax.set_xlabel(f"LLM-Judge-Score ({system_label})")
    ax.set_ylabel(f"Human-Score ({system_label})")
    ax.set_title(f"Validierung des LLM-Judges: {system_label}\n"
                 f"(Korrelation als Validitätsindikator)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    if rho is not None:
        stats_txt = f"Spearman ρ = {rho:.3f}"
        if mae is not None:
            stats_txt += f"\nMAE = {mae:.3f}"
        if n is not None:
            stats_txt += f"\nn = {n}"
        ax.text(0.97, 0.05, stats_txt, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7",
                          alpha=0.9))
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(config.DIR_ANALYSIS / f"6_judge_validation_{system_label.lower().replace(' ', '_')}.png",
                dpi=DPI)
    plt.close()


# ─── IRR ──────────────────────────────────────────────────────────────────────

def compute_irr(data: List[Dict]) -> Dict[str, float]:
    """Inter-Rater-Reliabilität zwischen den beiden Judge-Position-Swaps.

    Berechnet quadratisch gewichtetes Cohen's κ (ordinal angemessen) und
    mittlere absolute Differenz |r1 − r2|. Aggregiert über alle Systeme,
    weil dieselbe Judge-Konfiguration alle bewertet.

    Greift auf das correctness_likert-Feld zu (Position-Swap-Doppelbewertung
    mit rohen 1–5-Scores). Das Primärfeld correctness enthält den
    proportionalen Score ohne Position-Swap und damit keine
    raw_scores. Älteres Schema (nur correctness mit Likert-Scores) wird
    als Fallback ebenfalls unterstützt.
    """
    a, b = [], []
    diffs: List[int] = []
    for entry in data:
        for system in ("baseline", "rag_semantic", "rag_raw"):
            raw = (_extract_raw_scores(entry, system, "correctness_likert")
                   or _extract_raw_scores(entry, system, "correctness"))
            if raw and len(raw) == 2:
                a.append(raw[0])
                b.append(raw[1])
                diffs.append(abs(raw[0] - raw[1]))
    if not a:
        return {"n": 0, "kappa_weighted": float("nan"), "mean_abs_diff": float("nan"),
                "exact_agreement_pct": float("nan")}
    kappa = cohens_kappa_quadratic_weighted(a, b, n_categories=5)
    return {
        "n": len(a),
        "kappa_weighted": kappa,
        "mean_abs_diff": float(np.mean(diffs)),
        "exact_agreement_pct": 100.0 * sum(1 for d in diffs if d == 0) / len(diffs),
    }


# ─── Hypothesentests ──────────────────────────────────────────────────────────

def _section(title: str) -> str:
    """Formatiere einen Zwischenüberschrift-Block für den Textreport."""
    return f"\n--- {title} ---"


def _paired_diff_summary(diffs: np.ndarray) -> List[str]:
    """Zähle pos/neg/ties und warne bei Decken-/Boden-Effekt.

    Wilcoxon ist underpowered, wenn der Großteil der Paare unentschieden ist.
    Eine Tie-Quote > 50 % wird ausdrücklich ausgewiesen, damit ein p≈α nicht
    fehlinterpretiert wird.
    """
    n = diffs.size
    eps = 1e-9
    n_pos = int((diffs > eps).sum())
    n_neg = int((diffs < -eps).sum())
    n_ties = int(np.sum(np.abs(diffs) <= eps))
    line = f"Verteilung der Differenzen: pos={n_pos}, neg={n_neg}, ties={n_ties} (n={n})"
    out = [line]
    if n > 0 and n_ties / n >= 0.5:
        out.append(
            f"⚠ Decken-/Boden-Effekt: {n_ties}/{n} Paare ({100 * n_ties / n:.0f} %) "
            f"sind unentschieden - der Test entscheidet effektiv aus {n - n_ties} "
            f"Datenpunkten und ist statistisch underpowered."
        )
    return out


def _h1_test(df: pd.DataFrame, *, label: str = "H1: alle Items") -> List[str]:
    """Gepaarter einseitiger Wilcoxon: RAG Semantic Correctness > Baseline."""
    out: List[str] = [_section(label)]
    out.append("H_a (einseitig): RAG Semantic Correctness > Baseline Correctness")
    valid = df[["base_corr", "sem_corr"]].dropna()
    if len(valid) < 5:
        out.append(f"Zu wenige Datenpunkte (n={len(valid)}), Test übersprungen.")
        return out
    base, sem = valid["base_corr"], valid["sem_corr"]
    n = len(valid)
    res: Any = stats.wilcoxon(sem, base, alternative="greater", zero_method="wilcox")
    raw_p = float(res.pvalue)
    corr_p = bonferroni(raw_p, BONFERRONI_N)
    z_stat = float(getattr(res, "zstatistic", float("nan")))
    r = rank_biserial_wilcoxon(raw_p, n, z_statistic=z_stat)
    diff_array = np.asarray(sem - base)
    diff_lo, diff_hi = bootstrap_ci_paired_diff(sem, base, n_iter=BOOT_N, seed=BOOT_SEED)
    out += [
        f"Test: Wilcoxon-Vorzeichen-Rang (gepaart, einseitig) (n={n})",
        f"Shapiro-Wilk (Doku): Baseline normal={shapiro_is_normal(base)}, "
        f"RAG Sem normal={shapiro_is_normal(sem)}",
    ]
    out += _paired_diff_summary(diff_array)
    out += [
        f"p (roh): {raw_p:.4f}    p (Bonferroni): {corr_p:.4f}",
        f"Effektstärke (r): {r:.3f}",
        f"Mittlere Differenz (sem − base): {diff_array.mean():.3f}  "
        f"[95% CI {diff_lo:.3f}, {diff_hi:.3f}]",
        f"Signifikant (α' = {ALPHA / BONFERRONI_N:.4f}): "
        f"{'JA' if corr_p < ALPHA / BONFERRONI_N else 'NEIN'}",
        f"Ø Baseline Correctness:     {base.mean():.3f}  (SD={base.std():.3f})",
        f"Ø RAG Semantic Correctness: {sem.mean():.3f}  (SD={sem.std():.3f})",
    ]
    return out


def _h2_test(df: pd.DataFrame) -> List[str]:
    """Unabhängiger Mann-Whitney-U: Faithfulness(Definition) > Faithfulness(Transfer/Anwendung)."""
    out: List[str] = [_section("H2: Definition vs. Transfer/Anwendung (Faithfulness, RAG Sem)")]
    out.append("H_a (einseitig): Faithfulness(Definition) > Faithfulness(Transfer/Anwendung)")
    valid = df.dropna(subset=["sem_faith"])
    def_scores = valid[valid["question_type"] == "Definition"]["sem_faith"]
    trans_scores = valid[valid["question_type"].isin(["Transfer", "Anwendung"])]["sem_faith"]
    if len(def_scores) < 3 or len(trans_scores) < 3:
        out.append(f"Zu wenige Datenpunkte (n_def={len(def_scores)}, "
                   f"n_trans={len(trans_scores)}). Test übersprungen.")
        return out
    res: Any = stats.mannwhitneyu(def_scores, trans_scores, alternative="greater")
    raw_p = float(res.pvalue)
    corr_p = bonferroni(raw_p, BONFERRONI_N)
    r = rank_biserial_mannwhitney(float(res.statistic), len(def_scores), len(trans_scores))
    out += [
        f"Test: Mann-Whitney-U (einseitig, unabhängig) "
        f"(n_def={len(def_scores)}, n_trans/anw={len(trans_scores)})",
        f"Shapiro-Wilk (Doku): Definition normal={shapiro_is_normal(def_scores)}, "
        f"Transfer/Anwendung normal={shapiro_is_normal(trans_scores)}",
        f"p (roh): {raw_p:.4f}    p (Bonferroni): {corr_p:.4f}",
        f"Effektstärke (r): {r:.3f}",
        f"Signifikant (α' = {ALPHA / BONFERRONI_N:.4f}): "
        f"{'JA' if corr_p < ALPHA / BONFERRONI_N else 'NEIN'}",
        f"Ø Definition Faithfulness:         {def_scores.mean():.3f}  (SD={def_scores.std():.3f})",
        f"Ø Transfer/Anwendung Faithfulness: {trans_scores.mean():.3f}  (SD={trans_scores.std():.3f})",
    ]
    return out


def _h3_test(df: pd.DataFrame, *, label: Optional[str] = None) -> List[str]:
    """Gepaarter einseitiger Wilcoxon: RAG Semantic Faithfulness > RAG Raw Faithfulness."""
    title = label or "H3: RAG Raw vs. RAG Semantic (Faithfulness, kanonischer Kontext)"
    out: List[str] = [_section(title)]
    out.append("H_a (einseitig): RAG Semantic Faithfulness > RAG Raw Faithfulness")
    valid = df[["raw_faith", "sem_faith"]].dropna()
    if len(valid) < 5:
        out.append(f"Zu wenige Datenpunkte (n={len(valid)}). Test übersprungen.")
        return out
    raw_f, sem_f = valid["raw_faith"], valid["sem_faith"]
    n = len(valid)
    try:
        res: Any = stats.wilcoxon(sem_f, raw_f, alternative="greater", zero_method="wilcox")
    except ValueError as e:
        out.append(f"Wilcoxon nicht möglich (vermutlich identische Werte): {e}")
        return out
    raw_p = float(res.pvalue)
    corr_p = bonferroni(raw_p, BONFERRONI_N)
    z_stat = float(getattr(res, "zstatistic", float("nan")))
    r = rank_biserial_wilcoxon(raw_p, n, z_statistic=z_stat)
    diff_array = np.asarray(sem_f - raw_f)
    diff_lo, diff_hi = bootstrap_ci_paired_diff(sem_f, raw_f, n_iter=BOOT_N, seed=BOOT_SEED)
    out += [
        f"Test: Wilcoxon-Vorzeichen-Rang (gepaart, einseitig) (n={n})",
        f"Shapiro-Wilk (Doku): RAG Raw normal={shapiro_is_normal(raw_f)}, "
        f"RAG Sem normal={shapiro_is_normal(sem_f)}",
    ]
    out += _paired_diff_summary(diff_array)
    out += [
        f"p (roh): {raw_p:.4f}    p (Bonferroni): {corr_p:.4f}",
        f"Effektstärke (r): {r:.3f}",
        f"Mittlere Differenz (sem − raw): {diff_array.mean():.3f}  "
        f"[95% CI {diff_lo:.3f}, {diff_hi:.3f}]",
        f"Signifikant (α' = {ALPHA / BONFERRONI_N:.4f}): "
        f"{'JA' if corr_p < ALPHA / BONFERRONI_N else 'NEIN'}",
        f"Ø RAG Raw Faithfulness:      {raw_f.mean():.3f}  (SD={raw_f.std():.3f})",
        f"Ø RAG Semantic Faithfulness: {sem_f.mean():.3f}  (SD={sem_f.std():.3f})",
    ]
    return out


def perform_statistical_tests(df: pd.DataFrame, irr: Dict[str, float]) -> List[str]:
    """Führe alle Hypothesentests + Sensitivitätsanalysen aus und liefere den Report-Block."""
    lines: List[str] = []
    corr_alpha = ALPHA / BONFERRONI_N
    lines.append("=" * 72)
    lines.append("STATISTISCHE AUSWERTUNG DER HYPOTHESEN")
    lines.append(f"α = {ALPHA}    Bonferroni-α' = {corr_alpha:.4f} ({BONFERRONI_N} Tests)")
    lines.append("Tests: Wilcoxon (gepaart, H1/H3), Mann-Whitney-U (unabhängig, H2)")
    lines.append("CIs: Perzentil-Bootstrap (n_iter=2000)")
    lines.append("=" * 72)

    lines.append("\n--- INTER-RATER-RELIABILITÄT (Validität des LLM-Judges) ---")
    lines.append(f"n (Bewertungs-Paare aller Systeme): {irr['n']}")
    lines.append(f"Quadratisch gewichtetes Cohen's κ:  {irr['kappa_weighted']:.3f}")
    lines.append(f"Mittlere absolute Score-Differenz:  {irr['mean_abs_diff']:.3f} (auf 1–5-Skala)")
    lines.append(f"Anteil exakter Übereinstimmung:     {irr['exact_agreement_pct']:.1f} %")
    lines.append("Interpretation κ: <0.4 schwach · 0.4–0.6 moderat · 0.6–0.8 substantiell · >0.8 exzellent")

    lines += _h1_test(df, label="H1 (alle Items): Baseline vs. RAG Semantic (Correctness)")

    sub_required = cast(pd.DataFrame, df[df["requires_context"]])
    if len(sub_required) >= 5 and len(sub_required) < len(df):
        lines += _h1_test(
            sub_required,
            label=f"H1 SENSITIVITÄT: nur requires_context=True (n={len(sub_required)})",
        )
    sub_general = cast(pd.DataFrame, df[~df["requires_context"]])
    if len(sub_general) >= 5:
        lines += _h1_test(
            sub_general,
            label=f"H1 SENSITIVITÄT: nur requires_context=False (n={len(sub_general)})",
        )

    lines += _h2_test(df)
    lines += _h3_test(df)

    # Sensitivitätsanalyse H3: nur kanonisch gemessene Items (Fallbacks sind
    # methodisch schwächer und werden separat ausgewiesen).
    if "canonical_used" in df.columns:
        df_canon = cast(pd.DataFrame, df[df["canonical_used"]])
        n_canon = len(df_canon)
        n_total = len(df)
        if 0 < n_canon < n_total and n_canon >= 5:
            lines += _h3_test(
                df_canon,
                label=f"H3 SENSITIVITÄT: nur kanonisch gemessene Items (n={n_canon})",
            )

    return lines


def build_summary(df: pd.DataFrame) -> List[str]:
    """Erzeuge den deskriptiven Kopf-Block des Reports (Mittelwerte, SDs, CIs)."""
    lines: List[str] = []
    lines.append("\n" + "=" * 72)
    lines.append("DESKRIPTIVE ZUSAMMENFASSUNG")
    lines.append("=" * 72)
    lines.append(f"Gesamtanzahl evaluierter Fragen: {len(df)}")
    lines.append(f"  davon requires_context=True:  {df['requires_context'].sum()}")
    lines.append(f"  davon requires_context=False: {(~df['requires_context']).sum()}")
    if "canonical_used" in df.columns and len(df) > 0:
        n_canon = int(df["canonical_used"].sum())
        pct = 100 * n_canon / len(df)
        lines.append(
            f"  Faithfulness-Referenz: kanonisch={n_canon}/{len(df)} ({pct:.0f} %), "
            f"Fallback=retrieved_context={len(df) - n_canon}"
        )
        if pct < 95:
            lines.append(
                "  ⚠ Bei <95 % kanonischer Coverage ist H3 nur eingeschränkt "
                "interpretierbar - fallback-Items messen System gegen eigenen Kontext."
            )
    lines.append("")

    def line_for(col: str, label: str) -> str:
        s = df[col].dropna()
        if s.empty:
            return f"  {label}: n=0"
        lo, hi = bootstrap_ci_mean(s.to_numpy(), n_iter=BOOT_N, seed=BOOT_SEED)
        return (f"  {label}: M={s.mean():.3f}  SD={s.std():.3f}  Median={s.median():.3f}  "
                f"95% CI [{lo:.3f}, {hi:.3f}]  n={len(s)}")

    lines.append("CORRECTNESS (LLM-as-a-Judge vs. Ground Truth):")
    lines.append(line_for("base_corr", "Baseline    "))
    lines.append(line_for("sem_corr",  "RAG Semantic"))
    lines.append(line_for("raw_corr",  "RAG Raw     "))
    lines.append("")
    lines.append("FAITHFULNESS (Ragas-style proportional, gegen kanonischen PDF-Text):")
    lines.append(line_for("sem_faith", "RAG Semantic"))
    lines.append(line_for("raw_faith", "RAG Raw     "))
    lines.append("")
    lines.append("FAITHFULNESS NACH FRAGETYP (RAG Semantic):")
    for qtype in df["question_type"].dropna().unique():
        s = cast(pd.DataFrame, df[df["question_type"] == qtype])["sem_faith"].dropna()
        if not s.empty:
            lo, hi = bootstrap_ci_mean(s.to_numpy(), n_iter=BOOT_N, seed=BOOT_SEED)
            lines.append(f"  {qtype}: n={len(s)}  M={s.mean():.3f}  SD={s.std():.3f}  "
                         f"95% CI [{lo:.3f}, {hi:.3f}]")
    return lines


# ─── Human↔LLM-Validierung (CSV) ──────────────────────────────────────────────

def analyze_human_validation() -> List[str]:
    """Lese die manuelle Review-CSV. Wenn mind. ein Human-Score eingetragen
    ist, berechne Spearman-Korrelation pro Spalte und plotte Scatter-Vergleich."""
    out: List[str] = []
    csv_path = config.FILE_MANUAL_REVIEW_CSV
    if not csv_path.exists():
        return out
    try:
        df = pd.read_csv(csv_path, sep=";")
    except Exception as e:
        out.append(f"\nManuelle Review-CSV konnte nicht gelesen werden: {e}")
        return out

    pairs = [
        ("Baseline Correctness (LLM)", "Human Baseline Correctness", "Baseline Correctness"),
        ("RAG Sem Correctness (LLM)",  "Human Sem Correctness",      "RAG Sem Correctness"),
        ("RAG Sem Faithfulness (LLM)", "Human Sem Faithfulness",     "RAG Sem Faithfulness"),
        ("RAG Raw Correctness (LLM)",  "Human Raw Correctness",      "RAG Raw Correctness"),
        ("RAG Raw Faithfulness (LLM)", "Human Raw Faithfulness",     "RAG Raw Faithfulness"),
    ]

    rows: List[str] = []
    plotted: List[str] = []
    for llm_col, human_col, label in pairs:
        if llm_col not in df.columns or human_col not in df.columns:
            continue
        # cast: pandas-Stubs typisieren .apply()/.dropna() als Series | DataFrame.
        sub = cast(
            pd.DataFrame,
            df[[llm_col, human_col]].apply(pd.to_numeric, errors="coerce").dropna(),
        )
        if len(sub) < 3:
            continue
        try:
            # spearmanr ist in den scipy-Stubs als Tuple typisiert -> über Any entpacken.
            sp_res: Any = stats.spearmanr(sub[llm_col], sub[human_col])
        except Exception:
            continue
        rho, p = float(sp_res[0]), float(sp_res[1])
        mae = float(np.mean(np.abs(sub[llm_col] - sub[human_col])))
        rows.append(f"  {label}: n={len(sub)}  Spearman ρ={rho:.3f} (p={p:.4f})  MAE={mae:.3f}")
        plot_human_vs_llm(
            human=cast(pd.DataFrame, sub.rename(columns={human_col: label})[[label]]),
            llm=cast(pd.DataFrame, sub.rename(columns={llm_col: label})[[label]]),
            metric=label, system_label=label,
            rho=rho, mae=mae, n=len(sub),
        )
        plotted.append(label)

    if not rows:
        return out
    out.append("\n" + "=" * 72)
    out.append("VALIDIERUNG DES LLM-JUDGES DURCH MANUELLE STICHPROBE")
    out.append("=" * 72)
    out += rows
    out.append(f"Streudiagramme exportiert: {len(plotted)} Datei(en) in {rel(config.DIR_ANALYSIS)}/")
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_analysis(input_path: Optional[str] = None) -> None:
    """Führe die komplette statistische Auswertung aus.

    Direkt aufrufbar (z.B. aus dem Orchestrator), ohne argparse-Side-Effects.
    Das Logging wird bewusst NICHT hier konfiguriert: So bleibt beim
    Orchestrator-Lauf der zentrale 0_orchestrator.log aktiv (sonst würden
    Schritt 6–8 in eine separate Datei umgelenkt). Beim Standalone-Aufruf
    richtet main() den Logger auf 6_analysis.log ein.
    """
    in_path = Path(input_path or config.FILE_EVALUATION)
    if not in_path.exists():
        raise FileNotFoundError(f"Evaluations-Datei nicht gefunden: {rel(in_path)}")

    data = load_json(in_path)
    df = to_dataframe(data)

    plot_correctness_boxplot(df)
    plot_faithfulness_by_type(df)
    plot_faithfulness_comparison(df)

    irr = compute_irr(data)
    summary_lines = build_summary(df)
    test_lines = perform_statistical_tests(df, irr)
    human_lines = analyze_human_validation()
    all_lines = summary_lines + test_lines + human_lines

    for line in all_lines:
        print(line)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))

    print(f"\nReport gespeichert: {REPORT_FILE}")

    # Ergänzende Post-hoc- und Robustheits-Analysen (eigener Report).
    from . import interaction_analysis
    interaction_analysis.run(str(in_path))


def main() -> None:
    """CLI-Einstieg für die alleinstehende statistische Auswertung."""
    parser = argparse.ArgumentParser(description="Statistische Auswertung & Plots.")
    parser.add_argument("--input", type=str, default=str(config.FILE_EVALUATION),
                        help="Pfad zur Evaluations-JSON.")
    args = parser.parse_args()
    config.ensure_output_dirs()
    setup_logger(config.DIR_ANALYSIS / "6_analysis.log")
    run_analysis(args.input)


if __name__ == "__main__":
    main()
