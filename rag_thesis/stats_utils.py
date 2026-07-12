"""Statistik-Hilfsfunktionen für die Auswertung (Schritte 6–8).

Sammelt alle nicht-trivialen statistischen Bausteine an einer Stelle, damit sie
zentral getestet werden können (siehe tests/test_stats_utils.py) und die
Auswertungs-Skripte selbst schlank bleiben:

- bonferroni: Korrektur für multiples Testen (gekappt bei 1,0).
- bootstrap_ci_mean / bootstrap_ci_paired_diff: Perzentil-Bootstrap-CIs.
- rank_biserial_wilcoxon / rank_biserial_mannwhitney: Effektstärken.
- cohens_kappa_quadratic_weighted: IRR des Likert-Judges (ordinal).
- cohens_kappa_unweighted: IRR der A/B/C-Kategorisierung (nominal).
- wilson_ci: Konfidenzintervall für Anteile (robust nahe 0 und 1).
- shapiro_is_normal: Normalverteilungs-Check zur Testwahl.

Bewusst ohne externe Statistik-Pakete (statsmodels o. Ä.). Nur numpy und
scipy.stats, damit die Verfahren nachvollziehbar und prüfbar bleiben.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

# 95-%-Standardnormal-Quantil (z), einmal berechnet für Wilson-CI.
_Z_95 = float(stats.norm.ppf(0.975))


# ─── Multiples Testen ─────────────────────────────────────────────────────────

def bonferroni(p_value: float, n_tests: int) -> float:
    """Bonferroni-korrigierter p-Wert, gekappt bei 1,0.

    Multipliziert den rohen p-Wert mit der Zahl der Tests. Der Cap verhindert,
    dass ein korrigierter p-Wert unsinnig über 1 liegt.
    """
    return float(min(p_value * n_tests, 1.0))


# ─── Bootstrap-Konfidenzintervalle ────────────────────────────────────────────

def _percentile_ci(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Zweiseitiges Perzentil-Intervall aus einer Bootstrap-Verteilung."""
    lo = float(np.percentile(samples, 100 * (alpha / 2)))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_ci_mean(
    data: ArrayLike,
    n_iter: int = 2000,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Perzentil-Bootstrap-CI für den Mittelwert (Ziehen mit Zurücklegen).

    seed=None lässt den Generator zufällig starten (Standardverhalten). 
    Für reproduzierbare Tests wird ein fester Seed übergeben.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    means = arr[idx].mean(axis=1)
    return _percentile_ci(means, alpha)


def bootstrap_ci_paired_diff(
    a: ArrayLike,
    b: ArrayLike,
    n_iter: int = 2000,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Perzentil-Bootstrap-CI für die mittlere gepaarte Differenz a − b.

    Paare mit fehlendem Wert auf einer Seite werden verworfen. Bei identischen
    Reihen ist die Differenz konstant 0 -> das CI kollabiert korrekt auf [0, 0].
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    diffs = arr_a[mask] - arr_b[mask]
    n = diffs.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(diffs[0]), float(diffs[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    means = diffs[idx].mean(axis=1)
    return _percentile_ci(means, alpha)


# ─── Effektstärken (rangbiserial) ─────────────────────────────────────────────

def rank_biserial_wilcoxon(
    p_value: float,
    n: int,
    z_statistic: Optional[float] = None,
) -> float:
    """Effektstärke r für den (gepaarten) Wilcoxon-Test: r = z / √n.

    Bevorzugt die von SciPy gelieferte z-Statistik. Liefert SciPy keine
    (ältere Versionen, exakte Methode -> NaN), wird z aus dem einseitigen
    p-Wert über die inverse Standardnormalverteilung approximiert. Das Vorzeichen 
    folgt der Testrichtung: p < 0,5 -> positiver Effekt, p > 0,5 -> negativer Effekt.
    """
    if n <= 0:
        return float("nan")
    if z_statistic is not None and not math.isnan(z_statistic):
        return float(z_statistic) / math.sqrt(n)
    # Fallback: z aus dem einseitigen p-Wert.
    p = min(max(p_value, 1e-12), 1 - 1e-12)
    z = float(stats.norm.ppf(1 - p))
    return z / math.sqrt(n)


def rank_biserial_mannwhitney(u_stat: float, n1: int, n2: int) -> float:
    """Rangbiseriale Korrelation r für den Mann-Whitney-U-Test.

    r = 2·U / (n1·n2) − 1, liegt in [−1, 1]. U = n1·n2 (perfekte 
    Trennung in Testrichtung) ergibt r = 1, U = 0 ergibt r = −1.
    """
    denom = n1 * n2
    if denom == 0:
        return float("nan")
    return 2.0 * u_stat / denom - 1.0


# ─── Cohen's κ (Inter-Rater-Reliabilität) ─────────────────────────────────────

def cohens_kappa_quadratic_weighted(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    n_categories: int,
) -> float:
    """Quadratisch gewichtetes Cohen's κ für ordinale Skalen (z.B. Likert 1–5).

    Gewichtet Abweichungen quadratisch mit ihrem Abstand: eine 1↔2-Abweichung
    zählt weniger als eine 1↔5-Abweichung. Geeignet für die Position-Swap-
    Doppelbewertung des Judges. Werte: 1 = perfekt, 0 = Zufallsniveau,
    < 0 = systematische Gegen-Übereinstimmung.
    """
    a = [int(round(x)) - 1 for x in rater_a]  # 1..K -> 0..K-1
    b = [int(round(x)) - 1 for x in rater_b]
    k = n_categories
    if not a or len(a) != len(b) or k < 2:
        return float("nan")

    observed = np.zeros((k, k), dtype=float)
    for i, j in zip(a, b):
        if 0 <= i < k and 0 <= j < k:
            observed[i, j] += 1
    total = observed.sum()
    if total == 0:
        return float("nan")

    # Quadratische Distanz-Gewichte (0 auf der Diagonalen, max an den Ecken).
    idx = np.arange(k)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2

    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col) / total

    denom = float((weights * expected).sum())
    if denom == 0:  # entartet (alle in einer Kategorie)
        return 1.0 if float((weights * observed).sum()) == 0 else float("nan")
    return 1.0 - float((weights * observed).sum()) / denom


def cohens_kappa_unweighted(
    rater_a: Sequence,
    rater_b: Sequence,
    categories: Optional[Sequence] = None,
) -> float:
    """Ungewichtetes Cohen's κ für nominale Kategorien (z.B. A/B/C).

    Misst die um Zufall korrigierte Übereinstimmung zweier Rater. Bei
    ungleicher Länge der Eingaben nicht definiert -> NaN.
    """
    if len(rater_a) != len(rater_b):
        return float("nan")
    n = len(rater_a)
    if n == 0:
        return float("nan")
    if categories is None:
        categories = sorted(set(rater_a) | set(rater_b))

    # Beobachtete Übereinstimmung p_o.
    p_o = sum(1 for x, y in zip(rater_a, rater_b) if x == y) / n

    # Erwartete Zufalls-Übereinstimmung p_e aus den Randverteilungen.
    p_e = 0.0
    for c in categories:
        pa = sum(1 for x in rater_a if x == c) / n
        pb = sum(1 for y in rater_b if y == c) / n
        p_e += pa * pb

    if math.isclose(p_e, 1.0):  # nur eine Kategorie belegt
        return 1.0 if math.isclose(p_o, 1.0) else float("nan")
    return (p_o - p_e) / (1.0 - p_e)


# ─── Anteils-Konfidenzintervall ───────────────────────────────────────────────

def wilson_ci(successes: int, total: int) -> Tuple[float, float]:
    """Wilson-Score-Intervall (95 %) für einen Anteil.

    Robuster als das Wald-Intervall, besonders bei kleinen Anteilen nahe 0
    oder 1. Bleibt stets innerhalb [0, 1]. Bei total == 0 undefiniert
    -> (NaN, NaN).
    """
    if total <= 0:
        return float("nan"), float("nan")
    z = _Z_95
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ─── Verteilungs-Check ────────────────────────────────────────────────────────

def shapiro_is_normal(values: ArrayLike, alpha: float = 0.05) -> bool:
    """True, wenn der Shapiro-Wilk-Test Normalverteilung NICHT verwirft (p > α).

    Dient nur der Dokumentation der Testwahl (die Daten sind durchgehend nicht
    normalverteilt -> verteilungsfreie Verfahren). Bei < 3 Werten ist der Test
    nicht definiert -> False.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3:
        return False
    try:
        _, p = stats.shapiro(arr)
    except Exception:
        return False
    return bool(p > alpha)


# ─── Alternative Korrekturen für multiples Testen (Robustheit) ────────────────

def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni-korrigierte p-Werte (stufenweise, gekappt bei 1,0).

    Weniger konservativ als klassisches Bonferroni, kontrolliert aber weiterhin
    die familienweise Fehlerrate. Rückgabe in Eingabereihenfolge.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg-korrigierte p-Werte (False Discovery Rate).

    Kontrolliert die FDR statt der familienweisen Fehlerrate und ist damit
    weniger konservativ als Bonferroni/Holm. Rückgabe in Eingabereihenfolge.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):  # von hinten, um Monotonie zu sichern
        idx = order[rank]
        running = min(running, m / (rank + 1) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


# ─── Konfidenzintervall einer Korrelation (Fisher-z) ─────────────────────────

def fisher_z_ci(rho: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Konfidenzintervall für eine Spearman-/Pearson-Korrelation via Fisher-z.

    Transformiert rho über artanh, bildet das symmetrische Normal-Intervall und
    transformiert zurück. Für n ≤ 3 nicht definiert -> (NaN, NaN).
    """
    if n <= 3:
        return float("nan"), float("nan")
    if rho >= 1.0:
        return 1.0, 1.0
    if rho <= -1.0:
        return -1.0, -1.0
    z = math.atanh(rho)
    se = 1.0 / math.sqrt(n - 3)
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return math.tanh(z - crit * se), math.tanh(z + crit * se)


# ─── Power-Analyse (gepaart, einseitig) ───────────────────────────────────────

def achieved_power_paired_t(d_z: float, n: int, alpha: float) -> float:
    """Erreichte Power eines einseitigen gepaarten t-Tests (Approximation).

    Nutzt die nichtzentrale t-Verteilung mit Nichtzentralität d_z·√n und
    df = n − 1. Liefert die Wahrscheinlichkeit, bei wahrer Effektstärke d_z einen
    signifikanten Befund (Niveau alpha, einseitig) zu erhalten.
    """
    if n < 2:
        return float("nan")
    df = n - 1
    ncp = d_z * math.sqrt(n)
    t_crit = float(stats.t.ppf(1 - alpha, df))
    return float(stats.nct.sf(t_crit, df, ncp))


def mdes_paired_t(n: int, alpha: float, power: float = 0.8) -> float:
    """Minimal detektierbare Effektstärke d_z für gegebene Power (gepaart, einseitig).

    Sucht das d_z, bei dem achieved_power_paired_t die Zielpower erreicht,
    per Bisektion. Bei zu kleinem n nicht definiert -> NaN.
    """
    if n < 2:
        return float("nan")
    lo, hi = 0.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if achieved_power_paired_t(mid, n, alpha) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─── Scheirer-Ray-Hare (nichtparametrische 2-Faktor-ANOVA auf Rängen) ────────

def scheirer_ray_hare(
    values: Sequence[float],
    factor_a: Sequence,
    factor_b: Sequence,
) -> Dict[str, Tuple[float, int, float]]:
    """Scheirer-Ray-Hare-Test: nichtparametrische 2-Faktor-ANOVA auf Rängen.

    Rangiert alle Beobachtungen gemeinsam und zerlegt die Rang-Quadratsumme in
    die Haupteffekte (Faktor A, Faktor B) und die Interaktion. Jede Statistik H
    ist χ²-verteilt. Liefert pro Effekt (H, df, p) unter den Schlüsseln
    "a", "b" und "ab". Ties gehen über die geteilten Ränge korrekt ein.
    """
    vals = np.asarray(values, dtype=float)
    n = vals.size
    ranks = stats.rankdata(vals)
    grand = float(ranks.sum())
    ms_total = float(((ranks - ranks.mean()) ** 2).sum() / (n - 1))

    def _ss(levels: Sequence) -> Tuple[float, int]:
        groups: dict = {}
        for i, lev in enumerate(levels):
            groups.setdefault(lev, []).append(i)
        ss = sum(ranks[idx].sum() ** 2 / len(idx) for idx in groups.values())
        return ss - grand ** 2 / n, len(groups)

    ss_a, k_a = _ss(factor_a)
    ss_b, k_b = _ss(factor_b)
    cells: dict = {}
    for i in range(n):
        cells.setdefault((factor_a[i], factor_b[i]), []).append(i)
    ss_cells = sum(ranks[idx].sum() ** 2 / len(idx) for idx in cells.values())
    ss_ab = (ss_cells - grand ** 2 / n) - ss_a - ss_b

    out: Dict[str, Tuple[float, int, float]] = {}
    for key, ss, df in (("a", ss_a, k_a - 1), ("b", ss_b, k_b - 1),
                        ("ab", ss_ab, (k_a - 1) * (k_b - 1))):
        h = ss / ms_total if ms_total > 0 and df > 0 else float("nan")
        p = float(stats.chi2.sf(h, df)) if df > 0 and not math.isnan(h) else float("nan")
        out[key] = (float(h), int(df), p)
    return out
