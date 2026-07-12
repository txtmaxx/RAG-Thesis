"""Tests für Statistik-Hilfsfunktionen."""

import math

import numpy as np

from rag_thesis.stats_utils import (
    achieved_power_paired_t,
    benjamini_hochberg,
    bonferroni,
    bootstrap_ci_mean,
    bootstrap_ci_paired_diff,
    cohens_kappa_quadratic_weighted,
    cohens_kappa_unweighted,
    fisher_z_ci,
    holm_bonferroni,
    mdes_paired_t,
    rank_biserial_mannwhitney,
    rank_biserial_wilcoxon,
    scheirer_ray_hare,
    wilson_ci,
)


def test_bonferroni_capped_at_one():
    assert bonferroni(0.5, 3) == 1.0
    assert bonferroni(0.01, 3) == 0.03
    assert bonferroni(0.0, 3) == 0.0


def test_bootstrap_ci_mean_recovers_mean():
    rng = np.random.default_rng(123)
    data = rng.normal(loc=0.5, scale=0.1, size=200)
    lo, hi = bootstrap_ci_mean(data, n_iter=1000, seed=7)
    assert lo < 0.5 < hi
    assert hi - lo < 0.05


def test_bootstrap_ci_paired_diff_zero():
    a = list(range(50))
    b = list(range(50))
    lo, hi = bootstrap_ci_paired_diff(a, b, n_iter=500, seed=1)
    assert math.isclose(lo, 0.0) and math.isclose(hi, 0.0)


def test_rank_biserial_wilcoxon_sign():
    # p < 0.5 -> positiver Effekt (in getestete Richtung)
    r_pos = rank_biserial_wilcoxon(p_value=0.01, n=30)
    assert r_pos > 0
    # p > 0.5 -> negativer Effekt (gegen die getestete Richtung)
    r_neg = rank_biserial_wilcoxon(p_value=0.9, n=30)
    assert r_neg < 0


def test_rank_biserial_wilcoxon_prefers_zstatistic():
    # Wenn z-Statistik übergeben wird, soll sie die Ableitung steuern,
    # nicht der p-Wert.
    r = rank_biserial_wilcoxon(p_value=0.5, n=25, z_statistic=2.5)
    assert math.isclose(r, 2.5 / math.sqrt(25))
    # Negative z -> negativer Effekt, unabhängig vom p-Wert.
    r_neg = rank_biserial_wilcoxon(p_value=0.01, n=25, z_statistic=-2.5)
    assert r_neg < 0


def test_rank_biserial_wilcoxon_nan_zstatistic_falls_back():
    # NaN-z (SciPy < 1.11) -> Fallback auf die p-Wert-Approximation.
    r = rank_biserial_wilcoxon(p_value=0.01, n=30, z_statistic=float("nan"))
    assert r > 0


def test_rank_biserial_mannwhitney_bounds():
    # u_stat = n1*n2 (perfekte Trennung) -> r = 1
    r = rank_biserial_mannwhitney(u_stat=100, n1=10, n2=10)
    assert math.isclose(r, 1.0)
    # u_stat = 0 -> r = -1
    r = rank_biserial_mannwhitney(u_stat=0, n1=10, n2=10)
    assert math.isclose(r, -1.0)


def test_cohens_kappa_perfect_agreement_is_one():
    a = [1, 2, 3, 4, 5, 1, 2, 3]
    b = [1, 2, 3, 4, 5, 1, 2, 3]
    assert math.isclose(cohens_kappa_quadratic_weighted(a, b, n_categories=5), 1.0)


def test_cohens_kappa_handles_disagreement():
    a = [1, 2, 3, 4, 5]
    b = [5, 4, 3, 2, 1]
    k = cohens_kappa_quadratic_weighted(a, b, n_categories=5)
    assert k < 0.0  # systematische Gegen-Übereinstimmung -> negatives κ


def test_wilson_ci_contains_point_estimate_and_stays_in_bounds():
    lo, hi = wilson_ci(55, 528)  # Definitions-C-Anteil ≈ 10,4 %
    assert 0.0 <= lo < 0.104 < hi <= 1.0
    # Bekannter Referenzwert (Wilson, 95 %): rund [8,1 %, 13,3 %]
    assert math.isclose(lo, 0.081, abs_tol=0.003)
    assert math.isclose(hi, 0.133, abs_tol=0.003)


def test_wilson_ci_extremes_do_not_leave_unit_interval():
    lo, hi = wilson_ci(0, 30)
    assert lo >= 0.0
    lo2, hi2 = wilson_ci(30, 30)
    assert hi2 <= 1.0


def test_wilson_ci_empty_is_nan():
    lo, hi = wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_cohens_kappa_unweighted_perfect():
    a = ["A", "B", "C", "A", "C"]
    assert math.isclose(cohens_kappa_unweighted(a, a, categories=("A", "B", "C")), 1.0)


def test_cohens_kappa_unweighted_partial_agreement():
    machine = ["A", "B", "C", "A", "C", "A"]
    human = ["A", "A", "C", "A", "C", "B"]  # 4/6 gleich
    k = cohens_kappa_unweighted(machine, human, categories=("A", "B", "C"))
    assert 0.0 < k < 1.0


def test_cohens_kappa_unweighted_mismatched_length_is_nan():
    assert math.isnan(cohens_kappa_unweighted(["A", "B"], ["A"]))


def test_holm_bonferroni_known_values_and_cap():
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert math.isclose(adj[0], 0.03, abs_tol=1e-9)
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert holm_bonferroni([0.9, 0.8, 0.7])[0] <= 1.0


def test_benjamini_hochberg_le_bonferroni():
    raw = [0.01, 0.04, 0.03]
    bh = benjamini_hochberg(raw)
    assert all(0.0 <= a <= 1.0 for a in bh)
    assert all(bh[i] <= bonferroni(raw[i], 3) + 1e-9 for i in range(3))


def test_fisher_z_ci_contains_rho_and_extremes():
    lo, hi = fisher_z_ci(0.9, 18)
    assert lo < 0.9 < hi
    assert lo >= -1.0 and hi <= 1.0
    assert fisher_z_ci(1.0, 18) == (1.0, 1.0)
    assert all(math.isnan(v) for v in fisher_z_ci(0.5, 3))


def test_power_increases_and_mdes_recovers_target():
    alpha = 0.0167
    p_small = achieved_power_paired_t(0.1, 40, alpha)
    p_large = achieved_power_paired_t(0.8, 40, alpha)
    assert 0.0 <= p_small < p_large <= 1.0
    mdes = mdes_paired_t(40, alpha, 0.8)
    assert math.isclose(achieved_power_paired_t(mdes, 40, alpha), 0.8, abs_tol=0.02)


def test_scheirer_ray_hare_detects_dominant_factor():
    vals = [1, 2, 1, 2, 10, 11, 10, 11]
    factor_a = ["lo", "lo", "lo", "lo", "hi", "hi", "hi", "hi"]
    factor_b = ["x", "y", "x", "y", "x", "y", "x", "y"]
    res = scheirer_ray_hare(vals, factor_a, factor_b)
    h_a, df_a, p_a = res["a"]
    assert df_a == 1
    assert h_a > res["b"][0]  # Faktor A trennt stärker als B
    assert p_a < 0.1
