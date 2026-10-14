"""Unit tests for varlab.calibration.

Section 1 replicates EVERY hand-computed scoring case in tests/test_utils.R
(same inputs, same expected values, same or tighter tolerances).
Section 2 pins the R primitives (seq / cut / quantile) that the binning rests
on. Section 3 holds R-verified fixtures: deterministic datasets built from
integer arithmetic (identical in R and Python, no RNG) whose expected values
were printed by R/utils.R and R/04_calibration.R with 17 significant digits.
Sections 4-7 test reliability_data, isotonic_by_fold, group_folds and
abstention_curve.
"""

import math

import numpy as np
import pandas as pd
import pytest

from varlab import (
    abstention_curve,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    group_folds,
    isotonic_by_fold,
    log_loss,
    reliability_data,
)
from varlab.calibration import (
    _cut_right_closed,
    _equal_width_breaks,
    _quantile_type7,
)


# --------------------------------------------------------------------------- #
# deterministic datasets shared with the R reference run (no RNG involved)
# --------------------------------------------------------------------------- #
def dataset_a():
    """240 rows, 97 distinct p values with many ties and values sitting
    exactly on equal-width bin edges (0.1, 0.2, ...)."""
    i = np.arange(240)
    k = (37 * i) % 97 + 1
    p = k / 100
    y = ((31 * i) % 100 < k).astype(float)
    return p, y


def dataset_b():
    """240 rows, 240 distinct p values (no ties), three folds."""
    i = np.arange(240)
    k = (37 * i) % 240
    p = (k + 0.5) / 240
    y = ((31 * i) % 240 < k).astype(float)
    fold = i % 3 + 1
    return p, y, fold


# =========================================================================== #
# 1. hand-computed cases from tests/test_utils.R
# =========================================================================== #
class TestHandComputedFromR:
    def test_log_loss_toy(self):
        # p=.5,.5 ; y=1,0  -> -log(0.5)
        assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(-math.log(0.5), abs=1e-9)
        # confident + correct beats unsure
        assert log_loss([0.99], [1]) < log_loss([0.5], [1])

    def test_brier_score_toy(self):
        # p=.8,.2 ; y=1,0 -> mean(.04,.04)=.04
        assert brier_score([0.8, 0.2], [1, 0]) == pytest.approx(0.04, abs=1e-12)

    def test_ece_hand_computed(self):
        # two bins, each off by 0.1, equal weight -> ECE = 0.10
        assert expected_calibration_error(
            [0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1]
        ) == pytest.approx(0.10, abs=1e-9)
        # perfectly calibrated -> 0
        assert expected_calibration_error(
            [0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]
        ) == pytest.approx(0.0, abs=1e-9)

    def test_brier_decomposition_recomposes(self):
        # p=.2,.2,.8,.8 ; y=0,0,1,1 (perfect within-bin):
        #   uncertainty = .5*.5 = .25
        #   reliability = (2*.04 + 2*.04)/4 = .04
        #   resolution  = (2*.25 + 2*.25)/4 = .25
        #   recomposed  = .04 - .25 + .25 = .04 = brier
        d = brier_decomposition([0.2, 0.2, 0.8, 0.8], [0, 0, 1, 1])
        assert d["uncertainty"] == pytest.approx(0.25, abs=1e-12)
        assert d["reliability"] == pytest.approx(0.04, abs=1e-12)
        assert d["resolution"] == pytest.approx(0.25, abs=1e-12)
        assert d["brier"] == pytest.approx(0.04, abs=1e-12)
        assert d["recomposed"] == pytest.approx(d["brier"], abs=1e-12)
        assert set(d) == {"reliability", "resolution", "uncertainty", "brier", "recomposed"}


# =========================================================================== #
# 2. R primitives: seq(), cut(), quantile(type = 7)
# =========================================================================== #
class TestRPrimitives:
    def test_equal_width_breaks_match_r_seq(self):
        # R: sprintf("%.17g", seq(0, 1, length.out = 11))
        r_seq = [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5,
                 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0]
        b = _equal_width_breaks(10)
        assert b.tolist() == r_seq  # bit-exact
        assert _equal_width_breaks(1).tolist() == [0.0, 1.0]

    def test_cut_is_right_closed_and_includes_lowest(self):
        # R: cut(c(0, 0.1, 0.1000000000000001, 0.3, 0.30000000000000004, 1),
        #        seq(0, 1, length.out = 11), include.lowest = TRUE, labels = FALSE)
        #    -> 1 1 2 3 3 10
        x = np.array([0.0, 0.1, 0.1000000000000001, 0.3, 0.30000000000000004, 1.0])
        assert _cut_right_closed(x, _equal_width_breaks(10)).tolist() == [1, 1, 2, 3, 3, 10]

    def test_cut_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            _cut_right_closed(np.array([-0.1]), _equal_width_breaks(10))
        with pytest.raises(ValueError):
            _cut_right_closed(np.array([1.1]), _equal_width_breaks(10))

    def test_cut_single_break_puts_everything_in_one_bin(self):
        assert _cut_right_closed(np.array([0.3, 0.3]), np.array([0.3])).tolist() == [1, 1]

    def test_quantile_type7_matches_r(self):
        # R: quantile(c(.1,.2,.3,.4), c(0,.5,1)) -> 0.1 0.25 0.4
        q = _quantile_type7(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 0.5, 1.0]))
        assert q.tolist() == [0.1, 0.25, 0.4]
        # single value, all probs -> that value
        q1 = _quantile_type7(np.array([0.7]), np.array([0.0, 0.5, 1.0]))
        assert q1.tolist() == [0.7, 0.7, 0.7]
        # agrees with numpy's default (linear == type 7) away from the fuzz
        x = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95, 0.97])
        probs = _equal_width_breaks(4)
        np.testing.assert_allclose(_quantile_type7(x, probs), np.quantile(x, probs), rtol=0, atol=1e-15)

    def test_ece_bin_edges_follow_r(self):
        # R-verified: 0.1 -> bin (0, 0.1], 0.15 -> bin (0.1, 0.2]  => 0.475
        # (a left-closed convention would give 0.375)
        assert expected_calibration_error([0.1, 0.15], [0, 1]) == pytest.approx(0.475, abs=1e-12)
        # R-verified with 2 bins: {0, 0.1, 0.2} -> bin 1, {1} -> bin 2 => 0.675
        assert expected_calibration_error([0, 0.1, 0.2, 1], [1, 1, 0, 0], 2) == pytest.approx(0.675, abs=1e-12)
        # R-verified: 0.5 is in bin 1, 0.5000001 in bin 2 => 0.49999995
        assert expected_calibration_error([0.5, 0.5000001], [0, 1], 2) == pytest.approx(0.49999995, abs=1e-12)


# =========================================================================== #
# 3. R-verified fixtures on the deterministic datasets
# =========================================================================== #
class TestRVerifiedFixtures:
    def test_dataset_a_is_what_r_saw(self):
        p, y = dataset_a()
        assert p.size == 240 and y.sum() == 116
        assert p.sum() == pytest.approx(116.66, abs=1e-12)
        assert len(np.unique(p)) == 97

    def test_scores_dataset_a(self):
        p, y = dataset_a()
        assert log_loss(p, y) == pytest.approx(0.54042698363780406, abs=1e-9)
        assert brier_score(p, y) == pytest.approx(0.17406083333333333, abs=1e-9)
        assert expected_calibration_error(p, y, 10) == pytest.approx(0.12466666666666666, abs=1e-9)
        assert expected_calibration_error(p, y, 7) == pytest.approx(0.12041666666666664, abs=1e-9)

    def test_brier_decomposition_dataset_a(self):
        p, y = dataset_a()
        d = brier_decomposition(p, y, 10)
        assert d["reliability"] == pytest.approx(0.020780855528846152, abs=1e-9)
        assert d["resolution"] == pytest.approx(0.09484615384615383, abs=1e-9)
        assert d["uncertainty"] == pytest.approx(0.24972222222222221, abs=1e-9)
        assert d["brier"] == pytest.approx(0.17406083333333333, abs=1e-9)
        assert d["recomposed"] == pytest.approx(0.17565692390491452, abs=1e-9)
        d7 = brier_decomposition(p, y, 7)
        assert d7["reliability"] == pytest.approx(0.019366577583270025, abs=1e-9)
        assert d7["resolution"] == pytest.approx(0.096828361975420801, abs=1e-9)
        assert d7["recomposed"] == pytest.approx(0.17226043783007144, abs=1e-9)

    def test_scores_dataset_b(self):
        p, y, _ = dataset_b()
        assert y.sum() == 117 and len(np.unique(p)) == 240
        assert log_loss(p, y) == pytest.approx(0.48508630852192203, abs=1e-9)
        assert expected_calibration_error(p, y, 10) == pytest.approx(0.035833333333333342, abs=1e-9)
        d = brier_decomposition(p, y, 10)
        assert d["reliability"] == pytest.approx(0.0017708333333333339, abs=1e-9)
        assert d["resolution"] == pytest.approx(0.087864583333333329, abs=1e-9)
        assert d["uncertainty"] == pytest.approx(0.24984374999999998, abs=1e-9)
        assert d["brier"] == pytest.approx(0.16234230324074073, abs=1e-9)
        assert d["recomposed"] == pytest.approx(0.16374999999999998, abs=1e-9)


# =========================================================================== #
# 4. reliability_data (equal-count quantile bins)
# =========================================================================== #
# (bin, mean_pred, obs_rate, n, se) printed by R/04_calibration.R::reliability_data
R_RELIABILITY_A_10 = [
    (1, 0.0516, 0.23999999999999999, 25, 0.085416626016250496),
    (2, 0.14826086956521739, 0.34782608695652173, 23, 0.099311346212201862),
    (3, 0.24440000000000001, 0.16, 25, 0.073321211119293433),
    (4, 0.34639999999999999, 0.040000000000000001, 25, 0.039191835884530846),
    (5, 0.43954545454545452, 0.40909090909090912, 22, 0.10482356110715554),
    (6, 0.53559999999999997, 0.47999999999999998, 25, 0.099919967974374371),
    (7, 0.63666666666666671, 0.5, 24, 0.10206207261596575),
    (8, 0.7326086956521739, 0.69565217391304346, 23, 0.095943875542151391),
    (9, 0.82583333333333331, 1, 24, 0.00020412414523193149),
    (10, 0.92416666666666669, 1, 24, 0.00020412414523193149),
]
R_RELIABILITY_A_5 = [
    (1, 0.097916666666666666, 0.29166666666666669, 48, 0.065605707040594388),
    (2, 0.2954, 0.10000000000000001, 50, 0.042426406871192854),
    (3, 0.49063829787234042, 0.44680851063829785, 47, 0.072518620566894329),
    (4, 0.68361702127659574, 0.5957446808510638, 47, 0.071582857226408184),
    (5, 0.875, 1, 48, 0.00014433756729740642),
]
R_RELIABILITY_B_10 = [
    (1, 0.050000000000000003, 0, 24, 0.00020412414523193149),
    (2, 0.14999999999999999, 0.16666666666666666, 24, 0.076072577431273075),
    (3, 0.25, 0.25, 24, 0.088388347648318447),
    (4, 0.34999999999999998, 0.29166666666666669, 24, 0.092780480665884638),
    (5, 0.45000000000000001, 0.5, 24, 0.10206207261596575),
    (6, 0.55000000000000004, 0.5, 24, 0.10206207261596575),
    (7, 0.65000000000000002, 0.625, 24, 0.09882117688026186),
    (8, 0.75, 0.75, 24, 0.088388347648318447),
    (9, 0.84999999999999998, 0.79166666666666663, 24, 0.082898169349398074),
    (10, 0.94999999999999996, 1, 24, 0.00020412414523193149),
]


def _assert_reliability_matches(df: pd.DataFrame, expected):
    assert list(df.columns) == ["bin", "mean_pred", "obs_rate", "n", "se"]
    assert len(df) == len(expected)
    for row, (b, mp, obs, n, se) in zip(df.itertuples(index=False), expected):
        assert row.bin == b
        assert row.n == n
        assert row.mean_pred == pytest.approx(mp, abs=1e-9)
        assert row.obs_rate == pytest.approx(obs, abs=1e-9)
        assert row.se == pytest.approx(se, abs=1e-9)


class TestReliabilityData:
    def test_hand_computed_two_bins(self):
        # quantile(c(.1,.2,.3,.4), c(0,.5,1)) = .1 .25 .4 -> bins [.1,.25], (.25,.4]
        df = reliability_data([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1], n_bins=2)
        assert df["bin"].tolist() == [1, 2]
        assert df["n"].tolist() == [2, 2]
        assert df["mean_pred"].tolist() == pytest.approx([0.15, 0.35], abs=1e-12)
        assert df["obs_rate"].tolist() == pytest.approx([0.0, 1.0], abs=1e-12)
        # se floor: sqrt(max(0, 1e-6) / 2)
        assert df["se"].tolist() == pytest.approx([math.sqrt(5e-7)] * 2, abs=1e-15)

    def test_r_verified_dataset_a(self):
        p, y = dataset_a()
        _assert_reliability_matches(reliability_data(p, y, 10), R_RELIABILITY_A_10)
        _assert_reliability_matches(reliability_data(p, y, 5), R_RELIABILITY_A_5)

    def test_r_verified_dataset_b(self):
        p, y, _ = dataset_b()
        _assert_reliability_matches(reliability_data(p, y, 10), R_RELIABILITY_B_10)

    def test_counts_sum_and_monotone_bins(self):
        p, y = dataset_a()
        df = reliability_data(p, y, 10)
        assert df["n"].sum() == p.size
        assert df["mean_pred"].is_monotonic_increasing
        assert np.all(df["se"] > 0)

    def test_duplicate_quantiles_collapse(self):
        # 90 copies of 0.2 and 10 of 0.9: most deciles coincide -> 2 bins
        p = np.r_[np.full(90, 0.2), np.full(10, 0.9)]
        y = np.r_[np.zeros(90), np.ones(10)]
        df = reliability_data(p, y, 10)
        assert df["bin"].tolist() == [1, 2]
        assert df["n"].tolist() == [90, 10]

    def test_constant_predictions_one_bin(self):
        df = reliability_data([0.3] * 5, [0, 1, 0, 0, 1], 10)
        assert len(df) == 1 and df.loc[0, "n"] == 5
        assert df.loc[0, "obs_rate"] == pytest.approx(0.4)


# =========================================================================== #
# 5. isotonic_by_fold
# =========================================================================== #
class TestIsotonicByFold:
    def test_hand_computed(self):
        # fold 2 (calibration for fold 1): p=.1,.2,.3,.4 y=0,0,1,1 -> step at .25
        # fold 1 test points .05 (below range -> 0), .25 (midway -> .5), .5 (above -> 1)
        # R-verified: isotonic_recalibrate() gives 0, 0.5, 1 for fold 1.
        p = [0.05, 0.25, 0.5, 0.1, 0.2, 0.3, 0.4]
        y = [0, 0, 1, 0, 0, 1, 1]
        fold = [1, 1, 1, 2, 2, 2, 2]
        out = isotonic_by_fold(p, y, fold)
        assert out[:3].tolist() == pytest.approx([0.0, 0.5, 1.0], abs=1e-12)
        assert np.all(np.isfinite(out))

    def test_tied_predictions_pooled_like_r(self):
        # Calibration fold has a tie at 0.2 with outcomes 0 and 1. R's isoreg
        # orders ties by order(x, -y), so the tie group is pooled to 0.5 and
        # approxfun interpolates 0.1->0, 0.2->0.5, 0.3->1. R-verified output
        # of isotonic_recalibrate() for the fold-1 rows (same for either row
        # order of the tied outcomes): 0, 0.24999999999999994, 0.5, 0.75, 1
        p = [0.05, 0.15, 0.2, 0.25, 0.9, 0.1, 0.2, 0.2, 0.3]
        y = [0, 0, 0, 1, 1, 0, 0, 1, 1]
        fold = [1, 1, 1, 1, 1, 2, 2, 2, 2]
        out = isotonic_by_fold(p, y, fold)[:5]
        assert out.tolist() == pytest.approx([0, 0.25, 0.5, 0.75, 1], abs=1e-12)
        y_swapped = [0, 0, 0, 1, 1, 0, 1, 0, 1]
        assert isotonic_by_fold(p, y_swapped, fold)[:5].tolist() == pytest.approx(out.tolist(), abs=1e-12)

    def test_r_verified_dataset_b(self):
        p, y, fold = dataset_b()
        iso = isotonic_by_fold(p, y, fold)
        first12 = [0, 0, 0.11428571428571428, 0.5, 0.5, 0.77884615384615385, 1, 0,
                   0.11428571428571428, 0.22857142857142856, 0.40384615384615385,
                   0.55769230769230771]
        last6 = [0, 0.20192307692307759, 0.5, 0.5, 0.8571428571428571, 1]
        assert iso[:12].tolist() == pytest.approx(first12, abs=1e-9)
        assert iso[-6:].tolist() == pytest.approx(last6, abs=1e-9)
        assert iso.sum() == pytest.approx(117.0, abs=1e-9)
        assert expected_calibration_error(iso, y, 10) == pytest.approx(0.13189102564102564, abs=1e-9)
        assert log_loss(iso, y) == pytest.approx(1.8132726981057226, abs=1e-9)
        assert brier_score(iso, y) == pytest.approx(0.22053979404015647, abs=1e-9)

    def test_output_shape_and_range(self):
        p, y, fold = dataset_b()
        iso = isotonic_by_fold(p, y, fold)
        assert iso.shape == p.shape
        assert np.all(np.isfinite(iso))
        assert iso.min() >= 0.0 and iso.max() <= 1.0

    def test_leakage_safe(self):
        # The map applied to fold f must not depend on fold f's outcomes.
        p, y, fold = dataset_b()
        base = isotonic_by_fold(p, y, fold)
        y2 = y.copy()
        y2[fold == 2] = 1 - y2[fold == 2]  # flip every outcome in fold 2
        alt = isotonic_by_fold(p, y2, fold)
        np.testing.assert_array_equal(base[fold == 2], alt[fold == 2])
        assert not np.array_equal(base[fold != 2], alt[fold != 2])

    def test_monotone_within_fold(self):
        p, y, fold = dataset_b()
        iso = isotonic_by_fold(p, y, fold)
        for f in np.unique(fold):
            order = np.argsort(p[fold == f])
            assert np.all(np.diff(iso[fold == f][order]) >= -1e-12)

    def test_needs_two_folds(self):
        with pytest.raises(ValueError):
            isotonic_by_fold([0.1, 0.2], [0, 1], [1, 1])


# =========================================================================== #
# 6. group_folds
# =========================================================================== #
class TestGroupFolds:
    def test_no_group_spans_two_folds(self):
        rng = np.random.default_rng(0)
        groups = rng.integers(1000, 1100, size=2000)  # ~100 matches, ~20 shots each
        fold = group_folds(groups, k=5)
        df = pd.DataFrame({"g": groups, "f": fold})
        assert (df.groupby("g")["f"].nunique() == 1).all()

    def test_balanced_over_groups_and_all_folds_used(self):
        groups = np.repeat(np.arange(103), 3)
        fold = group_folds(groups, k=5)
        per_group = pd.DataFrame({"g": groups, "f": fold}).drop_duplicates("g")["f"]
        counts = per_group.value_counts()
        assert sorted(counts.index) == [1, 2, 3, 4, 5]
        assert counts.max() - counts.min() <= 1  # rep_len(1:k, n_groups) then permute

    def test_shape_dtype_and_range(self):
        groups = ["m3", "m1", "m2", "m1", "m3"]
        fold = group_folds(groups, k=3)
        assert fold.shape == (5,) and fold.dtype.kind == "i"
        assert set(fold) <= {1, 2, 3}
        assert fold[0] == fold[4] and fold[1] == fold[3]

    def test_deterministic_and_seed_sensitive(self):
        groups = np.repeat(np.arange(50), 2)
        a = group_folds(groups, k=5, seed=7406)
        b = group_folds(groups, k=5, seed=7406)
        c = group_folds(groups, k=5, seed=1)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_row_order_independent(self):
        groups = np.repeat(np.arange(20), 2)
        perm = np.random.default_rng(3).permutation(groups.size)
        a = group_folds(groups, k=4)
        b = group_folds(groups[perm], k=4)
        np.testing.assert_array_equal(a[perm], b)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            group_folds([], k=5)
        with pytest.raises(ValueError):
            group_folds([1, 2], k=0)


# =========================================================================== #
# 7. abstention_curve
# =========================================================================== #
class TestAbstentionCurve:
    def test_hand_computed(self):
        p = [0.05, 0.3, 0.7, 0.95]
        y = [0, 1, 1, 1]
        df = abstention_curve(p, y, thresholds=[0.0, 0.1, 0.5])
        assert list(df.columns) == ["threshold", "coverage", "n_covered", "accuracy", "log_loss"]
        # t=0: nothing is exactly 0 or 1 -> no coverage, NaN quality
        r0 = df.iloc[0]
        assert r0.coverage == 0 and r0.n_covered == 0
        assert math.isnan(r0.accuracy) and math.isnan(r0.log_loss)
        # t=0.1: covers .05 and .95, both correct, log loss = -log(.95)
        r1 = df.iloc[1]
        assert r1.coverage == pytest.approx(0.5) and r1.n_covered == 2
        assert r1.accuracy == pytest.approx(1.0)
        assert r1.log_loss == pytest.approx(-math.log(0.95), abs=1e-12)
        # t=0.5: covers everything; .3 -> class 0 but y=1 -> 3/4 accuracy
        r2 = df.iloc[2]
        assert r2.coverage == pytest.approx(1.0) and r2.n_covered == 4
        assert r2.accuracy == pytest.approx(0.75)
        assert r2.log_loss == pytest.approx(log_loss(p, y), abs=1e-12)

    def test_coverage_monotone_in_threshold(self):
        p, y, _ = dataset_b()
        df = abstention_curve(p, y, thresholds=np.linspace(0, 0.5, 11))
        assert df["coverage"].is_monotonic_increasing
        assert df["coverage"].iloc[-1] == 1.0

    def test_boundaries_are_inclusive(self):
        df = abstention_curve([0.2, 0.8, 0.5], [0, 1, 1], thresholds=[0.2])
        assert df.loc[0, "n_covered"] == 2


# =========================================================================== #
# input validation shared by the scoring functions
# =========================================================================== #
class TestValidation:
    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            log_loss([0.1, 0.2], [1])

    def test_empty(self):
        with pytest.raises(ValueError):
            brier_score([], [])

    def test_probability_range_for_binning(self):
        with pytest.raises(ValueError):
            expected_calibration_error([1.5], [1])
        with pytest.raises(ValueError):
            brier_decomposition([-0.1, 0.5], [0, 1])

    def test_log_loss_clamps_instead_of_failing(self):
        # R: pmin(pmax(p, eps), 1 - eps) -> finite loss at p = 0 / 1
        assert math.isfinite(log_loss([0.0, 1.0], [1, 0]))
        assert log_loss([0.0], [1]) == pytest.approx(-math.log(1e-15), rel=1e-12)
