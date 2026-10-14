"""Calibration and evaluation helpers ported from xg-lab's R code.

This module is a line-by-line port of the probabilistic-forecast scoring
helpers in ``R/utils.R`` (``log_loss``, ``brier_score``,
``expected_calibration_error``, ``brier_decomposition``) and the calibration
tooling in ``R/04_calibration.R`` (``reliability_data``,
``isotonic_recalibrate``) plus the match-grouped fold assignment in
``R/03_models.R`` (``make_folds``).

Conventions shared with the R code
----------------------------------
* ``p`` are predicted probabilities, ``y`` are 0/1 outcomes, both 1-D and of
  equal length.
* Equal-width bins use the breaks ``seq(0, 1, length.out = n_bins + 1)`` and
  R's ``cut(..., include.lowest = TRUE, right = TRUE)`` semantics: bin ``k``
  is ``(b[k-1], b[k]]`` except the first, which is ``[b[0], b[1]]``.
* Equal-count bins use R's default ``quantile`` (type 7) with duplicated
  breaks collapsed, then the same right-closed cut.
* Bin codes are 1-based, as in R.

Documented differences from the R implementation
------------------------------------------------
* R returns ``NA`` (or errors) when a probability falls outside ``[0, 1]``
  in the equal-width binning; this port raises ``ValueError``. Empty inputs
  raise ``ValueError`` instead of returning ``NaN``.
* When all predictions are identical, R's ``cut`` errors (a single break is
  interpreted as a bin count); ``reliability_data`` here puts everything in
  one bin.
* ``isotonic_by_fold`` uses scikit-learn's ``IsotonicRegression`` instead of
  R's ``isoreg`` + ``approxfun(rule = 2, ties = "ordered")``. Both fit the
  same function: ``isoreg`` orders tied predictions by ``order(x, -y)``, which
  forces every tie group into one pooled block, exactly what scikit-learn's
  tie averaging does; between fitted points both interpolate linearly and
  beyond the range both hold the end value. Verified on synthetic data with
  and without heavy ties (per-row differences at the 1e-13 level, i.e.
  floating point only). The remaining difference is cosmetic: scikit-learn
  drops interior points that lie on a flat segment, which does not change the
  interpolant.
* ``group_folds`` cannot reproduce R's Mersenne-Twister ``sample`` stream in
  numpy, so the fold *membership* differs from ``make_folds``; the guarantee
  (every group sits in exactly one fold, fold sizes in groups differ by at
  most one) is identical. Groups are assigned in sorted order here, whereas
  R uses order of first appearance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

__all__ = [
    "log_loss",
    "brier_score",
    "expected_calibration_error",
    "brier_decomposition",
    "reliability_data",
    "isotonic_by_fold",
    "group_folds",
    "abstention_curve",
]


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #
def _as_1d_float(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} must not be empty")
    return a


def _check_pair(p, y) -> tuple[np.ndarray, np.ndarray]:
    p = _as_1d_float(p, "p")
    y = _as_1d_float(y, "y")
    if p.shape != y.shape:
        raise ValueError(f"p and y must have the same length ({p.size} != {y.size})")
    return p, y


def _check_prob(p: np.ndarray) -> None:
    if not np.all(np.isfinite(p)):
        raise ValueError("p must be finite")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("p must lie in [0, 1]")


# --------------------------------------------------------------------------- #
# R primitives: seq / cut / quantile / unique
# --------------------------------------------------------------------------- #
def _equal_width_breaks(n_bins: int) -> np.ndarray:
    """``seq(0, 1, length.out = n_bins + 1)`` exactly as R computes it.

    R's ``seq.default`` returns ``c(from, from + (1:(n-1)) * ((to - from)/n), to)``
    so interior breaks are ``i * (1/n_bins)`` and the end points are exact.
    """
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    breaks = np.empty(n_bins + 1, dtype=float)
    breaks[0] = 0.0
    breaks[-1] = 1.0
    if n_bins > 1:
        breaks[1:-1] = np.arange(1, n_bins) * ((1.0 - 0.0) / n_bins)
    return breaks


def _cut_right_closed(x: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """``cut(x, breaks, include.lowest = TRUE, right = TRUE, labels = FALSE)``.

    Returns 1-based integer bin codes: ``breaks[k-1] < x <= breaks[k]`` maps
    to ``k``, and ``x == breaks[0]`` maps to 1 (``include.lowest``).
    ``breaks`` must be sorted and unique.
    """
    breaks = np.asarray(breaks, dtype=float)
    if breaks.size < 2:
        # Degenerate (all predictions identical): one bin. R's cut() would
        # misread a single break as a number of intervals and error.
        return np.ones(x.size, dtype=int)
    if (x < breaks[0]).any() or (x > breaks[-1]).any():
        raise ValueError("values outside the break range")
    code = np.searchsorted(breaks, x, side="left")  # breaks[code-1] < x <= breaks[code]
    code = np.where(code == 0, 1, code)  # x == breaks[0] -> first bin
    return code.astype(int)


def _unique_in_order(codes: np.ndarray) -> np.ndarray:
    """``unique()`` in R: unique values in order of first appearance."""
    _, first = np.unique(codes, return_index=True)
    return codes[np.sort(first)]


def _quantile_type7(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """R's ``quantile(x, probs, type = 7)`` including its index fuzz.

    Ported from ``stats:::quantile.default`` (R 4.x):
    ``nppm = 1 + probs * (n - 1); j = floor(nppm + fuzz); h = nppm - j``
    with ``h`` snapped to 0 when ``|h| < fuzz`` and ``fuzz = 4 * eps``.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    probs = np.asarray(probs, dtype=float)
    fuzz = 4.0 * np.finfo(float).eps
    nppm = 1.0 + probs * (n - 1)  # type 7: a = b = 1
    j = np.floor(nppm + fuzz)
    h = nppm - j
    h = np.where(np.abs(h) < fuzz, 0.0, h)
    j = j.astype(int)
    # R pads the sorted vector with two copies of each end then indexes j+2, j+3
    xp = np.concatenate([x[:1], x[:1], x, x[-1:], x[-1:]])
    lo = xp[j + 1]  # 1-based x[j + 2]
    hi = xp[j + 2]  # 1-based x[j + 3]
    qs = lo.copy()
    qs = np.where(h == 1.0, hi, qs)
    other = (h > 0.0) & (h < 1.0) & (lo != hi)
    qs = np.where(other, (1.0 - h) * lo + h * hi, qs)
    return qs


# --------------------------------------------------------------------------- #
# scoring (R/utils.R)
# --------------------------------------------------------------------------- #
def log_loss(p, y, eps: float = 1e-15) -> float:
    """Mean binary cross-entropy with ``p`` clamped to ``[eps, 1 - eps]``.

    Port of ``log_loss()`` in ``R/utils.R``.
    """
    p, y = _check_pair(p, y)
    p = np.minimum(np.maximum(p, eps), 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(p, y) -> float:
    """Mean squared error of the probability forecast (``R/utils.R``)."""
    p, y = _check_pair(p, y)
    return float(np.mean((p - y) ** 2))


def _bin_summaries(p: np.ndarray, y: np.ndarray, codes: np.ndarray):
    """Yield ``(n_k, mean_p_k, mean_y_k)`` per occupied bin, in R's
    ``unique(bin)`` order so floating-point accumulation matches R."""
    for k in _unique_in_order(codes):
        idx = codes == k
        yield int(idx.sum()), float(p[idx].mean()), float(y[idx].mean())


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins.

    ``ECE = sum_k (n_k / N) * |mean_pred_k - mean_obs_k|`` over occupied bins,
    with the bins defined exactly as in ``R/utils.R`` (``seq(0, 1)`` breaks,
    ``cut(..., include.lowest = TRUE)``, right-closed).
    """
    p, y = _check_pair(p, y)
    _check_prob(p)
    codes = _cut_right_closed(p, _equal_width_breaks(n_bins))
    n = p.size
    ece = 0.0
    for nk, pbar_k, obar_k in _bin_summaries(p, y, codes):
        ece += (nk / n) * abs(pbar_k - obar_k)
    return float(ece)


def brier_decomposition(p, y, n_bins: int = 10) -> dict:
    """Murphy (binned) decomposition of the Brier score.

    Returns ``dict(reliability, resolution, uncertainty, brier, recomposed)``
    where ``recomposed = reliability - resolution + uncertainty``; the identity
    with ``brier`` is exact only in the binning limit, so both are returned.
    Port of ``brier_decomposition()`` in ``R/utils.R``.
    """
    p, y = _check_pair(p, y)
    _check_prob(p)
    codes = _cut_right_closed(p, _equal_width_breaks(n_bins))
    n = p.size
    obar = float(y.mean())
    reliability = 0.0
    resolution = 0.0
    for nk, pbar_k, obar_k in _bin_summaries(p, y, codes):
        reliability += nk * (pbar_k - obar_k) ** 2
        resolution += nk * (obar_k - obar) ** 2
    reliability /= n
    resolution /= n
    uncertainty = obar * (1.0 - obar)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier": brier_score(p, y),
        "recomposed": float(reliability - resolution + uncertainty),
    }


# --------------------------------------------------------------------------- #
# calibration tooling (R/04_calibration.R)
# --------------------------------------------------------------------------- #
def reliability_data(p, y, n_bins: int = 10) -> pd.DataFrame:
    """Reliability-diagram table with equal-count (quantile) bins.

    Port of ``reliability_data()`` in ``R/04_calibration.R`` for a single
    model: breaks are ``unique(quantile(p, seq(0, 1, length.out = n_bins + 1)))``
    (R type 7), bins are right-closed with the lowest included, and the
    returned frame has one row per occupied bin, sorted by bin, with columns
    ``bin, mean_pred, obs_rate, n, se`` where
    ``se = sqrt(max(obs_rate * (1 - obs_rate), 1e-6) / n)``.
    """
    p, y = _check_pair(p, y)
    if not np.all(np.isfinite(p)):
        raise ValueError("p must be finite")
    probs = _equal_width_breaks(n_bins)  # same construction as seq(0, 1, ...)
    q = np.unique(_quantile_type7(p, probs))  # unique() keeps sorted order here
    codes = _cut_right_closed(p, q)
    rows = []
    for k in np.unique(codes):  # group_by sorts the key
        idx = codes == k
        nk = int(idx.sum())
        obs = float(y[idx].mean())
        rows.append(
            {
                "bin": int(k),
                "mean_pred": float(p[idx].mean()),
                "obs_rate": obs,
                "n": nk,
                "se": float(np.sqrt(max(obs * (1.0 - obs), 1e-6) / nk)),
            }
        )
    return pd.DataFrame(rows, columns=["bin", "mean_pred", "obs_rate", "n", "se"])


def isotonic_by_fold(p, y, fold) -> np.ndarray:
    """Leakage-safe isotonic recalibration: for each fold, fit the isotonic
    map on the OTHER folds and apply it to this fold.

    Port of ``isotonic_recalibrate()`` in ``R/04_calibration.R``. The map is a
    scikit-learn ``IsotonicRegression(out_of_bounds="clip")``, i.e. linear
    interpolation between the fitted points and constant extrapolation at the
    ends, which is what R's ``approxfun(..., method = "linear", rule = 2)`` on
    the ``isoreg`` fit does; tied predictions are pooled by both (see the
    module docstring). The result is clipped to ``[0, 1]`` like the R code.
    """
    p, y = _check_pair(p, y)
    fold = np.asarray(fold)
    if fold.shape != p.shape:
        raise ValueError("fold must have the same length as p")
    folds = np.unique(fold)
    if folds.size < 2:
        raise ValueError("need at least two folds")
    out = np.full(p.shape, np.nan)
    for f in folds:
        test = fold == f
        cal = ~test
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(p[cal], y[cal])
        out[test] = np.clip(iso.predict(p[test]), 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# grouped folds (R/03_models.R)
# --------------------------------------------------------------------------- #
def group_folds(groups, k: int = 5, seed: int = 7406) -> np.ndarray:
    """Assign every GROUP (e.g. match) to one of ``k`` folds, like
    ``make_folds()`` in ``R/03_models.R``.

    R does ``sample(rep_len(1:k, n_groups))`` on the unique group ids; this
    does the numpy equivalent (a random permutation of the balanced label
    vector, ``numpy.random.default_rng(seed)``) over the *sorted* unique
    groups. The RNG streams differ, so fold membership is not the same as
    R's, but the guarantees are: every row of a group gets the same fold and
    group counts per fold differ by at most one. Returns 1-based fold ids.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("groups must be a non-empty 1-D array")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    uniq, inverse = np.unique(groups, return_inverse=True)
    labels = np.resize(np.arange(1, k + 1), uniq.size)  # rep_len(1:k, n_groups)
    rng = np.random.default_rng(seed)
    labels = rng.permutation(labels)  # sample(x) = random permutation
    return labels[inverse].astype(int)


# --------------------------------------------------------------------------- #
# abstention / selective prediction
# --------------------------------------------------------------------------- #
def abstention_curve(p, y, thresholds) -> pd.DataFrame:
    """Coverage versus quality when abstaining on uncertain predictions.

    For each margin ``t`` a prediction is *covered* when ``p <= t`` or
    ``p >= 1 - t`` (confidently low or high; ``t`` in ``[0, 0.5]`` is the
    meaningful range, ``t = 0.5`` covers everything). On the covered subset
    the hard decision ``p >= 0.5`` is scored by accuracy and the probabilities
    by ``log_loss``. Returns a frame with ``threshold, coverage, n_covered,
    accuracy, log_loss`` (``NaN`` for the last two when nothing is covered).
    """
    p, y = _check_pair(p, y)
    thresholds = _as_1d_float(thresholds, "thresholds")
    n = p.size
    rows = []
    for t in thresholds:
        covered = (p <= t) | (p >= 1.0 - t)
        n_cov = int(covered.sum())
        if n_cov:
            pc, yc = p[covered], y[covered]
            acc = float(np.mean((pc >= 0.5) == (yc >= 0.5)))
            ll = log_loss(pc, yc)
        else:
            acc = ll = float("nan")
        rows.append(
            {
                "threshold": float(t),
                "coverage": n_cov / n,
                "n_covered": n_cov,
                "accuracy": acc,
                "log_loss": ll,
            }
        )
    return pd.DataFrame(
        rows, columns=["threshold", "coverage", "n_covered", "accuracy", "log_loss"]
    )
