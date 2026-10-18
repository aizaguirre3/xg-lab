"""R-vs-Python parity on the real out-of-fold predictions.

`tools/export_parity.R` writes (from the gitignored data/oof_preds.rds):

    data/parity/oof.csv            columns p, y, fold [, p_iso]
    data/parity/metrics_r.csv      columns metric, value
    data/parity/reliability_r.csv  columns bin, mean_pred, obs_rate, n, se (optional)

The scoring metrics (log loss, Brier, ECE, Murphy components) must agree at
1e-6, and so must the metrics of the leakage-safe isotonic recalibration
(R's isoreg/approxfun and scikit-learn's IsotonicRegression fit the same
function, ties included; see varlab.calibration's docstring). Per-row
isotonic predictions are compared at 1e-8 (only floating-point interpolation
noise is expected).

Set VARLAB_PARITY_DIR to point at another directory holding the same files.
All tests skip cleanly when the files are absent.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from varlab import (
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    isotonic_by_fold,
    log_loss,
    reliability_data,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_DIR = Path(os.environ.get("VARLAB_PARITY_DIR", REPO_ROOT / "data" / "parity"))
OOF_CSV = PARITY_DIR / "oof.csv"
METRICS_CSV = PARITY_DIR / "metrics_r.csv"
RELIABILITY_CSV = PARITY_DIR / "reliability_r.csv"

STRICT_TOL = 1e-6
ISO_TOL = 1e-6
ISO_ROW_TOL = 1e-8  # floating-point interpolation noise only

STRICT_METRICS = ("log_loss", "brier", "ece", "reliability", "resolution",
                  "uncertainty", "recomposed")
ISO_METRICS = ("ece_iso", "log_loss_iso", "brier_iso")


@pytest.fixture(scope="module")
def oof():
    if not OOF_CSV.exists() or not METRICS_CSV.exists():
        pytest.skip(f"parity files not present in {PARITY_DIR} (run tools/export_parity.R)")
    df = pd.read_csv(OOF_CSV)
    missing = {"p", "y", "fold"} - set(df.columns)
    assert not missing, f"oof.csv is missing columns {sorted(missing)}"
    return df


@pytest.fixture(scope="module")
def r_metrics(oof):
    m = pd.read_csv(METRICS_CSV)
    assert {"metric", "value"} <= set(m.columns)
    return dict(zip(m["metric"], m["value"].astype(float)))


def test_oof_file_is_sane(oof, r_metrics):
    assert len(oof) > 0
    assert oof["p"].between(0, 1).all()
    assert set(oof["y"].unique()) <= {0, 1}
    if "n" in r_metrics:
        assert int(r_metrics["n"]) == len(oof)


def test_scoring_metrics_match_r_at_1e6(oof, r_metrics):
    p = oof["p"].to_numpy(dtype=float)
    y = oof["y"].to_numpy(dtype=float)
    d = brier_decomposition(p, y, n_bins=10)
    ours = {
        "log_loss": log_loss(p, y),
        "brier": brier_score(p, y),
        "ece": expected_calibration_error(p, y, n_bins=10),
        "reliability": d["reliability"],
        "resolution": d["resolution"],
        "uncertainty": d["uncertainty"],
        "recomposed": d["recomposed"],
    }
    missing = [m for m in STRICT_METRICS if m not in r_metrics]
    assert not missing, f"metrics_r.csv lacks {missing}"
    diffs = {m: abs(ours[m] - r_metrics[m]) for m in STRICT_METRICS}
    bad = {m: v for m, v in diffs.items() if not v <= STRICT_TOL}
    assert not bad, f"R-vs-Python differences above {STRICT_TOL}: {bad}"


def test_reliability_bins_match_r(oof):
    if not RELIABILITY_CSV.exists():
        pytest.skip("reliability_r.csv not present")
    r = pd.read_csv(RELIABILITY_CSV).sort_values("bin").reset_index(drop=True)
    ours = reliability_data(oof["p"].to_numpy(float), oof["y"].to_numpy(float), n_bins=10)
    assert ours["bin"].tolist() == r["bin"].astype(int).tolist()
    assert ours["n"].tolist() == r["n"].astype(int).tolist()
    for col in ("mean_pred", "obs_rate", "se"):
        np.testing.assert_allclose(ours[col].to_numpy(), r[col].to_numpy(float),
                                   rtol=0, atol=STRICT_TOL, err_msg=col)


def test_isotonic_recalibration_matches_r(oof, r_metrics):
    have = [m for m in ISO_METRICS if m in r_metrics]
    if not have:
        pytest.skip("metrics_r.csv has no isotonic metrics")
    p = oof["p"].to_numpy(float)
    y = oof["y"].to_numpy(float)
    fold = oof["fold"].to_numpy()
    iso = isotonic_by_fold(p, y, fold)
    ours = {
        "ece_iso": expected_calibration_error(iso, y, n_bins=10),
        "log_loss_iso": log_loss(iso, y),
        "brier_iso": brier_score(iso, y),
    }
    n_ties = int(len(p) - len(np.unique(p)))
    diffs = {m: abs(ours[m] - r_metrics[m]) for m in have}
    bad = {m: v for m, v in diffs.items() if not v <= ISO_TOL}
    assert not bad, (
        f"isotonic metrics differ from R by more than {ISO_TOL}: {bad} "
        f"({n_ties} tied predictions out of {len(p)})"
    )
    if "p_iso" in oof.columns:
        row_diff = np.abs(iso - oof["p_iso"].to_numpy(float))
        print(f"[parity] isotonic per-row max |diff| = {row_diff.max():.3g}, "
              f"mean = {row_diff.mean():.3g}, ties = {n_ties}/{len(p)}")
        assert row_diff.max() <= ISO_ROW_TOL, (
            f"per-row isotonic predictions differ from R by up to {row_diff.max():.3g}"
        )
