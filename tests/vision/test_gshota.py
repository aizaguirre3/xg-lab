"""GS-HOTA maths on toy tracks (no data)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from varlab_vision.gshota import ALPHAS, gs_hota, gs_hota_clips, id_sim_matrix, loc_sim, similarity_matrix

COLS = ["frame", "track_id", "role", "team", "jersey", "x_pitch", "y_pitch"]


def two_tracks(n_frames: int = 4) -> pd.DataFrame:
    rows = []
    for f in range(1, n_frames + 1):
        rows.append([f, 1, "player", "left", "7", -10.0 + f, 0.0])
        rows.append([f, 2, "player", "right", "10", 10.0, 5.0 + f])
    return pd.DataFrame(rows, columns=COLS)


def test_loc_sim_kernel():
    assert loc_sim(0.0) == pytest.approx(1.0)
    assert loc_sim(5.0) == pytest.approx(0.05)
    assert loc_sim(2.0) == pytest.approx(math.exp(math.log(0.05) * 4 / 25))
    assert np.all(np.diff(loc_sim(np.array([0, 1, 2, 3, 10]))) < 0)


def test_alphas_cover_0_05_to_0_95():
    assert ALPHAS[0] == 0.05 and ALPHAS[-1] == 0.95 and len(ALPHAS) == 19


def test_perfect_prediction_scores_one():
    gt = two_tracks()
    r = gs_hota(gt, gt.copy())
    assert r.hota == pytest.approx(1.0) and r.deta == pytest.approx(1.0) and r.assa == pytest.approx(1.0)
    assert r.loca == pytest.approx(1.0)
    assert r.n_gt == r.n_pred == 8 and r.n_frames == 4


def test_prediction_ids_need_not_equal_gt_ids():
    gt = two_tracks()
    pr = gt.copy()
    pr["track_id"] = pr["track_id"].map({1: 77, 2: 5})
    assert gs_hota(gt, pr).hota == pytest.approx(1.0)


def test_location_error_lowers_deta_monotonically():
    gt = two_tracks()
    scores = []
    for shift in (0.0, 1.0, 2.0, 4.0, 6.0):
        pr = gt.copy()
        pr["x_pitch"] += shift
        scores.append(gs_hota(gt, pr).hota)
    assert scores[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(scores, scores[1:]))
    assert scores[-1] == pytest.approx(0.0)  # 6 m > tau: below every alpha
    # 2 m: LocSim 0.619 -> matched for alphas below it (12 of 19), unmatched above
    pr = gt.copy(); pr["x_pitch"] += 2.0
    r = gs_hota(gt, pr)
    expected_deta = sum(1 for a in ALPHAS if a < 0.619) / len(ALPHAS)
    assert r.deta == pytest.approx(expected_deta, abs=1e-9)
    # ids are consistent wherever a match exists; HOTA defines AssA as 0 at alphas with no TP
    matched = r.per_alpha[r.per_alpha["TP"] > 0]
    assert (matched["AssA"] == 1.0).all() and len(matched) == 12
    assert r.assa == pytest.approx(expected_deta, abs=1e-9)


def test_wrong_team_is_a_false_positive():
    gt = two_tracks()
    pr = gt.copy()
    pr["team"] = pr["team"].map({"left": "right", "right": "left"})
    r = gs_hota(gt, pr)
    assert r.hota == pytest.approx(0.0)
    assert r.per_alpha["FP"].iloc[0] == 8 and r.per_alpha["FN"].iloc[0] == 8


def test_wrong_jersey_or_role_breaks_the_match():
    gt = two_tracks()
    # the bad track is one FN and one FP per frame next to one TP: DetA = 1 / 3
    pr = gt.copy(); pr.loc[pr["track_id"] == 1, "jersey"] = "8"
    assert gs_hota(gt, pr).deta == pytest.approx(1 / 3)
    pr = gt.copy(); pr.loc[pr["track_id"] == 1, "role"] = "goalkeeper"
    assert gs_hota(gt, pr).deta == pytest.approx(1 / 3)
    # a missing jersey is a mismatch too: the metric expects jersey recognition
    pr = gt.copy(); pr["jersey"] = None
    assert gs_hota(gt, pr).hota == pytest.approx(0.0)


def test_id_switch_keeps_deta_but_lowers_assa():
    gt = two_tracks(n_frames=6)
    pr = gt.copy()
    # swap the two prediction ids from frame 4 on
    late = pr["frame"] >= 4
    pr.loc[late, "track_id"] = pr.loc[late, "track_id"].map({1: 2, 2: 1})
    r = gs_hota(gt, pr)
    assert r.deta == pytest.approx(1.0)
    assert 0.3 < r.assa < 1.0
    assert r.hota == pytest.approx(math.sqrt(r.assa))


def test_null_attribute_handling_for_referee():
    gt = pd.DataFrame([[1, 3, "referee", None, None, 0.0, 0.0]], columns=COLS)
    pr = pd.DataFrame([[1, 9, "referee", None, "5", 0.0, 0.0]], columns=COLS)
    assert id_sim_matrix(gt, pr, null_is_value=True)[0, 0] == 0.0
    assert id_sim_matrix(gt, pr, null_is_value=False)[0, 0] == 1.0
    assert gs_hota(gt, pr).hota == pytest.approx(0.0)
    assert gs_hota(gt, pr, null_is_value=False).hota == pytest.approx(1.0)
    ok = pd.DataFrame([[1, 9, "referee", None, None, 0.0, 0.0]], columns=COLS)
    assert gs_hota(gt, ok).hota == pytest.approx(1.0)


def test_nan_pitch_position_never_matches():
    gt = two_tracks(1)
    pr = gt.copy(); pr.loc[0, "x_pitch"] = float("nan")
    s = similarity_matrix(gt, pr)
    assert s[0, 0] == 0.0 and s[1, 1] == pytest.approx(1.0)


def test_ball_excluded_by_default_and_roles_filter():
    gt = pd.concat([two_tracks(2), pd.DataFrame([[1, 4, "ball", None, None, 0.0, 0.0]], columns=COLS)])
    pr = two_tracks(2)
    assert gs_hota(gt, pr).hota == pytest.approx(1.0)
    only_left = gs_hota(gt, pr.assign(team="left"), roles=("player",))
    assert only_left.deta == pytest.approx(1 / 3)


def test_extra_prediction_frames_count_as_fp_unless_restricted():
    gt = two_tracks(2)
    pr = two_tracks(3)
    r = gs_hota(gt, pr)
    assert r.per_alpha["FP"].iloc[0] == 2 and r.n_frames == 3
    r2 = gs_hota(gt, pr, frames=[1, 2])
    assert r2.hota == pytest.approx(1.0) and r2.n_frames == 2


def test_empty_prediction_is_zero_and_table_helper():
    gt = two_tracks(2)
    empty = gt.iloc[0:0]
    r = gs_hota(gt, empty)
    assert r.hota == 0.0 and r.n_pred == 0
    table = gs_hota_clips([("a", gt, gt.copy()), ("b", gt, empty)])
    assert table["clip"].tolist() == ["a", "b", "MEAN"]
    assert table.loc[2, "GS-HOTA"] == pytest.approx(0.5)
