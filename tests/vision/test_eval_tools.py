"""The GSR scoring tool's metric code on tiny hand-built tables (no data needed)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tool(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(items):
    """items: (frame, track_id, role, team, x, y, w, h)"""
    return pd.DataFrame(items, columns=["frame", "track_id", "role", "team", "x", "y", "w", "h"])


def test_iou_matrix():
    ev = _tool("eval_gsr_tracking")
    a = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], float)
    b = np.array([[5, 0, 10, 10], [0, 0, 10, 10]], float)
    m = ev.iou_matrix(a, b)
    assert m[0, 1] == pytest.approx(1.0) and m[0, 0] == pytest.approx(50 / 150) and m[1].max() == 0.0
    assert ev.iou_matrix(a[:0], b).shape == (0, 2)


def test_score_clip_counts_matches_switches_and_identity():
    ev = _tool("eval_gsr_tracking")
    gt = _rows([(1, 1, "player", "left", 0, 0, 10, 20), (2, 1, "player", "left", 1, 0, 10, 20),
                (3, 1, "player", "left", 2, 0, 10, 20), (1, 2, "referee", None, 50, 0, 10, 20)])
    pr = _rows([(1, 5, "player", "right", 0, 0, 10, 20), (2, 5, "player", "right", 1, 0, 10, 20),
                (3, 6, "player", "right", 2, 0, 10, 20),   # identity switch on frame 3
                (2, 9, "player", None, 300, 300, 10, 20)])  # a false positive
    s = ev.score_clip(gt, pr)
    assert (s["tp"], s["fp"], s["fn"], s["idsw"]) == (3, 1, 1, 1)
    r = ev.rates(s)
    assert r["precision"] == pytest.approx(0.75) and r["recall"] == pytest.approx(0.75)
    assert r["mota"] == pytest.approx(1 - (1 + 1 + 1) / 4)
    assert r["idf1"] == pytest.approx(2 * 2 / (4 + 4))  # gt track 1 pairs with pred 5 for two frames
    assert r["recall_by_role"]["referee"] == 0.0 and r["recall_by_role"]["player"] == 1.0
    assert r["team_accuracy"] == 1.0  # left/right naming is free: only the grouping counts


def test_ball_found_within_the_pixel_radius():
    ev = _tool("eval_gsr_tracking")
    gt = _rows([(1, 0, "ball", None, 100, 100, 4, 4), (2, 0, "ball", None, 200, 100, 4, 4)])
    pr = _rows([(1, 0, "ball", None, 108, 100, 4, 4), (2, 0, "ball", None, 260, 100, 4, 4)])
    r = ev.rates(ev.score_clip(gt, pr, ball_px=15.0))
    assert r["ball_recall"] == 0.5 and r["ball_precision"] == 0.5


def test_calibration_summary_reports_the_precision_of_the_claim():
    ev = _tool("eval_gsr_calibration")
    rows = [
        {"status": "calibrated", "seconds": 1.0, "median_m": 0.4, "errors_m": [0.3, 0.4, 0.5]},
        {"status": "calibrated", "seconds": 1.0, "median_m": 2.5, "errors_m": [2.0, 2.5, 3.0]},
        {"status": "calibrated", "seconds": 1.0, "median_m": 0.8, "errors_m": [0.8, 0.9]},
        {"status": "uncalibrated", "seconds": 2.0, "refused": True},
        {"status": "uncalibrated", "seconds": 2.0},
    ]
    s = ev.summary_of(rows, n_labelled_total=14)
    assert s["frames"] == 5 and s["calibrated_share"] == 0.6 and s["refused"] == 1
    assert s["precision_1m"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["claimed_by_median"] == {"under_1m": 2, "1_to_5m": 1, "over_5m": 0}
    assert s["within_1m_of_all_labelled_boxes"] == pytest.approx(5 / 14, abs=1e-4)
    assert ev.summary_of([], 0)["precision_1m"] is None
