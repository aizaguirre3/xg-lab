"""Calibration tests on the synthetic clip (rendered pitch markings under a known
camera; no SoccerNet data, no GPU).

The markings are perfect white lines on flat green, so these numbers are an
upper bound on real broadcast accuracy, which is unmeasured. What is tested:
the pitch model's geometry, that the global search recovers the known
homography on a wide and on a medium shot, the explicit "uncalibrated" state
when no markings are visible, tracking under a mild pan, and that a shot cut
resets the state instead of blending two cameras.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from varlab_vision import calibration as cal  # noqa: E402
from varlab_vision.pitch_model import DEFAULT_PITCH, X_CONST, Y_CONST, PitchModel  # noqa: E402
from varlab_vision.loaders import objects_frame, read_gsr_clip  # noqa: E402
from varlab_vision.synthetic import medium_camera, render_synthetic_clip, wide_camera  # noqa: E402

W, H = 1280, 720  # close to the 1920x1080 of real clips; errors scale with pixel size


# ----------------------------------------------------------------------------
# Pitch model
# ----------------------------------------------------------------------------


def test_pitch_model_geometry():
    m = PitchModel()
    assert (m.length, m.width) == (105.0, 68.0)
    xs = [ln.value for ln in m.lines_by_kind(X_CONST)]
    ys = [ln.value for ln in m.lines_by_kind(Y_CONST)]
    assert xs == [-52.5, -47.0, -36.0, 0.0, 36.0, 47.0, 52.5]
    assert ys == [-34.0, -20.16, -9.16, 9.16, 20.16, 34.0]
    kp = m.keypoints()
    assert kp["centre_mark"] == (0.0, 0.0)
    assert kp["penalty_spot_left"] == (-41.5, 0.0) and kp["penalty_spot_right"] == (41.5, 0.0)
    assert kp["corner_near_right"] == (52.5, 34.0)  # near = toward the camera = +y
    assert kp["penalty_left_front_near"] == (-36.0, 20.16)
    assert kp["goal_area_right_front_far"] == (47.0, -9.16)
    # every segment endpoint lies on its line
    for ln in m.lines():
        for p0, p1 in ln.segments:
            for p in (p0, p1):
                assert abs(np.array([p[0], p[1], 1.0]) @ ln.homogeneous) < 1e-9
    # the penalty arc stops where it meets the penalty-area front
    arc = [a for a in m.arcs() if a.name == "penalty_arc_left"][0]
    pts = m.arc_points(arc, 1.0)
    assert pts[:, 0].min() == pytest.approx(-36.0, abs=1e-6)
    assert m.contains(np.array([[0, 0], [52.4, 33.9], [53.0, 0.0]])).tolist() == [True, True, False]


def test_pitch_model_matches_r_engine_centre_origin():
    """``to_attacking_frame(origin = "centre")`` adds half the sizes; mirror the check."""
    from varlab_vision.pitch_model import to_corner_origin

    out = to_corner_origin(np.array([[-52.5, -34.0], [0.0, 0.0]]))
    assert out.tolist() == [[0.0, 0.0], [52.5, 34.0]]


# ----------------------------------------------------------------------------
# Line extraction on a rendered frame
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide_clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("wide")
    return render_synthetic_clip(out, n_frames=3, width=W, height=H, seed=1, name="SYN-W")


@pytest.fixture(scope="module")
def medium_clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("medium")
    return render_synthetic_clip(out, n_frames=3, width=W, height=H, seed=2, name="SYN-M",
                                 camera=medium_camera(W, H))


def _frame(clip, i):
    return cv2.imread(str(clip.image_dir / f"{i:06d}.jpg"))


def test_extract_lines_finds_the_markings(wide_clip):
    frame = _frame(wide_clip, 1)
    mask = cal.white_line_mask(frame)
    lines = cal.extract_lines(mask)
    assert 8 <= len(lines) <= 24
    # every extracted line lies on a model line under the true homography
    truth = cal.match_lines(np.linalg.inv(wide_clip.homography), lines, DEFAULT_PITCH, tau_px=4.0)
    assert len(truth) >= 8
    assert {m.model_name for m in truth} >= {"touchline_near", "touchline_far", "halfway",
                                              "goal_line_left", "goal_line_right"}
    # the long ones are the touchlines and they are sub-pixel accurate
    assert max(m.distance_px for m in truth) < 4.0


def test_scoring_ignores_the_scale_and_sign_of_the_homography(wide_clip):
    """A homography is defined up to scale and sign; the orientation checks must not care.
    (``np.linalg.inv`` of a ``h33 = 1`` matrix has an arbitrary sign of ``h33``.)"""
    lines = cal.extract_lines(cal.white_line_mask(_frame(wide_clip, 1)))
    h = np.linalg.inv(wide_clip.homography)
    counts = {k: len(cal.match_lines(k * h, lines, DEFAULT_PITCH, tau_px=4.0)) for k in (1.0, -1.0, 3.7, -0.02)}
    assert counts[1.0] >= 8
    assert len(set(counts.values())) == 1, counts
    # a mirrored pitch (x flipped) must still be rejected, whatever the normalisation
    mirror = h @ np.diag([-1.0, 1.0, 1.0])
    assert len(cal.match_lines(mirror, lines, DEFAULT_PITCH, tau_px=4.0)) == 0
    assert len(cal.match_lines(-mirror, lines, DEFAULT_PITCH, tau_px=4.0)) == 0


def test_one_line_per_marking(wide_clip):
    """Peeling: a thick marking must not come back as several near-identical lines. Two
    lines may lie on the same marking only when a player hides the stretch between them."""
    lines = cal.extract_lines(cal.white_line_mask(_frame(wide_clip, 1)))
    h_px2m = wide_clip.homography
    truth = cal.match_lines(np.linalg.inv(h_px2m), lines, DEFAULT_PITCH, tau_px=1.5)
    by_name = {}
    for m in truth:
        by_name.setdefault(m.model_name, []).append(m.image_index)
    assert {"touchline_near", "touchline_far", "halfway", "goal_line_left", "goal_line_right"} <= set(by_name)
    for name, idx in by_name.items():
        kind = DEFAULT_PITCH.line(name).kind
        spans = []
        for i in idx:
            xy = cal.apply_homography(h_px2m, lines[i].endpoints())
            t = xy[:, 1] if kind == X_CONST else xy[:, 0]
            spans.append((float(t.min()), float(t.max())))
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans[:-1], spans[1:]):
            assert b0 >= a1 - 0.5, (name, spans)  # no two lines cover the same stretch (0.5 m of slack)
    # and the sub-pixel fits sit on the truth
    assert max(m.distance_px for m in truth) < 1.5


# ----------------------------------------------------------------------------
# Global search: recover the known homography
# ----------------------------------------------------------------------------


def _accuracy(clip, res, frame_idx):
    """(keypoint errors, foot errors) in metres for the visible keypoints and the actors."""
    cam = clip.cameras[frame_idx - 1]
    names, kp = DEFAULT_PITCH.keypoint_array()
    px, depth = cam.project(kp)
    vis = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < clip.width) & (px[:, 1] >= 0) & (px[:, 1] < clip.height)
    kp_err = np.linalg.norm(res.project(px[vis]) - kp[vis], axis=1)
    df = objects_frame(read_gsr_clip(clip.labels_path))
    df = df[df["frame"] == frame_idx]
    feet = np.stack([df["x"] + df["w"] / 2, df["y"] + df["h"]], axis=1)
    foot_err = np.linalg.norm(res.project(feet) - df[["x_pitch", "y_pitch"]].to_numpy(), axis=1)
    return kp_err, foot_err


@pytest.mark.parametrize("which", ["wide", "medium"])
def test_global_search_recovers_the_homography(which, wide_clip, medium_clip):
    clip = wide_clip if which == "wide" else medium_clip
    res = cal.calibrate_frame(_frame(clip, 1))
    assert res.status == cal.STATUS_CALIBRATED and res.ok
    assert res.n_inliers >= 8
    kinds = {ln.name: ln.kind for ln in DEFAULT_PITCH.lines()}
    assert len({m.model_name for m in res.matches if kinds[m.model_name] == X_CONST}) >= 2
    assert len({m.model_name for m in res.matches if kinds[m.model_name] == Y_CONST}) >= 2
    assert res.fit_error_px < 2.5
    kp_err, foot_err = _accuracy(clip, res, 1)
    # Achieved on 2026-10-19 at 1280x720 (this fixture): keypoints median 0.013 m (wide) /
    # 0.006 m (medium), max 0.028 / 0.011 m; actor feet max 0.019 / 0.004 m. Over six seeds the
    # worst keypoint max is 0.040 m (wide) and 0.011 m (medium). The bounds below are a few
    # times that, not the result; the loose original ones were 0.25 m median and 0.5 m max.
    assert np.median(kp_err) < 0.05
    assert kp_err.max() < 0.15
    assert foot_err.max() < 0.15
    # projecting the four pitch corners back and forth is the identity
    corners = np.array([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34]], float)
    px = cal.apply_homography(res.h_m2px, corners)
    assert np.abs(cal.apply_homography(res.h_px2m, px) - corners).max() < 1e-6


def test_result_projection_api(wide_clip):
    res = cal.calibrate_frame(_frame(wide_clip, 1))
    boxes = np.array([[100, 100, 130, 160], [600, 300, 630, 360]], float)
    out = res.project_boxes(boxes)
    assert set(out) == {"left", "right", "middle"} and out["middle"].shape == (2, 2)
    assert np.allclose(out["middle"], (out["left"] + out["right"]) / 2)
    d = res.as_dict()
    assert d["status"] == "calibrated" and d["n_inliers"] == res.n_inliers
    assert isinstance(d["homography_px2m"], list) and len(d["homography_px2m"]) == 3


@pytest.mark.parametrize("thickness", [8, 14])
def test_thick_strokes_are_fitted_over_their_whole_width(tmp_path: Path, thickness):
    """Near-camera markings on a 1080p broadcast frame are many pixels wide; a fit band
    narrower than the stroke lets the angle wander inside it (0.3 degrees measured)."""
    clip = render_synthetic_clip(tmp_path, n_frames=1, width=W, height=H, seed=8, name="SYN-THICK",
                                 camera=medium_camera(W, H), line_thickness=thickness)
    res = cal.calibrate_frame(_frame(clip, 1))
    assert res.status == cal.STATUS_CALIBRATED
    kp_err, foot_err = _accuracy(clip, res, 1)
    assert np.median(kp_err) < 0.25 and kp_err.max() < 0.5 and foot_err.max() < 0.5


# ----------------------------------------------------------------------------
# The explicit uncalibrated state
# ----------------------------------------------------------------------------


def test_no_markings_is_uncalibrated(tmp_path: Path):
    clip = render_synthetic_clip(tmp_path, n_frames=1, width=640, height=360, seed=3, with_markings=False)
    res = cal.calibrate_frame(_frame(clip, 1))
    assert res.status == cal.STATUS_UNCALIBRATED and not res.ok
    assert np.isnan(res.project(np.array([[10.0, 10.0]]))).all()
    assert np.isnan(res.project_boxes(np.array([[0, 0, 10, 10]]))["middle"]).all()
    assert "need 4" in res.note


def test_too_few_lines_is_uncalibrated():
    """Three straight lines can never constrain a homography: no guess is returned."""
    frame = np.full((360, 640, 3), (40, 150, 40), np.uint8)
    for y in (100, 200, 300):
        cv2.line(frame, (0, y), (639, y), (255, 255, 255), 3)
    res = cal.calibrate_frame(frame)
    assert res.status == cal.STATUS_UNCALIBRATED
    assert res.n_lines == 3


# ----------------------------------------------------------------------------
# Tracking, smoothing and shot cuts
# ----------------------------------------------------------------------------


def test_tracking_follows_a_mild_pan(tmp_path: Path):
    clip = render_synthetic_clip(tmp_path, n_frames=8, width=W, height=H, seed=4, name="SYN-PAN",
                                 camera=medium_camera(W, H, look_x=-20.0), pan_m_per_frame=0.8)
    assert not clip.is_static
    fc = cal.FrameCalibrator()
    results = [fc.update(_frame(clip, i)) for i in range(1, 9)]
    assert results[0].status == cal.STATUS_CALIBRATED
    assert all(r.status == cal.STATUS_TRACKED for r in results[1:])
    assert not any(r.shot_cut for r in results)
    for i, r in enumerate(results, start=1):
        kp_err, foot_err = _accuracy(clip, r, i)
        assert kp_err.max() < 0.2 and foot_err.max() < 0.2  # measured worst: 0.060 m keypoints, 0.040 m feet


@pytest.mark.parametrize("pan", [1.5, 3.0])
def test_tracking_has_no_lag_on_a_steady_pan(tmp_path: Path, pan):
    """The filter extrapolates a steady pan instead of lagging it: a linear blend of
    consecutive homographies was 1.8 m behind a 1.5 m/frame pan. After the start-up
    frames every frame must be tracked and within 0.3 m."""
    clip = render_synthetic_clip(tmp_path, n_frames=12, width=W, height=H, seed=4, name=f"SYN-PAN{pan}",
                                 camera=medium_camera(W, H, look_x=-20.0), pan_m_per_frame=pan)
    fc = cal.FrameCalibrator()
    results = [fc.update(_frame(clip, i)) for i in range(1, 13)]
    assert all(r.status == cal.STATUS_TRACKED for r in results[3:])
    assert not any(r.shot_cut for r in results)
    for i, r in enumerate(results, start=1):
        kp_err, foot_err = _accuracy(clip, r, i)
        assert kp_err.max() < 0.3, (i, kp_err.max())
        assert len(foot_err) == 0 or foot_err.max() < 0.3


def test_tracking_does_not_lock_onto_the_wrong_lines_on_a_very_fast_pan(tmp_path: Path):
    """Between frames the camera moves further than the tracking tolerance, so the lines
    can snap onto their neighbours with a fine residual (6 m wrong, measured). Fewer
    inliers than the last frame sends it to the global search instead."""
    clip = render_synthetic_clip(tmp_path, n_frames=8, width=W, height=H, seed=4, name="SYN-FASTPAN",
                                 camera=medium_camera(W, H, look_x=-20.0), pan_m_per_frame=6.0)
    fc = cal.FrameCalibrator()
    for i in range(1, 9):
        r = fc.update(_frame(clip, i))
        kp_err, _ = _accuracy(clip, r, i)
        assert r.ok and kp_err.max() < 0.5, (i, r.status, kp_err.max())


def test_shot_cut_resets_the_calibration(tmp_path: Path):
    """Frames 1-4 from the wide camera, 5-8 from the medium camera on different grass:
    the detector must flag the cut, and the solution after it must be the new camera's,
    not a blend of the two."""
    clip = render_synthetic_clip(tmp_path, n_frames=8, width=W, height=H, seed=5, name="SYN-CUT",
                                 cut_at_frame=5, green_after_cut=(60, 110, 30))
    fc = cal.FrameCalibrator()
    results = [fc.update(_frame(clip, i)) for i in range(1, 9)]
    assert [r.shot_cut for r in results] == [False] * 4 + [True] + [False] * 3
    assert results[4].status == cal.STATUS_CALIBRATED  # a fresh global search, not tracking
    assert results[3].status == cal.STATUS_TRACKED
    for i in (4, 5, 8):
        kp_err, _ = _accuracy(clip, results[i - 1], i)
        assert kp_err.max() < 0.2  # measured 0.045 / 0.018 / 0.017 m
    # the two cameras really are different, so blending would have failed the bound above
    assert not np.allclose(clip.homographies[3], clip.homographies[4], atol=1e-3)


def test_cut_detector_has_margin_on_both_sides(tmp_path: Path):
    """Largest within-shot value on a fast pan versus the cut, against the thresholds."""
    pan = render_synthetic_clip(tmp_path, n_frames=8, width=W, height=H, seed=4, name="SYN-PAN-M",
                                camera=medium_camera(W, H, look_x=-20.0), pan_m_per_frame=3.0)
    cut = render_synthetic_clip(tmp_path, n_frames=8, width=W, height=H, seed=5, name="SYN-CUT-M",
                                cut_at_frame=5, green_after_cut=(60, 110, 30))

    def changes(clip):
        sigs = [cal.shot_signature(_frame(clip, i)) for i in range(1, 9)]
        return np.array([cal.shot_change(a, b) for a, b in zip(sigs[:-1], sigs[1:])])

    within, across = changes(pan), changes(cut)
    # within a shot both measures stay at least 3x below their thresholds
    assert within[:, 0].max() < cal.CUT_GREY_THRESHOLD / 3
    assert within[:, 1].max() < cal.CUT_HIST_THRESHOLD / 3
    # at the cut the colour distance clears its threshold with room to spare, on its own
    assert across[3, 1] > 1.5 * cal.CUT_HIST_THRESHOLD
    assert (across[[0, 1, 2, 4, 5, 6]] < [cal.CUT_GREY_THRESHOLD / 3, cal.CUT_HIST_THRESHOLD / 3]).all()


def test_a_new_frame_size_restarts_the_tracker(tmp_path: Path, wide_clip):
    """The filter state is in units of the frame size; a different video must not inherit it."""
    small = render_synthetic_clip(tmp_path, n_frames=2, width=640, height=360, seed=9, name="SYN-SMALL",
                                  camera=medium_camera(640, 360))
    fc = cal.FrameCalibrator()
    fc.update(_frame(wide_clip, 1))
    assert fc.update(_frame(wide_clip, 2)).status == cal.STATUS_TRACKED
    res = fc.update(_frame(small, 1))
    assert res.status == cal.STATUS_CALIBRATED and not res.shot_cut
    kp_err, _ = _accuracy(small, res, 1)
    assert kp_err.max() < 0.5


def test_tracking_loss_falls_back_to_search_without_a_pixel_cut(tmp_path: Path):
    """Same cut but without a grass change: the frame difference may stay small, and
    tracking from the stale prior must fail over to the global search."""
    clip = render_synthetic_clip(tmp_path, n_frames=6, width=W, height=H, seed=6, name="SYN-CUT2",
                                 cut_at_frame=4)
    fc = cal.FrameCalibrator(cut_threshold=1e9)  # disable the pixel detector on purpose
    results = [fc.update(_frame(clip, i)) for i in range(1, 7)]
    assert results[0].status == cal.STATUS_CALIBRATED
    assert results[3].status == cal.STATUS_CALIBRATED
    assert not any(r.shot_cut for r in results)
    kp_err, _ = _accuracy(clip, results[3], 4)
    assert kp_err.max() < 0.2  # measured 0.011 m


# ----------------------------------------------------------------------------
# The Colab probe's clip
# ----------------------------------------------------------------------------


def test_probe_clip_has_every_actor_in_frame_and_none_overlapping(tmp_path: Path):
    """``render_synthetic_clip(out, n_frames=30, width=1920, height=1080, n_per_team=5,
    box_wh=(60, 140), seed=0)`` is what the Colab probe renders: 10 players and the referee,
    all in frame in every frame and never overlapping (the smoke detector would merge them)."""
    clip = render_synthetic_clip(tmp_path, n_frames=30, width=1920, height=1080, n_per_team=5,
                                 box_wh=(60, 140), seed=0)
    df = objects_frame(read_gsr_clip(clip.labels_path))
    per_frame = df.groupby("frame").size()
    assert (per_frame == 11).all() and len(per_frame) == 30
    for _, g in df.groupby("frame"):
        x0, y0 = g["x"].to_numpy(), g["y"].to_numpy()
        x1, y1 = x0 + g["w"].to_numpy(), y0 + g["h"].to_numpy()
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                assert x1[a] <= x0[b] or x1[b] <= x0[a] or y1[a] <= y0[b] or y1[b] <= y0[a]
