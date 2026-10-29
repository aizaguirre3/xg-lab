"""Tests for what real broadcast frames need from the calibration (no SoccerNet data, no GPU).

Each mechanism added for the SoccerNet GSR frames is pinned on frames rendered here: the
contrast-based marking mask (faint lines found, white shirts and board text left out), the
curve test on candidate lines, the centre circle as an ellipse (a view with only the halfway
line, one touchline and the circle is solved from the circle), and the verification that
refuses a solution instead of reporting a wrong one.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from varlab_vision import calibration as cal  # noqa: E402
from varlab_vision.pitch_model import DEFAULT_PITCH  # noqa: E402
from varlab_vision.synthetic import GREEN_BGR, BroadcastCamera, wide_camera  # noqa: E402

W, H = 1280, 720


def _grass(seed: int = 0, noise: float = 6.0) -> np.ndarray:
    """Textured grass: flat green with pixel noise, as a JPEG-decoded frame has."""
    rng = np.random.default_rng(seed)
    frame = np.empty((H, W, 3), np.float32)
    frame[:] = GREEN_BGR
    frame += rng.normal(0.0, noise, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _draw(frame: np.ndarray, cam: BroadcastCamera, colour=(255, 255, 255), thickness: int = 3,
          keep=None) -> None:
    """Draw the pitch markings (or only the polylines named in ``keep``) under ``cam``, in place."""
    for name, poly in DEFAULT_PITCH.polylines(arc_step_deg=2.0):
        if keep is not None and name not in keep:
            continue
        dense = np.concatenate([np.linspace(a, b, max(2, int(math.ceil(np.linalg.norm(b - a))) + 1))
                                for a, b in zip(poly[:-1], poly[1:])])
        px, depth = cam.project(dense)
        fp = np.round(px * 16.0).astype(np.int64)
        for i in range(len(px) - 1):
            if depth[i] > 0 and depth[i + 1] > 0 and np.abs(px[i:i + 2]).max() < 8 * W:
                cv2.line(frame, tuple(int(v) for v in fp[i]), tuple(int(v) for v in fp[i + 1]), colour,
                         thickness=thickness, lineType=cv2.LINE_AA, shift=4)


def _centre_camera() -> BroadcastCamera:
    """A main-camera view of the centre: the halfway line, the far touchline and the centre circle, nothing else."""
    return BroadcastCamera(width=W, height=H, hfov_deg=40.0, position=(0.0, 64.0, 18.0), look_at=(0.0, 5.0))


def _keypoint_error(cam: BroadcastCamera, res: cal.CalibrationResult) -> float:
    """Largest error (metres) over a grid of pitch points in view."""
    xs, ys = np.meshgrid(np.linspace(-30, 30, 13), np.linspace(-30, 30, 13))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    px, depth = cam.project(pts)
    vis = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    return float(np.linalg.norm(res.project(px[vis]) - pts[vis], axis=1).max())


# ----------------------------------------------------------------------------
# 1. Marking pixels by local contrast
# ----------------------------------------------------------------------------


def test_marking_mask_finds_a_faint_line_and_leaves_out_shirts_and_boards():
    frame = _grass(seed=1)
    frame[:120] = (60, 40, 160)  # an advertising board above the grass ...
    cv2.putText(frame, "SPONSOR", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)  # ... with white text
    faint = tuple(int(c) + 35 for c in GREEN_BGR)  # a distant line: 35 grey levels over the grass
    cv2.line(frame, (0, 400), (W - 1, 380), faint, 2)
    cv2.line(frame, (600, 130), (640, 719), (255, 255, 255), 4)
    shirt = np.array([[200, 450, 240, 560]])  # a white shirt, inside a person box
    frame[460:520, 205:235] = 255
    m = cal.marking_mask(frame, person_boxes=shirt)
    on_faint = [m[int(round(400 - 20 * x / (W - 1))) - 2:int(round(400 - 20 * x / (W - 1))) + 3, x].any()
                for x in range(0, W, 8) if not 560 <= x <= 680]
    assert np.mean(on_faint) > 0.9
    assert m[:120].sum() == 0  # the board text is outside the grass
    assert m[450:560, 200:240].sum() == 0  # the shirt is inside a person box
    assert m[300, 615:625].any()  # the bright line is found too
    # the old fixed-threshold mask misses the faint line and takes the shirt
    old = cal.white_line_mask(frame)
    assert np.mean([old[396:405, x].any() for x in range(0, 500, 8)]) < 0.1
    assert old[460:520, 205:235].sum() > 0
    # few false pixels on textured grass
    assert m[150:350, 700:1200].mean() < 0.01


def test_grass_region_fills_the_markings_and_players_and_stops_at_the_boards():
    frame = _grass(seed=2)
    frame[:100] = (90, 90, 90)
    _draw(frame, wide_camera(W, H), thickness=14)  # thick enough to cut the grass in a naive labelling
    frame[400:500, 300:340] = (30, 30, 220)  # a player
    g = cal.grass_region(frame)
    assert g[:95].mean() < 0.02
    assert g[110:].mean() > 0.97
    assert g[420:480, 310:330].all()


def test_person_box_mask_pads_and_clips():
    m = cal.person_box_mask((100, 200), np.array([[-5.0, 10.0, 20.0, 50.0], [190.0, 90.0, 260.0, 130.0]]))
    assert m[10:50, 0:20].all() and not m[70, 100]
    assert m[95, 199] and m.shape == (100, 200)


# ----------------------------------------------------------------------------
# 2. Curved candidates are not lines
# ----------------------------------------------------------------------------


def test_bend_test_keeps_straight_lines_and_drops_arc_chords():
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, (100, 600), (1100, 560), 1, 3)  # straight
    cv2.ellipse(mask, (640, 300), (420, 110), 0, 0, 360, 1, 3)  # a projected centre circle
    plain = cal.extract_lines(mask)
    bent = cal.extract_lines(mask, max_bend_px=cal.DEFAULT_MAX_BEND_PX,
                             min_bend_radius_px=cal.DEFAULT_MIN_BEND_RADIUS * W)
    straight = [ln for ln in bent if ln.length > 900]
    assert len(straight) == 1 and straight[0].bend_px < 0.3
    on_ellipse = [ln for ln in plain if ln.length < 900]
    assert len(on_ellipse) >= 3  # without the test the circle comes back as chords
    assert len([ln for ln in bent if ln.length < 900]) < len(on_ellipse) / 2
    assert max(ln.bend_px for ln in on_ellipse) > 1.0


# ----------------------------------------------------------------------------
# 3. The centre circle
# ----------------------------------------------------------------------------


def test_detect_ellipses_puts_the_circle_on_the_projected_circle():
    cam = _centre_camera()
    frame = _grass(seed=3)
    _draw(frame, cam)
    m = cal.marking_mask(frame)
    lines = cal.extract_lines(m, max_bend_px=cal.DEFAULT_MAX_BEND_PX, min_bend_radius_px=cal.DEFAULT_MIN_BEND_RADIUS * W)
    ells = cal.detect_ellipses(m, lines)
    assert len(ells) >= 1
    hp = cam.homography_px2m()
    true = hp.T @ cal.circle_conic((0.0, 0.0), DEFAULT_PITCH.circle_radius) @ hp
    best = min(float(np.median(cal.conic_distance(true, e.points))) for e in ells)
    assert best < 0.5
    assert max(e.span_deg for e in ells) >= 300


def test_circle_hypotheses_contain_the_true_homography():
    cam = _centre_camera()
    h_m2px = cam.homography_m2px()
    hp = np.linalg.inv(h_m2px)

    def image_line(name):
        ml = DEFAULT_PITCH.line(name)
        a, b = ml.segments[0]
        px, _ = cam.project(np.array([a, b], float))
        return cal.ImageLine(p0=px[0], p1=px[1], abc=cal.project_model_lines(hp, ml.homogeneous[None])[0], n_pixels=100)

    conic = hp.T @ cal.circle_conic((0.0, 0.0), DEFAULT_PITCH.circle_radius) @ hp
    conic /= np.linalg.norm(conic)
    ell = cal.ImageEllipse(conic=conic, centre=np.zeros(2), axes=np.ones(2), angle_deg=0.0,
                           points=np.zeros((0, 2)), span_deg=360.0, rms_px=0.0, n_pixels=1000)
    lines = [image_line("halfway"), image_line("touchline_far")]
    src, dst = cal._circle_hypotheses(lines, [0, 1], [ell], DEFAULT_PITCH)
    assert len(src) >= 2
    t_m = np.diag([1 / 52.5, 1 / 34.0, 1.0])
    t_px = np.array([[1 / W, 0, -0.5], [0, 1 / H, -0.5], [0, 0, 1.0]])
    hn = cal._dlt_points(src @ t_m.T, dst @ t_px.T)
    hs = np.einsum("ij,bjk,kl->bil", np.linalg.inv(t_px), hn, t_m)
    shifts = [cal.homography_shift_px(h_m2px, h / h[2, 2], W, H) for h in hs if np.isfinite(h).all()]
    assert min(shifts) < 1e-3


def test_halfway_line_one_touchline_and_the_circle_are_enough():
    """The typical midfield view. Two lines alone cannot fix a homography; with the circle they do."""
    cam = _centre_camera()
    frame = _grass(seed=4)
    _draw(frame, cam)
    res = cal.calibrate_frame(frame)
    assert res.status == cal.STATUS_CALIBRATED, res.note
    assert {a.model_name for a in res.arc_matches} == {"centre_circle"}
    assert {m.model_name for m in res.matches} <= {"halfway", "touchline_far"}
    assert _keypoint_error(cam, res) < 0.3
    without = cal.calibrate_frame(frame, use_arcs=False)
    assert without.status == cal.STATUS_UNCALIBRATED


# ----------------------------------------------------------------------------
# 4. Refuse instead of guess
# ----------------------------------------------------------------------------


def test_camera_from_homography_recovers_the_camera():
    cam = wide_camera(W, H)
    f, incons, centre = cal.camera_from_homography(cam.homography_m2px(), W, H)
    assert f == pytest.approx(cam.focal_px, rel=1e-6)
    assert incons < 1e-6
    assert np.allclose(centre, cam.position, atol=1e-6)  # x, y and the height above the pitch
    # a homography no pinhole camera produces: stretch one pitch axis. (Seen from straight down the
    # halfway line one of the two conditions on the focal length is void and a stretch reads as a
    # different focal length, so the camera is panned.)
    panned = BroadcastCamera(width=W, height=H, hfov_deg=50.0, position=(0.0, 64.0, 18.0), look_at=(-25.0, 0.0))
    assert cal.camera_from_homography(panned.homography_m2px(), W, H)[1] < 1e-6
    _f, incons2, _c = cal.camera_from_homography(panned.homography_m2px() @ np.diag([1.3, 1.0, 1.0]), W, H)
    assert incons2 > 0.1


def test_the_search_camera_prior_agrees_with_the_decomposition():
    """The batched test the search applies keeps real main cameras under any scale or sign of H and drops
    mirrored ones and cameras behind the goal or below the pitch."""
    cams = [wide_camera(W, H), _centre_camera(),
            BroadcastCamera(width=W, height=H, hfov_deg=30.0, position=(5.0, 60.0, 12.0), look_at=(-20.0, 3.0))]
    hs = [c.homography_m2px() * k for c in cams for k in (1.0, -2.5)]
    cfg = cal.DEFAULT_VERIFY
    assert cal._camera_ok_batch(np.array(hs), W, H, cfg.focal_range, cfg.camera_box).all()
    bad = [cams[0].homography_m2px() @ np.diag([-1.0, 1.0, 1.0]),  # mirrored
           BroadcastCamera(width=W, height=H, hfov_deg=40.0, position=(80.0, 0.0, 15.0),
                           look_at=(30.0, 0.0)).homography_m2px(),  # behind a goal
           cams[1].homography_m2px() @ np.diag([1.0, 3.0, 1.0])]  # no camera
    assert not cal._camera_ok_batch(np.array(bad), W, H, cfg.focal_range, cfg.camera_box).any()


def test_camera_refinement_keeps_a_camera_homography():
    cam = _centre_camera()
    q = cal._camera_params(cam.homography_m2px(), W, H)
    h = cal._camera_h(q, W, H)
    assert np.allclose(h / h[2, 2], cam.homography_m2px(), atol=1e-9)


def test_verification_accepts_the_truth_and_refuses_a_shifted_pitch():
    cam = wide_camera(W, H)
    frame = _grass(seed=5)
    _draw(frame, cam)
    res = cal.calibrate_frame(frame)
    assert res.ok and not cal.verify_failures(res.verify, cal.DEFAULT_VERIFY, W)
    assert res.verify["support"] > 0.9 and res.verify["sigma_grid_m"] < 0.25  # the 0.5 px noise floor, far side
    grass = cal.grass_region(frame)
    m = cal.marking_mask(frame, grass=grass)
    shifted = res.h_m2px @ np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 5 m along the pitch
    v = cal.verify_solution(shifted, m, grass, res.lines, [], res.ellipses, [], with_loo=False)
    assert v["support"] < 0.6
    assert v["missing"] >= 1


def test_fit_uncertainty_grows_away_from_the_markings():
    """Lines on the left of the frame only: the right of the pitch is an extrapolation."""
    cam = BroadcastCamera(width=W, height=H, hfov_deg=60.0, position=(0.0, 64.0, 18.0), look_at=(-30.0, 0.0))
    h_m2px = cam.homography_m2px()
    hp = np.linalg.inv(h_m2px)
    names = ["goal_line_left", "penalty_front_left", "penalty_side_far", "penalty_side_near"]
    lines, matches = [], []
    for i, name in enumerate(names):
        ml = DEFAULT_PITCH.line(name)
        a, b = ml.segments[0]
        px, _ = cam.project(np.array([a, b], float))
        lines.append(cal.ImageLine(p0=px[0], p1=px[1], abc=cal.project_model_lines(hp, ml.homogeneous[None])[0],
                                   n_pixels=100))
        matches.append(cal.LineMatch(image_index=i, model_name=name, distance_px=0.0, extent_fraction=1.0))
    fn = cal._fit_residuals(lines, matches, DEFAULT_PITCH)
    near, _ = cam.project(np.array([[-40.0, 0.0]]))
    far, _ = cam.project(np.array([[10.0, 0.0]]))
    s = cal.fit_uncertainty_m(h_m2px, fn, np.concatenate([near, far]), W, H)
    assert np.isfinite(s).all() and s[1] > 3 * s[0]


def test_a_refused_solution_is_uncalibrated_with_the_reasons():
    cam = wide_camera(W, H)
    frame = _grass(seed=6)
    _draw(frame, cam)
    strict = cal.VerifyConfig(max_sigma_grid_m=0.0)
    res = cal.calibrate_frame(frame, verify=strict)
    assert res.status == cal.STATUS_UNCALIBRATED and not res.ok
    assert res.note.startswith("refused") and "standard error over the pitch" in res.note
    assert res.verify["support"] > 0.9  # the evidence is kept
    assert np.isnan(res.project(np.array([[640.0, 500.0]]))).all()
    recorded = cal.calibrate_frame(frame, verify=cal.VerifyConfig(max_sigma_grid_m=0.0, refuse=False))
    assert recorded.ok and "sigma_grid_m" in recorded.verify


def test_no_real_camera_is_refused():
    """A homography that projects the pitch with no positive focal length is never reported."""
    cam = wide_camera(W, H)
    frame = _grass(seed=7)
    _draw(frame, cam)
    grass = cal.grass_region(frame)
    m = cal.marking_mask(frame, grass=grass)
    warped = cam.homography_m2px() @ np.diag([1.0, -0.4, 1.0])
    v = cal.verify_solution(warped, m, grass, [], [], [], [], with_loo=False)
    assert any("focal" in w or "camera" in w for w in cal.verify_failures(v, cal.DEFAULT_VERIFY, W))


def test_frame_calibrator_takes_person_boxes():
    cam = _centre_camera()
    frame = _grass(seed=8)
    _draw(frame, cam)
    frame[300:420, 700:730] = 255  # a white shirt next to the circle
    fc = cal.FrameCalibrator(frame_stride=1)
    res = fc.update(frame, person_boxes=np.array([[695.0, 295.0, 735.0, 425.0]]))
    assert res.ok, res.note
    assert _keypoint_error(cam, res) < 0.3
