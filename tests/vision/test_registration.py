"""Frame-to-frame camera motion and the bridging of uncalibrated frames (no data, no GPU)."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from varlab_vision import calibration as cal  # noqa: E402
from varlab_vision.registration import FrameRegistrar, Motion  # noqa: E402
from varlab_vision.synthetic import BroadcastCamera, render_textured, textured_world  # noqa: E402

W, H = 640, 360


@pytest.fixture(scope="module")
def world():
    return textured_world(seed=3)


def _cam(look_x: float, look_y: float = 0.0, hfov: float = 40.0) -> BroadcastCamera:
    return BroadcastCamera(width=W, height=H, hfov_deg=hfov, position=(0.0, 60.0, 22.0), look_at=(look_x, look_y))


def _zoom_cams(n: int = 50):
    """Wide on the penalty area, zoom over 15 frames into open grass with no markings, hold, zoom back out."""
    cams = []
    for i in range(n):
        u = 0.0 if i < 10 else (i - 10) / 15 if i < 25 else 1.0 if i < 35 else max(0.0, 1 - (i - 35) / 15)
        cams.append(_cam(-30 + 10 * u, -12 * u, 40 - 28 * u))
    return cams


def _motion_error_px(m: Motion, prev: BroadcastCamera, cur: BroadcastCamera) -> float:
    return cal.homography_shift_px(m.h @ prev.homography_m2px(), cur.homography_m2px(), W, H)


def _max_error_m(cam: BroadcastCamera, res: cal.CalibrationResult) -> float:
    xs, ys = np.meshgrid(np.linspace(-52, 52, 27), np.linspace(-34, 34, 18))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    px, depth = cam.project(pts)
    vis = (depth > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    return float(np.linalg.norm(res.project(px[vis]) - pts[vis], axis=1).max())


def test_registration_follows_a_pan_and_a_zoom(world):
    cams = _zoom_cams(30)
    reg = FrameRegistrar()
    assert reg.push(render_textured(cams[0], world)) is None
    errs = []
    for prev, cur in zip(cams[:-1], cams[1:]):
        m = reg.push(render_textured(cur, world))
        assert m.ok
        errs.append(_motion_error_px(m, prev, cur))
    assert max(errs) < 0.6


def test_a_static_overlay_and_a_moving_person_do_not_pull_the_motion(world):
    cams = [_cam(-30 + 0.8 * i) for i in range(6)]
    reg = FrameRegistrar()
    for i, cam in enumerate(cams):
        img = render_textured(cam, world)
        img[10:60, 10:250] = (30, 30, 30)  # a scoreboard that stays put on screen
        cv2.putText(img, "HOME 1 - 0 AWAY  63:12", (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        box = np.array([[300 - 6 * i, 200, 330 - 6 * i, 270]])  # a player running against the pan
        img[200:270, 300 - 6 * i:330 - 6 * i] = (20, 20, 200)
        m = reg.push(img, box)
        if i:
            assert m.ok and _motion_error_px(m, cams[i - 1], cam) < 0.6


def test_a_cut_fails_the_motion(world):
    reg = FrameRegistrar()
    reg.push(render_textured(_cam(-30), world))
    other = textured_world(seed=11)
    assert not reg.push(render_textured(_cam(20, 5, 30), other)).ok


def test_motion_is_none_after_a_reset_or_a_new_frame_size(world):
    reg = FrameRegistrar()
    reg.push(render_textured(_cam(-30), world))
    reg.reset()
    assert reg.push(render_textured(_cam(-30), world)) is None
    assert reg.push(np.zeros((180, 320, 3), np.uint8)) is None


# ----------------------------------------------------------------------------
# bridge_gaps on results built from the true cameras (exact motions)
# ----------------------------------------------------------------------------


def _results(cams, calibrated, cut_at=(), wrong=None):
    """One result per camera: the true homography where ``calibrated``, else uncalibrated; exact motions.
    ``wrong`` maps a frame to a camera whose homography is reported there instead (a false lock)."""
    out = []
    for i, cam in enumerate(cams):
        m = None if i == 0 else Motion(h=cam.homography_m2px() @ np.linalg.inv(cams[i - 1].homography_m2px()),
                                       n_inliers=500, spread=0.8, residual_px=0.05, ok=True)
        if i in calibrated:
            h = (wrong or {}).get(i, cam).homography_m2px()
            r = cal.CalibrationResult(status=cal.STATUS_CALIBRATED, h_m2px=h, h_px2m=np.linalg.inv(h))
        else:
            r = cal.CalibrationResult(status=cal.STATUS_UNCALIBRATED, note="too few lines")
        r.motion, r.shot_cut = m, i in cut_at
        out.append(r)
    return out


def _pan(n):
    return [_cam(-30 + 0.6 * i) for i in range(n)]


def test_a_gap_between_two_fits_is_bridged_exactly():
    cams = _pan(30)
    out = cal.bridge_gaps(_results(cams, set(range(5)) | set(range(25, 30))), W, H)
    assert [r.status for r in out[5:25]] == [cal.STATUS_BRIDGED] * 20
    assert max(_max_error_m(c, r) for c, r in zip(cams[5:25], out[5:25])) < 1e-6
    assert "bridged between frames 5 and 26" in out[10].note and "too few lines" in out[10].note


def test_chains_that_disagree_bridge_nothing():
    cams = _pan(30)
    res = _results(cams, set(range(5)) | set(range(25, 30)), wrong={25: _cam(-12, 3.0)})  # frame 25 locked 3 m off
    out = cal.bridge_gaps(res, W, H)
    assert all(r.status == cal.STATUS_UNCALIBRATED for r in out[5:25])
    assert cal.bridge_gaps(res, W, H, agree_m=10.0)[10].status == cal.STATUS_BRIDGED


def test_one_sided_gaps_are_capped_and_cuts_break_the_chain():
    cams = _pan(80)
    out = cal.bridge_gaps(_results(cams, set(range(5))), W, H, one_sided_max=25)  # the clip ends uncalibrated
    assert [r.status for r in out[5:30]] == [cal.STATUS_BRIDGED] * 25
    assert all(r.status == cal.STATUS_UNCALIBRATED for r in out[30:])
    assert "unchecked" in out[6].note

    res = _results(cams, set(range(5)) | set(range(60, 80)), cut_at={20})
    out = cal.bridge_gaps(res, W, H, one_sided_max=25)
    assert all(r.status == cal.STATUS_BRIDGED for r in out[5:20])  # forward from frame 4 up to the cut
    assert all(r.status == cal.STATUS_UNCALIBRATED for r in out[20:35])  # past the cut, out of reach of frame 60
    assert all(r.status == cal.STATUS_BRIDGED for r in out[35:60])  # back from frame 60, 25 frames
    assert max(_max_error_m(c, r) for c, r in zip(cams, out) if r.status == cal.STATUS_BRIDGED) < 1e-6


def test_gaps_longer_than_max_gap_and_failed_motions_are_left_alone():
    cams = _pan(40)
    res = _results(cams, {0, 39})
    assert all(r.status == cal.STATUS_UNCALIBRATED for r in cal.bridge_gaps(res, W, H, max_gap=30, one_sided_max=0)[1:39])
    for r in res[1:]:  # every motion into and out of the gap failed
        r.motion = Motion(h=r.motion.h, n_inliers=10, spread=0.1, residual_px=0.05, ok=False)
    assert all(r.status == cal.STATUS_UNCALIBRATED for r in cal.bridge_gaps(res, W, H)[1:39])


# ----------------------------------------------------------------------------
# FrameCalibrator end to end: a zoom into open grass, bridged from both sides
# ----------------------------------------------------------------------------


def test_frame_calibrator_measures_motion_and_bridges_a_zoom_with_no_markings(world):
    cams = _zoom_cams(50)
    fc = cal.FrameCalibrator()
    res = []
    for cam in cams:
        res += fc.push(render_textured(cam, world))
    res += fc.flush()
    assert len(res) == 50 and res[0].motion is None and all(r.motion is not None and r.motion.ok for r in res[1:])
    unsolved = [i for i, r in enumerate(res) if not r.ok]
    assert len(unsolved) >= 10  # the zoomed-in frames have no markings to fit
    out = fc.bridge(res)
    assert all(out[i].status == cal.STATUS_BRIDGED for i in unsolved)
    assert max(_max_error_m(cams[i], out[i]) for i in unsolved) < 0.3
    assert all(o is r for o, r in zip(out, res) if r.ok)  # fitted frames are returned as they were


def test_frame_calibrator_without_registration_attaches_no_motion(world):
    fc = cal.FrameCalibrator(register=False)
    res = [fc.update(render_textured(_cam(-30 + 0.5 * i), world)) for i in range(3)]
    assert all(r.motion is None for r in res)
    assert fc.bridge(res) == res
