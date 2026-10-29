"""The pipeline with a ``FrameCalibrator``: pitch coordinates, camera records, the run report.

CPU only and no SoccerNet data. The clips come from ``synthetic.py``: rectangles standing on a pitch whose
real markings are drawn under a known broadcast camera, so the calibrated pitch positions can be scored
against the clip's ground-truth ``Labels-GameState.json``. Numbers measured here say nothing about real
footage (perfect lines, no crowd, no lens distortion); they guard the wiring and gross regressions.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from varlab_vision import pipeline as pl  # noqa: E402
from varlab_vision.calibration import STATUS_CALIBRATED, CalibrationResult, FrameCalibrator  # noqa: E402
from varlab_vision.gshota import gs_hota  # noqa: E402
from varlab_vision.gsr_format import LABELS_FILE  # noqa: E402
from varlab_vision.loaders import camera_frame, objects_frame, read_gsr_clip  # noqa: E402
from varlab_vision.synthetic import render_synthetic_clip  # noqa: E402

N_WIDE = 8


def _strict_json(path: Path):
    """Parse a JSON file and fail on NaN / Infinity, which Python's json module writes but JSON forbids."""

    def refuse(token: str):
        raise ValueError(f"non-finite constant {token} in {path}")

    return json.loads(Path(path).read_text(), parse_constant=refuse)


def _run(clip, out: Path, **kw) -> SimpleNamespace:
    pipe = pl.smoke_pipeline(tracker=pl.GreedyIoUTracker(), **kw)
    report = pipe.run_clip(clip.clip_dir, out / LABELS_FILE, game_id="syn-game-1")
    return SimpleNamespace(pipe=pipe, report=report, pred=read_gsr_clip(report.out_path))


@pytest.fixture(scope="module")
def wide(tmp_path_factory) -> SimpleNamespace:
    """Eight frames of the wide camera at 1280x720, three players a side, no referee (the pipeline writes
    every person as a player, so a referee would only add a constant false positive to the metric).
    Every frame is fitted (``frame_stride=1``): these tests pin the per-frame contract; the default
    frame stride has its own fixture, ``wide_stride``."""
    root = tmp_path_factory.mktemp("calib_wide")
    clip = render_synthetic_clip(root, n_frames=N_WIDE, width=1280, height=720, n_per_team=3,
                                 with_referee=False, seed=0)
    run = _run(clip, root / "pred", calibrator=FrameCalibrator(frame_stride=1))
    return SimpleNamespace(clip=clip, gt=read_gsr_clip(clip.labels_path), **vars(run))


N_STRIDE = 14


@pytest.fixture(scope="module")
def wide_stride(tmp_path_factory) -> SimpleNamespace:
    """The same scene over 14 frames through the default ``FrameCalibrator``, whose frame stride
    interpolates most frames once tracking is steady."""
    root = tmp_path_factory.mktemp("calib_wide_stride")
    clip = render_synthetic_clip(root, n_frames=N_STRIDE, width=1280, height=720, n_per_team=3,
                                 with_referee=False, seed=0)
    run = _run(clip, root / "pred", calibrator=FrameCalibrator())
    return SimpleNamespace(clip=clip, gt=read_gsr_clip(clip.labels_path), **vars(run))


@pytest.fixture(scope="module")
def bare(tmp_path_factory) -> SimpleNamespace:
    """The same scene without pitch markings: nothing to calibrate from."""
    root = tmp_path_factory.mktemp("calib_bare")
    clip = render_synthetic_clip(root, n_frames=4, width=1280, height=720, n_per_team=3,
                                 with_referee=True, seed=0, with_markings=False)
    run = _run(clip, root / "pred", calibrator=FrameCalibrator())
    return SimpleNamespace(clip=clip, gt=read_gsr_clip(clip.labels_path), **vars(run))


def _foot_errors_m(gt, pred) -> np.ndarray:
    """Pitch distance (m) between each predicted bottom-middle and the ground truth of the object whose
    image box has the same bottom-middle (the colour-blob boxes coincide with the drawn rectangles)."""
    gdf, pdf = objects_frame(gt), objects_frame(pred)
    errs = []
    for f, p in pdf.groupby("frame"):
        g = gdf[gdf["frame"] == f]
        gx, gy = (g["x"] + g["w"] / 2).to_numpy(), (g["y"] + g["h"]).to_numpy()
        for _, r in p.iterrows():
            d = np.hypot(gx - (r["x"] + r["w"] / 2), gy - (r["y"] + r["h"]))
            j = int(d.argmin())
            assert d[j] < 2.0, f"frame {f}: no ground-truth box near the predicted one"
            errs.append(math.hypot(r["x_pitch"] - g["x_pitch"].iloc[j], r["y_pitch"] - g["y_pitch"].iloc[j]))
    return np.array(errs)


# ----------------------------------------------------------------------------
# a. Calibrated run on the wide camera
# ----------------------------------------------------------------------------


def test_every_object_has_pitch_position_and_calibration_extras(wide):
    assert len(wide.pred.objects) == len(wide.gt.objects) == N_WIDE * 6
    for o in wide.pred.objects:
        for key in ("pitch_x", "pitch_y", "calib_error", "calib_status"):
            assert key in o.extra, f"{o.id} lacks {key}"
        assert o.extra["pitch_x"] is not None and o.extra["pitch_y"] is not None
        assert 0.0 < o.extra["calib_error"] < 10.0  # RMS line-fit error in pixels
        assert o.bbox_pitch is not None


def test_first_frame_is_calibrated_and_the_rest_are_tracked(wide):
    by_image = {}
    for o in wide.pred.objects:
        by_image.setdefault(o.image_id, set()).add(o.extra["calib_status"])
    statuses = [by_image[im.image_id] for im in wide.pred.images]
    assert statuses == [{"calibrated"}] + [{"tracked"}] * (N_WIDE - 1)
    cams = camera_frame(wide.pred)
    assert cams["status"].tolist() == ["calibrated"] + ["tracked"] * (N_WIDE - 1)
    assert objects_frame(wide.pred).groupby("frame")["calib_status"].first().tolist() == cams["status"].tolist()


def test_one_camera_record_per_frame_with_distinct_numeric_ids(wide):
    cams = [d for d in wide.pred.other_annotations if d["supercategory"] == "camera"]
    assert len(cams) == len(wide.pred.other_annotations) == N_WIDE
    assert [d["image_id"] for d in cams] == [im.image_id for im in wide.pred.images]
    ids = [d["id"] for d in cams]
    obj_ids = {o.id for o in wide.pred.objects}
    assert len(set(ids)) == N_WIDE and not set(ids) & obj_ids
    assert all(i.isdigit() and i.startswith(d["image_id"]) for i, d in zip(ids, cams))
    assert {d["category_id"] for d in cams} == {c for c, (n, _) in wide.pred.categories.items() if n == "camera"}
    for d in cams:
        h = np.asarray(d["calibration"]["homography_px2m"], float)
        assert h.shape == (3, 3) and np.isfinite(h).all()
        assert d["calibration"]["n_inliers"] >= 4 and d["calibration"]["shot_cut"] is False
    assert all(im.has_labeled_camera for im in wide.pred.images)


def test_output_round_trips_through_the_loaders(wide):
    raw = _strict_json(wide.report.out_path)  # no NaN / Infinity anywhere in the file
    assert len(raw["annotations"]) == len(wide.pred.objects) + N_WIDE
    again = read_gsr_clip(wide.report.out_path)
    assert again.to_json() == wide.pred.to_json()
    df = objects_frame(wide.pred)
    assert list(df.columns[-2:]) == ["calib_error", "calib_status"]
    assert len(df) == N_WIDE * 6 and df["x_pitch"].notna().all() and df["calib_error"].notna().all()
    assert df["calib_status"].isin(["calibrated", "tracked"]).all()
    cams = camera_frame(wide.pred)
    assert cams["frame"].tolist() == list(range(1, N_WIDE + 1))
    assert (cams["n_inliers"] >= 4).all() and cams["fit_error_px"].notna().all()
    assert cams["reprojection_error_m"].notna().all() and not cams["shot_cut"].any()
    # per-object calib_error is the frame's fit error
    per_frame = df.groupby("frame")["calib_error"].first().to_numpy()
    assert np.allclose(per_frame, cams["fit_error_px"].to_numpy(), atol=1e-3)


def test_pitch_extras_equal_the_bbox_pitch_bottom_middle(wide):
    for o in wide.pred.objects:
        x, y = o.bbox_pitch.xy
        assert abs(o.extra["pitch_x"] - x) < 1e-3 and abs(o.extra["pitch_y"] - y) < 1e-3
        # and the middle really is the middle of the two projected bottom corners
        assert x == pytest.approx((o.bbox_pitch.x_bottom_left + o.bbox_pitch.x_bottom_right) / 2)
        assert y == pytest.approx((o.bbox_pitch.y_bottom_left + o.bbox_pitch.y_bottom_right) / 2)


def test_written_homographies_place_the_true_feet_within_centimetres(wide):
    """Calibration on its own: the ground-truth boxes' bottom-middles mapped through each frame's
    homography as written in the camera record. Measured on this clip (seed 0, 8 frames, 1280x720,
    wide camera): mean 0.007 m, max 0.013 m."""
    h_by_image = {c["image_id"]: np.array(c["calibration"]["homography_px2m"], float)
                  for c in wide.pred.other_annotations if c.get("supercategory") == "camera"}
    image_of_frame = {im.frame_index: im.image_id for im in wide.pred.images}
    gdf = objects_frame(wide.gt)
    errs = []
    for f, g in gdf.groupby("frame"):
        h = h_by_image[image_of_frame[f]]
        pts = np.stack([g["x"] + g["w"] / 2, g["y"] + g["h"], np.ones(len(g))], axis=1) @ h.T
        xy = pts[:, :2] / pts[:, 2:3]
        errs.extend(np.hypot(*(xy - g[["x_pitch", "y_pitch"]].to_numpy()).T).tolist())
    errs = np.array(errs)
    assert len(errs) == N_WIDE * 6
    assert errs.max() < 0.05


def test_calibrated_positions_match_the_ground_truth(wide):
    """End to end, so this includes the colour-blob detector: its box bottoms land 0 or 1 px below the
    drawn rectangles (anti-aliasing and JPEG), and one pixel at the far touchline of a 720p wide shot spans
    about half a metre. Measured on this clip: mean 0.23 m, max 0.65 m; seeds 0-3 give max 0.43-0.65 m.
    The calibration's own share is checked above (under 2 cm)."""
    err = _foot_errors_m(wide.gt, wide.pred)
    assert len(err) == N_WIDE * 6
    assert err.mean() < 0.4 and err.max() < 1.0


def test_gs_hota_against_the_ground_truth_is_high(wide):
    """Measured on this clip (players only, greedy tracker, no referee, 8 frames): GS-HOTA, DetA and
    AssA 1.0000, LocA 0.9862; seeds 0-3 all give GS-HOTA 1.0 and LocA 0.986-0.996."""
    gdf, pdf = objects_frame(wide.gt), objects_frame(wide.pred)
    r = gs_hota(gdf, pdf, roles=("player",))
    assert r.n_gt == r.n_pred == N_WIDE * 6
    assert r.hota > 0.97 and r.deta > 0.97 and r.assa > 0.97
    assert r.loca > 0.97  # the 5 m kernel is far wider than the ~0.2 m mean foot error


# ----------------------------------------------------------------------------
# a'. The default frame stride
# ----------------------------------------------------------------------------


def test_frame_stride_writes_one_calibration_per_frame_and_says_which_are_interpolated(wide_stride):
    run = wide_stride
    cams = camera_frame(run.pred)
    statuses = cams["status"].tolist()
    assert len(statuses) == N_STRIDE and cams["frame"].tolist() == list(range(1, N_STRIDE + 1))
    assert statuses[0] == "calibrated" and set(statuses[1:]) == {"tracked", "interpolated"}
    interp = cams["status"] == "interpolated"
    # measured: c t t t i t i i t i i i t t (two steady fits, then stretches of 1, 2, 3 held frames;
    # the clip ends one frame into the next stretch, so flush fits that frame)
    assert interp.sum() >= 5
    assert (cams.loc[interp, "n_inliers"] == 0).all() and cams.loc[interp, "fit_error_px"].isna().all()
    assert (cams.loc[~interp, "n_inliers"] >= 4).all() and cams.loc[~interp, "fit_error_px"].notna().all()
    assert all(im.has_labeled_camera for im in run.pred.images)
    for d in run.pred.other_annotations:
        assert np.isfinite(np.asarray(d["calibration"]["homography_px2m"], float)).all()
    df = objects_frame(run.pred)
    assert len(df) == N_STRIDE * 6 and df["x_pitch"].notna().all()
    assert df.groupby("frame")["calib_status"].first().tolist() == statuses
    assert df.loc[df["calib_status"] == "interpolated", "calib_error"].isna().all()
    _strict_json(run.report.out_path)
    cal = run.report.as_dict()["calibration"]
    assert (cal["frames"], cal["calibrated"], cal["tracked"], cal["interpolated"], cal["uncalibrated"]) == (
        N_STRIDE, 1, int((~interp).sum()) - 1, int(interp.sum()), 0)
    # errors and inliers summarise the fitted frames only
    assert cal["inliers_mean"] == pytest.approx(cams.loc[~interp, "n_inliers"].mean(), abs=1e-2)
    assert cal["fit_error_px_mean"] == pytest.approx(cams.loc[~interp, "fit_error_px"].mean(), abs=2e-3)
    assert run.report.stages["calib"].items == N_STRIDE and run.report.notes == []


def test_frame_stride_homographies_place_the_true_feet_within_centimetres(wide_stride):
    """As for the per-frame run (max 0.013 m there), interpolated frames included."""
    run = wide_stride
    h_by_image = {c["image_id"]: np.array(c["calibration"]["homography_px2m"], float)
                  for c in run.pred.other_annotations}
    image_of_frame = {im.frame_index: im.image_id for im in run.pred.images}
    errs = []
    for f, g in objects_frame(run.gt).groupby("frame"):
        pts = np.stack([g["x"] + g["w"] / 2, g["y"] + g["h"], np.ones(len(g))], axis=1) @ h_by_image[image_of_frame[f]].T
        errs.extend(np.hypot(*(pts[:, :2] / pts[:, 2:3] - g[["x_pitch", "y_pitch"]].to_numpy()).T).tolist())
    assert len(errs) == N_STRIDE * 6 and max(errs) < 0.05


# ----------------------------------------------------------------------------
# b. Nothing to calibrate from
# ----------------------------------------------------------------------------


def test_uncalibrated_frames_write_null_pitch_data_and_say_so(bare):
    n = bare.clip.n_frames
    assert len(bare.pred.objects) == len(bare.gt.objects) == n * 7
    for o in bare.pred.objects:
        assert o.bbox_pitch is None
        assert o.extra["pitch_x"] is None and o.extra["pitch_y"] is None
        assert o.extra["calib_error"] is None and o.extra["calib_status"] == "uncalibrated"
    assert not any(im.has_labeled_camera for im in bare.pred.images)
    cams = camera_frame(bare.pred)
    assert cams["status"].tolist() == ["uncalibrated"] * n and (cams["n_inliers"] == 0).all()
    assert cams["fit_error_px"].isna().all() and cams["reprojection_error_m"].isna().all()
    assert all(d["calibration"]["homography_px2m"] is None for d in bare.pred.other_annotations)
    _strict_json(bare.report.out_path)
    df = objects_frame(bare.pred)
    assert df["x_pitch"].isna().all() and df["calib_error"].isna().all()

    assert any(f"all {n} frames uncalibrated" in note for note in bare.report.notes), bare.report.notes
    cal = bare.report.as_dict()["calibration"]
    assert (cal["frames"], cal["calibrated"], cal["tracked"], cal["uncalibrated"]) == (n, 0, 0, n)
    assert cal["fit_error_px_mean"] is None and cal["reprojection_error_m_mean"] is None and cal["inliers_mean"] is None


def test_gs_hota_treats_missing_positions_as_no_match(bare):
    gdf, pdf = objects_frame(bare.gt), objects_frame(bare.pred)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning from NaN arithmetic
        r = gs_hota(gdf, pdf, roles=("player",))
        full = gs_hota(gdf, pdf)
    assert (r.n_gt, r.n_pred) == (bare.clip.n_frames * 6, bare.clip.n_frames * 7)  # the pipeline calls the referee a player
    assert r.deta == 0.0 and r.hota == 0.0 and r.loca == 0.0
    assert full.deta == 0.0 and full.hota == 0.0


def test_gs_hota_ignores_only_the_rows_without_a_position():
    frames = np.repeat([1, 2, 3], 2)
    gt = pd.DataFrame({"frame": frames, "track_id": [1, 2] * 3, "role": "player", "team": "left", "jersey": None,
                       "x_pitch": [0.0, 10.0] * 3, "y_pitch": [0.0, 5.0] * 3})
    pred = gt.copy()
    pred.loc[pred["track_id"] == 2, ["x_pitch", "y_pitch"]] = np.nan  # one of the two players never localised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = gs_hota(gt, pred)
    # track 1 matches in every frame, track 2 never: TP = 3, FN = FP = 3 at every alpha
    assert r.deta == pytest.approx(3 / 9)
    assert r.per_alpha["TP"].eq(3).all() and r.per_alpha["FN"].eq(3).all() and r.per_alpha["FP"].eq(3).all()


def test_a_static_projector_is_not_a_fallback_for_an_uncalibrated_frame(tmp_path: Path):
    """With a calibrator the static projector is never consulted: a clip must not mix two coordinate sources."""
    clip = render_synthetic_clip(tmp_path, n_frames=3, width=640, height=360, n_per_team=2,
                                 with_referee=False, seed=2, with_markings=False)
    run = _run(clip, tmp_path / "pred", calibrator=FrameCalibrator(), projector=pl.HomographyProjector(clip.homography))
    assert run.pipe.backends()["projector"] == "line-calibration"
    assert all(o.bbox_pitch is None and o.extra["pitch_x"] is None for o in run.pred.objects)
    # the same clip without a calibrator does use the projector (existing behaviour), and writes no calibration keys
    plain = _run(clip, tmp_path / "plain", projector=pl.HomographyProjector(clip.homography))
    assert plain.pipe.backends()["projector"] == "homography"
    assert all(o.bbox_pitch is not None and "pitch_x" not in o.extra and "calib_status" not in o.extra
               for o in plain.pred.objects)
    assert plain.pred.other_annotations == [] and plain.report.calibration == {}


# ----------------------------------------------------------------------------
# c. Report and backends
# ----------------------------------------------------------------------------


def test_run_report_calibration_summary_and_backends(wide):
    d = wide.report.as_dict()
    cal = d["calibration"]
    assert set(cal) >= {"frames", "calibrated", "tracked", "uncalibrated", "shot_cuts", "fit_error_px_mean",
                        "fit_error_px_max", "reprojection_error_m_mean", "inliers_mean"}
    assert (cal["frames"], cal["calibrated"], cal["tracked"], cal["uncalibrated"], cal["shot_cuts"]) == (N_WIDE, 1, N_WIDE - 1, 0, 0)
    cams = camera_frame(wide.pred)
    assert cal["fit_error_px_mean"] == pytest.approx(cams["fit_error_px"].mean(), abs=2e-3)
    assert cal["fit_error_px_max"] == pytest.approx(cams["fit_error_px"].max(), abs=2e-3)
    assert cal["reprojection_error_m_mean"] == pytest.approx(cams["reprojection_error_m"].mean(), abs=2e-3)
    assert cal["inliers_mean"] == pytest.approx(cams["n_inliers"].mean(), abs=1e-2)
    assert 0 < cal["fit_error_px_mean"] <= cal["fit_error_px_max"] < 10
    json.dumps(d, allow_nan=False)  # the whole report serialises (colab/vision_worker writes it as report.json)
    assert d["notes"] == []
    assert wide.pipe.backends()["projector"] == d["backends"]["projector"] == "line-calibration"
    assert d["stages"]["calib"]["items"] == N_WIDE and d["stages"]["calib"]["seconds"] > 0
    assert "calib" in wide.report.summary()
    assert wide.pred.info.extra["backends"]["projector"] == "line-calibration"


def test_report_without_a_calibrator_has_no_calibration_block(tmp_path: Path):
    clip = render_synthetic_clip(tmp_path, n_frames=3, width=640, height=360, n_per_team=2, seed=3)
    run = _run(clip, tmp_path / "pred")
    assert run.report.as_dict()["calibration"] == {} and run.pipe.backends()["projector"] == "none"
    assert run.report.stages["calib"].items == 0


# ----------------------------------------------------------------------------
# Edge cases of the GSR writer
# ----------------------------------------------------------------------------


def _calibrated(h_px2m: np.ndarray) -> CalibrationResult:
    return CalibrationResult(status=STATUS_CALIBRATED, h_m2px=np.linalg.inv(h_px2m), h_px2m=h_px2m,
                             fit_error_px=1.0, reprojection_error_m=0.1)


def test_a_box_that_projects_to_infinity_gets_null_pitch_data_everywhere():
    """A bottom corner on the horizon (w = 0) is a calibrated frame but an unusable box: bbox_pitch,
    pitch_x and pitch_y are null together and the JSON stays valid."""
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.01, 0.0, -1.0]])  # w = 0.01 x - 1, zero at x = 100
    dets = pl.FrameDetections(xyxy=np.array([[10.0, 20.0, 50.0, 60.0], [60.0, 20.0, 100.0, 60.0]]),
                              confidence=np.ones(2), class_name=np.array(["person", "person"], dtype=object),
                              tracker_id=np.array([1, 2]))
    pipe = pl.VisionPipeline(detector=pl.ColorBlobDetector(), tracker=pl.GreedyIoUTracker(), calibrator=FrameCalibrator())
    clip = pipe._to_gsr("CAL-7", "g", [(1, 160, 90, dets)], {}, [_calibrated(h)])
    json.dumps(clip.to_json(), allow_nan=False)
    ok, horizon = clip.objects
    assert ok.bbox_pitch is not None and ok.extra["pitch_x"] is not None
    assert horizon.bbox_pitch is None and horizon.extra["pitch_x"] is None and horizon.extra["pitch_y"] is None
    assert horizon.extra["calib_status"] == "calibrated"  # the frame is still calibrated
    assert clip.images[0].has_labeled_camera


class _FlakyCalibrator:
    """Calibrates every frame with a fixed homography except frame 2, where it raises."""

    name = "flaky"

    def __init__(self, h_px2m: np.ndarray):
        self.h, self.calls, self.resets = h_px2m, 0, 0

    def reset(self) -> None:
        self.resets += 1

    def update(self, frame_bgr: np.ndarray) -> CalibrationResult:
        self.calls += 1
        if self.calls == 2:
            raise np.linalg.LinAlgError("singular matrix")
        return _calibrated(self.h)


def test_a_calibrator_failure_on_one_frame_does_not_abort_the_clip(tmp_path: Path):
    clip = render_synthetic_clip(tmp_path, n_frames=3, width=640, height=360, n_per_team=2,
                                 with_referee=False, seed=4, with_markings=False)
    cal = _FlakyCalibrator(clip.homography)
    pipe = pl.smoke_pipeline(tracker=pl.GreedyIoUTracker(), calibrator=cal)
    report = pipe.run_clip(clip.clip_dir, tmp_path / "pred" / LABELS_FILE, game_id="g")
    pred = read_gsr_clip(report.out_path)
    cams = camera_frame(pred)
    assert cams["status"].tolist() == ["calibrated", "uncalibrated", "calibrated"]
    notes = [d["calibration"]["note"] for d in pred.other_annotations]
    assert "LinAlgError: singular matrix" in notes[1]
    assert cal.resets == 2  # once at the start of the run, once after the failure
    assert report.calibration["uncalibrated"] == 1 and any("1 of 3 frames uncalibrated" in n for n in report.notes)
    by_frame = objects_frame(pred).groupby("frame")["x_pitch"].apply(lambda c: c.notna().all()).tolist()
    assert by_frame == [True, False, True]
    assert pipe.backends()["projector"] == "flaky"


class _HoldingFlakyCalibrator:
    """A ``push`` calibrator that holds frames 2 and 3 back and raises on frame 4."""

    name = "holding-flaky"

    def __init__(self, h_px2m: np.ndarray):
        self.h, self.calls, self.resets, self.held = h_px2m, 0, 0, 0

    @property
    def n_pending(self) -> int:
        return self.held

    def reset(self) -> None:
        self.resets += 1
        self.held = 0

    def push(self, frame_bgr: np.ndarray):
        self.calls += 1
        if self.calls in (2, 3):
            self.held += 1
            return []
        if self.calls == 4:
            raise np.linalg.LinAlgError("singular matrix")
        return [_calibrated(self.h)]

    def flush(self):
        out, self.held = [_calibrated(self.h)] * self.held, 0
        return out


def test_a_calibrator_failure_while_frames_are_held_marks_them_all_uncalibrated(tmp_path: Path):
    clip = render_synthetic_clip(tmp_path, n_frames=6, width=640, height=360, n_per_team=2,
                                 with_referee=False, seed=4, with_markings=False)
    cal = _HoldingFlakyCalibrator(clip.homography)
    pipe = pl.smoke_pipeline(tracker=pl.GreedyIoUTracker(), calibrator=cal)
    report = pipe.run_clip(clip.clip_dir, tmp_path / "pred" / LABELS_FILE, game_id="g")
    cams = camera_frame(read_gsr_clip(report.out_path))
    assert cams["status"].tolist() == ["calibrated"] + ["uncalibrated"] * 3 + ["calibrated"] * 2
    assert cal.resets == 2 and report.calibration["uncalibrated"] == 3


class _BoxRecordingCalibrator:
    """A ``push`` calibrator that takes person boxes and records what it was given."""

    name = "box-recording"

    def __init__(self, h_px2m: np.ndarray):
        self.h, self.boxes = h_px2m, []

    n_pending = 0

    def reset(self) -> None:
        pass

    def push(self, frame_bgr: np.ndarray, person_boxes=None):
        self.boxes.append(person_boxes)
        return [_calibrated(self.h)]

    def flush(self):
        return []


def test_the_pipeline_hands_the_person_detections_to_the_calibrator(tmp_path: Path):
    """The detector's person boxes reach a calibrator that takes them (the marking mask leaves them
    out); calibrators without the parameter, as above, keep working."""
    clip = render_synthetic_clip(tmp_path, n_frames=2, width=640, height=360, n_per_team=2,
                                 with_referee=False, seed=4, with_markings=False)
    cal = _BoxRecordingCalibrator(clip.homography)
    pipe = pl.smoke_pipeline(tracker=pl.GreedyIoUTracker(), calibrator=cal)
    pipe.run_clip(clip.clip_dir, tmp_path / "pred" / LABELS_FILE, game_id="g")
    assert len(cal.boxes) == 2
    for b in cal.boxes:
        assert b is not None and b.shape == (4, 4)  # two players a side, all detected
        assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()


def test_calibration_results_must_line_up_with_the_frames():
    pipe = pl.VisionPipeline(detector=pl.ColorBlobDetector(), tracker=pl.GreedyIoUTracker(), calibrator=FrameCalibrator())
    per_frame = [(1, 160, 90, pl.FrameDetections.empty()), (2, 160, 90, pl.FrameDetections.empty())]
    with pytest.raises(ValueError, match="calibration results"):
        pipe._to_gsr("CAL-8", "g", per_frame, {}, [_calibrated(np.eye(3))])


# ----------------------------------------------------------------------------
# Colab side: the worker and the runner pass calibration through (nothing runs remotely)
# ----------------------------------------------------------------------------

COLAB_DIR = Path(__file__).resolve().parents[2] / "colab"


def _load_colab(name: str):
    import importlib.util
    import sys

    key = f"{name}_under_calibration_test"
    spec = importlib.util.spec_from_file_location(key, COLAB_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod  # dataclasses on Python 3.14 look the module up here
    spec.loader.exec_module(mod)
    return mod


def test_worker_setup_forwards_the_calibration_switch(tmp_path: Path, monkeypatch):
    import sys
    import types

    worker = _load_colab("vision_worker")
    seen = []

    class FakePipe:
        def backends(self):
            return {"projector": "fake"}

    def fake_production_pipeline(**kw):
        seen.append(kw)
        return FakePipe()

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(pl, "production_pipeline", fake_production_pipeline)
    monkeypatch.setattr(sys, "path", list(sys.path))  # setup() prepends the repo dir
    monkeypatch.setattr(worker, "REPO_DIR", tmp_path)  # exists, so no clone
    monkeypatch.setattr(worker, "_sh", lambda *a, **k: None)
    monkeypatch.setattr(worker, "subprocess",
                        types.SimpleNamespace(run=lambda *a, **k: types.SimpleNamespace(stdout="deadbeef\n")))
    monkeypatch.setattr(worker, "_read_token", lambda: None)
    monkeypatch.setattr(worker, "_STATE", {})

    info = worker.setup(ref="abc", with_pose=False)
    assert seen[-1]["with_calibration"] is True and info["with_calibration"] is True  # default: on
    info = worker.setup(ref="abc", with_pose=False, with_calibration=False)
    assert seen[-1]["with_calibration"] is False and info["with_calibration"] is False
    assert worker._STATE["with_calibration"] is False


@pytest.mark.parametrize("calibrated", [True, False])
def test_worker_process_clip_reports_calibration_and_scores_it(tmp_path: Path, monkeypatch, calibrated: bool):
    worker = _load_colab("vision_worker")
    clip = render_synthetic_clip(tmp_path / "data" / "valid", n_frames=4, width=1280, height=720, n_per_team=3,
                                 with_referee=False, seed=0)
    pipe = pl.smoke_pipeline(tracker=pl.GreedyIoUTracker(), calibrator=FrameCalibrator() if calibrated else None)
    monkeypatch.setattr(worker, "_STATE", {"pipeline": pipe, "head": "deadbeef", "with_calibration": calibrated})
    monkeypatch.setattr(worker, "OUT_DIR", tmp_path / "out")
    rep = worker.process_clip(clip.clip_dir.name, split="valid", local_root=str(tmp_path / "data"))

    saved = json.loads((tmp_path / "out" / "valid" / clip.clip_dir.name / "report.json").read_text())
    assert saved["frames"] == 4 and saved["calibration"] == rep["calibration"]
    if calibrated:
        assert rep["calibration"]["calibrated"] == 1 and rep["calibration"]["tracked"] == 3
        assert rep["gs_hota"]["hota"] > 0.5 and rep["notes"] == []
    else:
        assert rep["calibration"] == {} and rep["gs_hota"] is None
        assert any("calibration is off" in n for n in rep["notes"]), rep["notes"]


def test_runner_passes_the_calibration_switch_to_setup(capsys, tmp_path: Path):
    runner = _load_colab("run_clips")
    base = ["--clips", "A", "--ref", "abc", "--out", str(tmp_path / "out"), "--dry-run"]  # dry run still makes folders
    assert runner.main(base) == 0
    assert "with_calibration=True" in capsys.readouterr().out
    assert runner.main(base + ["--no-calibration"]) == 0
    assert "with_calibration=False" in capsys.readouterr().out


def test_worker_reads_a_clip_out_of_the_split_zip(tmp_path: Path, monkeypatch):
    """hf_layout="zip": the clip is range-read out of <split>.zip (opened once per split), processed,
    and its frames deleted afterwards. The "remote" zip is in memory, in the SN-GSR-2025 layout."""
    import io
    import zipfile

    import varlab_vision.remote_zip as rz

    worker = _load_colab("vision_worker")
    clip = render_synthetic_clip(tmp_path / "src", n_frames=3, width=640, height=360, n_per_team=2,
                                 with_referee=False, seed=0, name="SNGS-077")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sequences_info.json", "{}")
        for p in sorted(clip.clip_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(clip.clip_dir.parent).as_posix())
    data = buf.getvalue()
    opened = []

    def fake_open(url, token=None, **kw):
        opened.append(url)
        rf = rz.RangeFile(lambda s, e: data[s:e], len(data), block_size=4096)
        return zipfile.ZipFile(rf), rf

    monkeypatch.setattr(rz, "open_remote_zip", fake_open)
    pipe = pl.smoke_pipeline(tracker=pl.GreedyIoUTracker())
    monkeypatch.setattr(worker, "_STATE", {"pipeline": pipe, "head": "deadbeef", "with_calibration": False,
                                           "hf_dataset": "SoccerNet/SN-GSR-2025", "hf_revision": "main",
                                           "hf_layout": "zip"})
    monkeypatch.setattr(worker, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(worker, "OUT_DIR", tmp_path / "out")

    rep = worker.process_clip("SNGS-077", split="valid")
    assert opened == ["https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/valid.zip"]
    assert rep["frames"] == 3 and rep["download"]["mb"] > 0 and rep["download"]["seconds"] >= 0
    assert (tmp_path / "out" / "valid" / "SNGS-077" / "Labels-GameState.json").is_file()
    assert not (tmp_path / "data" / "valid" / "SNGS-077" / "img1").exists()  # frames removed after the run
    worker.process_clip("SNGS-077", split="valid")
    assert len(opened) == 1  # the split's zip directory is read once per kernel


def test_worker_heartbeat_prints_while_a_step_is_silent(capsys):
    import time as _time

    worker = _load_colab("vision_worker")
    with worker._Heartbeat("SNGS-000", every=0.05):
        _time.sleep(0.2)
    out = capsys.readouterr().out
    assert out.count("SNGS-000: still working") >= 2
