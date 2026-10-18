"""CPU smoke test of detection -> tracking -> team assignment -> GSR JSON on a synthetic clip.

The clip is 30 frames of coloured rectangles drifting over a green background,
rendered with OpenCV. This exercises the code path only: the colour-blob detector
and mean-colour team split stand in for RF-DETR and SigLIP, and no claim about
real-footage accuracy follows from it. The GS-HOTA check at the end closes the
loop loader -> pipeline -> loader -> metric on a clip whose answer is known.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from varlab_vision import pipeline as pl  # noqa: E402
from varlab_vision.gshota import gs_hota  # noqa: E402
from varlab_vision.loaders import list_frames, objects_frame, read_gsr_clip  # noqa: E402
from varlab_vision.synthetic import render_synthetic_clip  # noqa: E402


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> tuple:
    out = tmp_path_factory.mktemp("syn")
    clip = render_synthetic_clip(out, n_frames=30, width=640, height=360, n_per_team=3, with_referee=True, seed=1)
    return clip, read_gsr_clip(clip.labels_path)


def test_synthetic_clip_is_loader_compatible(synthetic):
    clip, gt = synthetic
    assert gt.info.seq_length == 30 and len(gt.images) == 30
    assert len(gt.objects) == 30 * 7
    seq = list_frames(clip.clip_dir, gt.info)
    assert len(seq) == 30
    df = objects_frame(gt)
    assert set(df["team"].dropna()) == {"left", "right"} and (df["role"] == "referee").sum() == 30
    assert df["x_pitch"].abs().max() <= 52.5 and df["y_pitch"].abs().max() <= 34.0


def test_color_blob_detector_finds_every_rectangle(synthetic):
    clip, gt = synthetic
    frame = cv2.imread(str(clip.image_dir / "000001.jpg"))
    det = pl.ColorBlobDetector().detect(frame)
    assert len(det) == 7
    gt_boxes = np.array([o.bbox_image.xyxy for o in gt.objects if o.image_id == gt.images[0].image_id])
    iou = pl._iou_matrix(gt_boxes, det.xyxy)
    assert (iou.max(axis=1) > 0.8).all()


@pytest.mark.parametrize("tracker_name", ["greedy", "bytetrack"])
def test_smoke_pipeline_end_to_end(synthetic, tmp_path: Path, tracker_name: str):
    clip, gt = synthetic
    if tracker_name == "bytetrack":
        pytest.importorskip("trackers")
        tracker = pl.ByteTrackTracker(frame_rate=25)
    else:
        tracker = pl.GreedyIoUTracker()
    pipe = pl.smoke_pipeline(projector=pl.HomographyProjector(clip.homography), tracker=tracker)
    out = tmp_path / tracker_name / "Labels-GameState.json"
    report = pipe.run_clip(clip.clip_dir, out, game_id="syn-game-1")
    assert report.n_frames == 30 and report.n_tracks >= 7
    assert report.backends["detector"] == "color-blob" and report.backends["tracker"] == tracker.name
    assert set(report.stages) >= {"read", "detect", "track", "team", "write", "total"}
    assert report.stages["total"].per_second > 0
    d = report.as_dict()
    assert d["stages"]["detect"]["items"] == 30

    pred = read_gsr_clip(out)
    assert pred.info.name == clip.clip_dir.name and pred.info.seq_length == 30
    assert len(pred.images) == 30 and [im.frame_index for im in pred.images] == list(range(1, 31))
    pdf = objects_frame(pred)
    assert len(pdf) >= 6 * 29  # ByteTrack confirms ids from frame 2
    assert pdf["track_id"].nunique() >= 7
    assert pdf["role"].eq("player").all() and pdf["jersey"].isna().all()
    assert set(pdf["team"].dropna()) == {"left", "right"}
    assert pdf["x_pitch"].notna().all()
    assert pred.objects[0].extra["confidence"] > 0.9

    # Closed loop against the known answer. Players: localisation is exact and the
    # two colour clusters are the two teams, so only the referee (predicted as a
    # player) costs one FP and one FN per frame.
    gdf = objects_frame(gt)
    full = gs_hota(gdf, pdf)
    players = gs_hota(gdf, pdf, roles=("player",))
    assert full.loca > 0.95
    assert players.deta > 0.75 and players.hota > 0.75
    assert full.hota > 0.65


def test_team_assignment_helpers():
    x = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1], [10, 10, 10], [11, 11, 11], [10, 11, 10]], float)
    km = pl.KMeansTeams(k=2, seed=3).fit(x)
    lab = km.predict(x)
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] == lab[5] and lab[0] != lab[3]
    teams = pl.assign_teams_by_track(np.array([1, 1, 2, 2, 3]), np.array([0, 0, 1, 1, 0]),
                                     np.array([500.0, 520.0, 10.0, 30.0, 480.0]))
    assert teams == {1: "right", 2: "left", 3: "right"}


def test_greedy_tracker_keeps_ids_and_handles_ball():
    tr = pl.GreedyIoUTracker()
    a = pl.FrameDetections(xyxy=np.array([[0, 0, 10, 20], [50, 50, 60, 70], [90, 90, 94, 94]], float),
                           confidence=np.ones(3), class_name=np.array(["person", "person", "ball"], dtype=object))
    r1 = tr.update(a)
    assert r1.tracker_id.tolist() == [1, 2, pl.BALL_TRACK_ID]
    b = pl.FrameDetections(xyxy=np.array([[52, 51, 62, 71], [1, 1, 11, 21]], float),
                           confidence=np.ones(2), class_name=np.array(["person", "person"], dtype=object))
    r2 = tr.update(b)
    assert r2.tracker_id.tolist() == [2, 1]


def test_production_backends_need_their_packages():
    """Constructors import lazily; without the heavy packages they raise ImportError, not earlier."""
    pytest.importorskip("numpy")
    try:
        import rfdetr  # noqa: F401
        pytest.skip("rfdetr installed; the real constructor would download weights")
    except ImportError:
        with pytest.raises(ImportError):
            pl.RFDETRDetector(size="nano")
