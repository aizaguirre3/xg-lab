"""Loader tests on hand-written fixtures (no SoccerNet data, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from varlab_vision import gsr_format as gf
from varlab_vision.loaders import (
    camera_frame,
    check_game_disjoint,
    clip_game_ids,
    find_clips,
    iter_frames,
    list_frames,
    objects_frame,
    read_gsr_clip,
)


def test_read_clip_parses_info_images_objects_and_passthrough(clip_001: Path):
    clip = read_gsr_clip(clip_001)
    assert clip.info.name == "SNGS-001"
    assert clip.info.game_id == "11"
    assert clip.info.seq_length == 3 and clip.info.frame_rate == 25 and clip.info.im_ext == ".jpg"
    assert clip.info.extra["action_class"] == "Shots on target"  # unknown info keys survive
    assert [im.frame_index for im in clip.images] == [1, 2, 3]
    assert clip.images[2].is_labeled is False
    # 4 objects on frame 1, 2 on frame 2, 1 on frame 3; pitch + camera records pass through
    assert len(clip.objects) == 7
    assert {d["supercategory"] for d in clip.other_annotations} == {"pitch", "camera"}
    assert clip.categories[3] == ("referee", "object") and clip.categories[5] == ("pitch", "pitch")
    o = clip.objects[0]
    assert (o.track_id, o.role, o.team, o.jersey) == (1, "player", "left", "7")
    assert o.bbox_image.xyxy == (10.0, 20.0, 20.0, 50.0)
    assert o.bbox_image.bottom_middle == (15.0, 50.0)
    assert o.bbox_pitch is not None and o.bbox_pitch.xy == (-29.5, -10.0)
    assert o.bbox_pitch_raw is not None and o.bbox_pitch_raw.x_bottom_middle == pytest.approx(-29.7)
    ref = [x for x in clip.objects if x.role == "referee"][0]
    assert ref.team is None and ref.jersey is None
    ball = [x for x in clip.objects if x.role == "ball"][0]
    assert ball.bbox_pitch is None


def test_read_clip_accepts_folder_or_file(clip_001: Path):
    a = read_gsr_clip(clip_001)
    b = read_gsr_clip(clip_001 / "Labels-GameState.json")
    assert a.to_json() == b.to_json()


def test_round_trip_write_then_read(clip_001: Path, tmp_path: Path):
    clip = read_gsr_clip(clip_001)
    out = gf.write_labels_json(clip, tmp_path / "copy" / "Labels-GameState.json", indent=1)
    again = read_gsr_clip(out)
    assert again.to_json() == clip.to_json()
    raw = json.loads(out.read_text())
    assert set(raw) == {"info", "images", "annotations", "categories"}
    assert raw["annotations"][0]["attributes"] == {"role": "player", "jersey": "7", "team": "left"}
    assert raw["annotations"][0]["bbox_image"]["x_center"] == 15.0


def test_objects_frame_columns_and_labeled_filter(clip_001: Path):
    clip = read_gsr_clip(clip_001)
    df = objects_frame(clip)
    assert list(df.columns) == ["frame", "image_id", "track_id", "category_id", "role", "team", "jersey",
                                "x", "y", "w", "h", "x_pitch", "y_pitch", "calib_error", "calib_status"]
    assert df["frame"].tolist() == [1, 1, 1, 1, 2, 2]  # frame 3 is unlabelled
    assert np.isnan(df.loc[df["role"] == "ball", "x_pitch"]).all()
    assert objects_frame(clip, labeled_only=False)["frame"].tolist()[-1] == 3
    # the fixture's annotations never went through a calibrator: NaN / None in the optional columns
    assert df["calib_error"].isna().all() and df["calib_status"].isna().all()
    assert df["calib_error"].dtype == float


def test_missing_categories_falls_back_to_defaults(gsr_root: Path):
    clip = read_gsr_clip(gsr_root / "train" / "SNGS-002")  # fixture has no categories array
    assert clip.categories == gf.DEFAULT_CATEGORIES
    assert clip.objects[0].role == "goalkeeper" and clip.objects[0].category_id == 2


def test_role_from_category_when_attributes_missing(tmp_path: Path):
    raw = {"info": {"name": "X", "game_id": "1", "seq_length": 1},
           "images": [{"image_id": "1", "file_name": "000001.jpg", "width": 10, "height": 10}],
           "annotations": [{"id": "a", "image_id": "1", "track_id": 5, "category_id": 3,
                            "bbox_image": {"x": 0, "y": 0, "w": 2, "h": 2}}]}
    p = tmp_path / "Labels-GameState.json"
    p.write_text(json.dumps(raw))
    clip = read_gsr_clip(p)
    assert clip.objects[0].role == "referee"


def test_list_frames_and_iter_frames(clip_001: Path):
    clip = read_gsr_clip(clip_001)
    seq = list_frames(clip_001, clip.info)
    assert seq.frame_indices == [1, 2, 3]
    assert [p.name for p in seq.paths] == ["000001.jpg", "000002.jpg", "000003.jpg"]
    frames = list(iter_frames(seq))
    assert [i for i, _ in frames] == [1, 2, 3]
    assert frames[0][1].shape == (90, 160, 3)


def test_list_frames_strict_checks(clip_001: Path):
    clip = read_gsr_clip(clip_001)
    (clip_001 / "img1" / "000002.jpg").unlink()
    with pytest.raises(ValueError):
        list_frames(clip_001, clip.info)
    seq = list_frames(clip_001, clip.info, strict=False)
    assert seq.frame_indices == [1, 3]
    (clip_001 / "img1" / "notaframe.jpg").write_bytes(b"")
    with pytest.raises(ValueError):
        list_frames(clip_001, None)


def test_frame_index_parsing():
    assert gf.frame_index_from_file_name("000750.jpg") == 750
    assert gf.frame_file_name(7) == "000007.jpg"
    with pytest.raises(ValueError):
        gf.frame_index_from_file_name("frame_1.jpg")


def test_validate_attributes():
    assert gf.validate_attributes("player", "left", "10") == []
    assert gf.validate_attributes("player", "left", None) == []
    assert gf.validate_attributes("goalkeeper", "right", "1") == []
    assert gf.validate_attributes("referee", None, None) == []
    assert gf.validate_attributes("referee", "left", "3")
    assert gf.validate_attributes("player", "home", "10")
    assert gf.validate_attributes("player", "left", "ten")


def test_find_clips_and_game_ids(gsr_root: Path):
    assert [p.name for p in find_clips(gsr_root, "train")] == ["SNGS-001", "SNGS-002"]
    assert find_clips(gsr_root, "challenge") == []
    games = clip_game_ids(gsr_root)
    assert games["train"] == {"SNGS-001": "11", "SNGS-002": "12"}
    assert games["valid"] == {"SNGS-003": "11"}
    assert games["test"] == {} and games["challenge"] == {}


def test_game_disjoint_check_catches_the_leak(gsr_root: Path):
    res = check_game_disjoint(clip_game_ids(gsr_root))
    assert res.ok is False
    assert res.overlaps == {"11": ["train", "valid"]}
    assert res.clips_per_split["train"] == 2 and res.games_per_split["train"] == 2
    assert "VIOLATED" in res.summary() and "game 11 appears in train, valid" in res.summary()


def test_game_disjoint_check_passes_and_reports_missing_ids():
    ok = check_game_disjoint({"train": {"a": "1", "b": "2"}, "valid": {"c": "3"}, "test": {"d": "4"}})
    assert ok.ok and ok.overlaps == {}
    restricted = check_game_disjoint({"train": {"a": "1"}, "valid": {"c": "1"}, "test": {"d": "2"}},
                                     pairs=[("train", "test")])
    assert restricted.ok  # the train/valid leak is outside the requested pairs
    missing = check_game_disjoint({"train": {"a": "1", "b": ""}, "valid": {"c": "3"}})
    assert missing.ok is False and missing.missing_game_id == {"train": ["b"]}


# ----------------------------------------------------------------------------
# Calibration columns and camera records, on a tiny hand-written clip
# ----------------------------------------------------------------------------


def _calibrated_clip(tmp_path: Path) -> Path:
    """Three frames written as the vision pipeline writes them: frame 1 calibrated, frame 2 tracked
    (the camera records are listed out of order on purpose), frame 3 uncalibrated with null pitch data."""

    def obj(i: int, image_id: str, track: int, x_m, status: str, err):
        d = {"id": f"{image_id}{i:04d}", "image_id": image_id, "track_id": track, "supercategory": "object",
             "category_id": 1, "attributes": {"role": "player", "jersey": None, "team": "left"},
             "bbox_image": {"x": 10, "y": 20, "x_center": 15, "y_center": 35, "w": 10, "h": 30},
             "bbox_pitch": None, "confidence": 0.9, "calib_status": status, "calib_error": err}
        if x_m is not None:
            d["bbox_pitch"] = {"x_bottom_left": x_m - 1, "y_bottom_left": 5.0, "x_bottom_right": x_m + 1,
                               "y_bottom_right": 5.0, "x_bottom_middle": x_m, "y_bottom_middle": 5.0}
            d["pitch_x"], d["pitch_y"] = x_m, 5.0
        else:
            d["pitch_x"] = d["pitch_y"] = None
        return d

    def cam(image_id: str, status: str, n_in: int, fit, rep, cut: bool):
        return {"id": f"{image_id}9000", "image_id": image_id, "supercategory": "camera", "category_id": 6,
                "calibration": {"status": status, "homography_px2m": None, "n_lines": n_in + 2,
                                "n_inliers": n_in, "fit_error_px": fit, "reprojection_error_m": rep,
                                "shot_cut": cut, "matches": [], "note": ""}}

    images = [{"image_id": f"77{i:06d}", "file_name": f"{i:06d}.jpg", "width": 160, "height": 90,
               "is_labeled": True, "has_labeled_person": True, "has_labeled_camera": i < 3} for i in (1, 2, 3)]
    anns = [obj(1, "77000001", 1, -12.5, "calibrated", 1.25), obj(1, "77000002", 1, -12.0, "tracked", 1.5),
            obj(1, "77000003", 1, None, "uncalibrated", None),
            cam("77000003", "uncalibrated", 0, None, None, True),
            cam("77000001", "calibrated", 21, 1.6, 0.48, False),
            cam("77000002", "tracked", 19, 1.3, 0.39, False)]
    raw = {"info": {"name": "CAL-1", "game_id": "g", "seq_length": 3}, "images": images, "annotations": anns,
           "categories": gf.default_categories_list()}
    path = tmp_path / "CAL-1" / "Labels-GameState.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(raw))
    return path


def test_objects_frame_reads_calibration_extras(tmp_path: Path):
    df = objects_frame(read_gsr_clip(_calibrated_clip(tmp_path)))
    assert list(df.columns[-2:]) == ["calib_error", "calib_status"]  # appended at the end
    assert df["frame"].tolist() == [1, 2, 3]
    assert df["calib_status"].tolist() == ["calibrated", "tracked", "uncalibrated"]
    assert df["calib_error"].iloc[:2].tolist() == [1.25, 1.5]
    assert np.isnan(df["calib_error"].iloc[2])  # null in the JSON -> NaN
    assert df["calib_error"].dtype == float
    assert df["x_pitch"].iloc[:2].tolist() == [-12.5, -12.0] and np.isnan(df["x_pitch"].iloc[2])


def test_camera_frame_one_row_per_camera_record_in_frame_order(tmp_path: Path):
    clip = read_gsr_clip(_calibrated_clip(tmp_path))
    df = camera_frame(clip)
    assert list(df.columns) == ["frame", "image_id", "status", "n_inliers", "fit_error_px",
                                "reprojection_error_m", "shot_cut"]
    assert df["frame"].tolist() == [1, 2, 3]  # sorted although the records were not
    assert df["image_id"].tolist() == ["77000001", "77000002", "77000003"]
    assert df["status"].tolist() == ["calibrated", "tracked", "uncalibrated"]
    assert df["n_inliers"].tolist() == [21, 19, 0]
    assert df["fit_error_px"].iloc[:2].tolist() == [1.6, 1.3] and np.isnan(df["fit_error_px"].iloc[2])
    assert df["reprojection_error_m"].iloc[:2].tolist() == [0.48, 0.39] and np.isnan(df["reprojection_error_m"].iloc[2])
    assert df["shot_cut"].tolist() == [False, False, True] and df["shot_cut"].dtype == bool


def test_camera_frame_empty_and_foreign_camera_records(tmp_path: Path, clip_001: Path):
    empty_clip = read_gsr_clip(_calibrated_clip(tmp_path))
    empty_clip.other_annotations = []
    empty = camera_frame(empty_clip)
    assert len(empty) == 0
    assert list(empty.columns) == ["frame", "image_id", "status", "n_inliers", "fit_error_px",
                                   "reprojection_error_m", "shot_cut"]
    # the fixture's ground-truth camera record (pan / tilt / position, no calibration key) is still a row
    df = camera_frame(read_gsr_clip(clip_001))
    assert df["frame"].tolist() == [1] and df["status"].isna().all()
    assert df["n_inliers"].isna().all() and df["fit_error_px"].isna().all() and not df["shot_cut"].any()
    # a record that points at no image is an error, as for objects
    bad = read_gsr_clip(_calibrated_clip(tmp_path / "bad"))
    bad.other_annotations[0]["image_id"] = "nope"
    with pytest.raises(ValueError, match="unknown image_id"):
        camera_frame(bad)
