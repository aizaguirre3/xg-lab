"""Loader tests on hand-written fixtures (no SoccerNet data, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from varlab_vision import gsr_format as gf
from varlab_vision.loaders import (
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
                                "x", "y", "w", "h", "x_pitch", "y_pitch"]
    assert df["frame"].tolist() == [1, 1, 1, 1, 2, 2]  # frame 3 is unlabelled
    assert np.isnan(df.loc[df["role"] == "ball", "x_pitch"]).all()
    assert objects_frame(clip, labeled_only=False)["frame"].tolist()[-1] == 3


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
