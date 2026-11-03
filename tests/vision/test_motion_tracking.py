"""Tracking in the coordinates of a reference frame: a fast pan moves nobody (no data, no GPU)."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from varlab_vision.pipeline import (CLASS_PERSON, FrameDetections, GreedyIoUTracker,  # noqa: E402
                                    MotionCompensatedTracker, _warp_boxes)
from varlab_vision.registration import FrameRegistrar  # noqa: E402
from varlab_vision.synthetic import BroadcastCamera, render_textured, textured_world  # noqa: E402

W, H = 640, 360
PLAYERS = np.array([[-38.0, -6.0], [-30.0, 4.0], [-24.0, -12.0], [-20.0, 10.0], [-34.0, 14.0]])  # standing still


def _clip(n=12, pan_m=2.5):
    world = textured_world(seed=5)
    cams = [BroadcastCamera(width=W, height=H, hfov_deg=40.0, position=(0.0, 60.0, 22.0), look_at=(-30.0 + pan_m * i, 0.0))
            for i in range(n)]
    frames, dets = [], []
    for cam in cams:
        img = render_textured(cam, world)
        feet, _ = cam.project(PLAYERS)
        boxes = np.stack([feet[:, 0] - 8, feet[:, 1] - 36, feet[:, 0] + 8, feet[:, 1]], axis=1)
        for b in boxes:
            x0, y0, x1, y1 = np.round(b).astype(int)
            img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = (30, 30, 200)
        inside = (boxes[:, 0] >= 0) & (boxes[:, 2] < W)
        frames.append(img)
        dets.append(FrameDetections(xyxy=boxes[inside], confidence=np.full(int(inside.sum()), 0.9),
                                    class_name=np.array([CLASS_PERSON] * int(inside.sum()), dtype=object)))
    return frames, dets, cams


def _n_ids(tracker, frames, dets, compensate):
    reg = FrameRegistrar()
    ids = set()
    for img, d in zip(frames, dets):
        motion = reg.push(img, d.xyxy)
        out = tracker.update(d, img, motion=motion) if compensate else tracker.update(d, img)
        ids |= set(out.tracker_id.tolist())
    return len(ids)


def test_warp_boxes_maps_the_corners():
    h = np.array([[2.0, 0, 10], [0, 1, -5], [0, 0, 1]])
    assert np.allclose(_warp_boxes(np.array([[0, 0, 4, 2]]), h), [[10, -5, 18, -3]])
    assert _warp_boxes(np.zeros((0, 4)), h).shape == (0, 4)


def test_a_fast_pan_breaks_plain_iou_tracking_but_not_the_compensated_one():
    frames, dets, _ = _clip()
    assert _n_ids(GreedyIoUTracker(max_lost=0), frames, dets, compensate=False) > 2 * len(PLAYERS)
    assert _n_ids(MotionCompensatedTracker(GreedyIoUTracker(max_lost=0)), frames, dets, compensate=True) == len(PLAYERS)


def test_a_new_shot_starts_a_reference_far_from_the_old_one():
    frames, dets, _ = _clip(n=4, pan_m=0.5)
    t = MotionCompensatedTracker(GreedyIoUTracker(max_lost=30))
    reg = FrameRegistrar()
    first = t.update(dets[0], frames[0], motion=reg.push(frames[0], dets[0].xyxy))
    again = t.update(dets[0], frames[0], motion=None, new_shot=True)  # the same boxes, but after a cut
    assert not set(first.tracker_id.tolist()) & set(again.tracker_id.tolist())
    assert np.array_equal(again.xyxy, dets[0].xyxy)  # ids come back on the original boxes
