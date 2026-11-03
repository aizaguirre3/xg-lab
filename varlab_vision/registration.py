"""Camera motion between consecutive frames, measured from the image itself.

A broadcast camera pans, tilts and zooms about a fixed point, so one homography maps the
whole previous frame onto the next, whatever the depth of what is in view (stands, boards
and grass alike). It is measured without the pitch markings: corners of the previous frame
are tracked into the next with pyramidal Lucas-Kanade, kept only when the backward track
returns to where it started, with the people left out (they move on their own), and a
homography is fitted to them with RANSAC. The fit counts only when enough inliers spread
over enough of the frame; a cut, a whip pan or a frame of motion blur gives a failed
:class:`Motion`, which breaks any chain of motions through it.

``calibration.bridge_gaps`` chains these motions from a calibrated frame into the frames
around it that the markings alone could not calibrate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Motion:
    h: np.ndarray  # previous frame's pixels -> this frame's pixels (3x3, full resolution)
    n_inliers: int
    spread: float  # share of the frame covered by the bounding box of the inliers
    residual_px: float  # median inlier error, full-resolution pixels
    ok: bool


def _failed(n: int = 0) -> Motion:
    return Motion(h=np.eye(3), n_inliers=n, spread=0.0, residual_px=float("nan"), ok=False)


class FrameRegistrar:
    """Feed frames in order with ``push``; each returns the motion from the previous frame
    (``None`` for the first frame after a reset or a change of frame size).

    Work is done on a grey copy ``work_width`` pixels wide. ``person_boxes`` (N, 4) are the
    people of the frame pushed; the corners are taken from the previous frame outside its
    people, padded by ``box_pad_px``.
    """

    def __init__(self, work_width: int = 960, max_corners: int = 600, min_distance: int = 12,
                 min_inliers: int = 60, min_spread: float = 0.2, fb_tol_px: float = 0.5,
                 ransac_px: float = 1.0, box_pad_px: float = 8.0):
        self.work_width = work_width
        self.max_corners = max_corners
        self.min_distance = min_distance
        self.min_inliers = min_inliers
        self.min_spread = min_spread
        self.fb_tol_px = fb_tol_px
        self.ransac_px = ransac_px
        self.box_pad_px = box_pad_px
        self.reset()

    def reset(self) -> None:
        self._prev: Optional[np.ndarray] = None
        self._prev_boxes: Optional[np.ndarray] = None
        self._scale = 1.0

    def _grey(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        h, w = frame_bgr.shape[:2]
        grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        self._scale = min(1.0, self.work_width / float(w))
        if self._scale < 1.0:
            grey = cv2.resize(grey, (int(round(w * self._scale)), int(round(h * self._scale))),
                              interpolation=cv2.INTER_AREA)
        return grey

    def _mask(self, shape, boxes: Optional[np.ndarray]) -> np.ndarray:
        mask = np.full(shape, 255, np.uint8)
        if boxes is not None and len(boxes):
            s, pad = self._scale, self.box_pad_px * self._scale
            for x0, y0, x1, y1 in np.asarray(boxes, float).reshape(-1, 4):
                r0, r1 = max(0, int(y0 * s - pad)), min(shape[0], int(np.ceil(y1 * s + pad)))
                c0, c1 = max(0, int(x0 * s - pad)), min(shape[1], int(np.ceil(x1 * s + pad)))
                mask[r0:r1, c0:c1] = 0
        return mask

    def _corners(self, grey: np.ndarray, mask: np.ndarray, grid: int = 4) -> Optional[np.ndarray]:
        """Corners picked per cell of a ``grid`` x ``grid`` split, so that the strong corners of a
        scoreboard or a sponsor board do not set the quality bar for the grass everywhere else."""
        import cv2

        h, w = grey.shape
        per_cell = max(1, self.max_corners // (grid * grid))
        found = []
        for r in range(grid):
            for c in range(grid):
                y0, y1, x0, x1 = r * h // grid, (r + 1) * h // grid, c * w // grid, (c + 1) * w // grid
                pts = cv2.goodFeaturesToTrack(grey[y0:y1, x0:x1], per_cell, 0.01, self.min_distance,
                                              mask=np.ascontiguousarray(mask[y0:y1, x0:x1]), blockSize=7)
                if pts is not None:
                    found.append(pts + np.array([x0, y0], np.float32))
        return np.concatenate(found).astype(np.float32) if found else None

    def push(self, frame_bgr: np.ndarray, person_boxes: Optional[np.ndarray] = None) -> Optional[Motion]:
        import cv2

        grey = self._grey(frame_bgr)
        prev, prev_boxes = self._prev, self._prev_boxes
        self._prev = grey
        self._prev_boxes = None if person_boxes is None else np.asarray(person_boxes, float).reshape(-1, 4)
        if prev is None or prev.shape != grey.shape:
            return None
        p0 = self._corners(prev, self._mask(prev.shape, prev_boxes))
        if p0 is None or len(p0) < self.min_inliers:
            return _failed()
        lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, grey, p0, None, **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(grey, prev, p1, None, **lk)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < self.fb_tol_px)
        a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        if len(a) < self.min_inliers:
            return _failed(len(a))
        hs, inl = cv2.findHomography(a, b, cv2.RANSAC, self.ransac_px, maxIters=2000, confidence=0.999)
        if hs is None or inl is None:
            return _failed()
        inl = inl.ravel().astype(bool)
        n = int(inl.sum())
        s = np.diag([self._scale, self._scale, 1.0])
        h = np.linalg.inv(s) @ hs @ s
        h = h / h[2, 2]
        if n < 4:
            return _failed(n)
        q = a[inl]
        span = (np.ptp(q[:, 0]) * np.ptp(q[:, 1])) / float(grey.shape[0] * grey.shape[1])
        proj = cv2.perspectiveTransform(q.reshape(-1, 1, 2), hs).reshape(-1, 2)
        resid = float(np.median(np.linalg.norm(proj - b[inl], axis=1))) / self._scale
        ok = n >= self.min_inliers and span >= self.min_spread and np.isfinite(h).all()
        return Motion(h=h, n_inliers=n, spread=float(span), residual_px=resid, ok=bool(ok))
