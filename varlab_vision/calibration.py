"""Camera-to-pitch calibration from the pitch markings, written from scratch.

Classical and licence-free: no learned keypoint detector, no code from
PnLCalib (GPL-2) or TrackLab / sn-gamestate (GPL-3). The chain is

1. ``marking_mask``: thin structures brighter than the grass around them (a white top-hat
   on the minimum colour channel, thresholded against the local grass level of that
   response), inside the grass region (``grass_region``: the largest green area, holes
   filled) and outside the given person boxes. ``white_line_mask`` (bright, unsaturated
   pixels near grass) is the earlier mask, kept for comparison (``mask="white"``).
2. ``extract_lines``: probabilistic Hough segments as seeds; each seed is refitted to the
   mask pixels around it by a robust cross-section-centroid fit (Theil-Sen + Tukey, with a
   band that follows the stroke thickness) and then *peeled*: its pixels are removed from
   the live mask so no second line can be found on the same marking. A candidate that
   bows like a piece of a curve (``stroke_bend``) is dropped.
3. ``detect_ellipses``: the curved marking pixels (the centre circle, the penalty arcs) as
   ellipses, grown from connected pieces and refitted geometrically; straight candidates
   lying on one are dropped (``drop_chords``).
4. ``search_homography``: a RANSAC-style hypothesis search over assignments of
   four image lines to two ``x = const`` and two ``y = const`` lines of the
   :class:`~varlab_vision.pitch_model.PitchModel`, plus hypotheses from the centre circle,
   the halfway line and one touchline (``_circle_hypotheses``). Every hypothesis is
   scored on all detected lines and ellipses: an image line is an inlier of a model line
   when its endpoints lie within ``tau_px`` of the projected model line *and* its
   back-projection lies on a drawn segment of that line (not on the invisible part of the
   infinite line); an ellipse is an inlier of a model circle when its points lie on the
   projected circle's drawn part. The search is vectorised (batched DLT) and the scorer is
   invariant to the scale and sign a caller gives the homography.
5. ``refine_homography``: Levenberg-Marquardt on the eight homography parameters,
   minimising the pixel distances of the inlier segments' endpoints (and the matched
   ellipses' points) to the projected model markings, then again as a pinhole camera
   (focal length, rotation, position).
6. ``verify_solution``: the evidence for a solution in the frame itself (marking support,
   unexplained lines, leave-one-out agreement, the camera it implies, the fit's standard
   error where the people are); ``VerifyConfig`` decides when it is refused.
7. ``FrameCalibrator``: per-frame state. A shot-cut test (grey thumbnail difference or
   hue-saturation histogram distance) clears all state. Otherwise frame t is *tracked*
   from the filter's one-step prediction (then from the previous solution), falling back
   to the global search when tracking has too few correspondences or keeps clearly fewer
   inliers than the last frame (a false lock). A tracked fit goes through an alpha-beta
   (constant-velocity) filter on the eight homography parameters, so a steady pan is
   followed without lag while fit noise is averaged on a static shot. With ``push`` /
   ``flush`` a frame stride applies: during steady tracking only about one frame in six is
   fitted and the frames between are interpolated between fits (status ``interpolated``),
   each one checked against its own markings (see ``DEFAULT_FRAME_STRIDE``).

A frame is **calibrated** only when enough markings match (two lines of each family, or
a circle with a line of each) *and* the solution passes verification; anything else is
reported as ``uncalibrated`` with NaN pitch coordinates and the reason in ``note``, never
as a guess.

Errors reported per frame: ``fit_error_px`` (RMS pixel distance of inlier
endpoints to their projected model lines, the quantity minimised) and
``reprojection_error_m`` (RMS metre distance of the same endpoints,
back-projected, to the model lines: how the pixel error looks on the pitch).

Limits, stated once: single camera, planar pitch (no height, no lens distortion), a
broadcast main camera on the ``+y`` side (the verification's camera prior), and a view
that shows enough markings. On every 50th frame of the 58 SoccerNet GSR validation clips
(``tools/eval_gsr_calibration.py``, numbers in ``analysis/vision_env.md``) it claims 40% of
the frames, 97% of those with a median error under 1 m against the labels and none over
5 m, and refuses the rest.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pitch_model import DEFAULT_PITCH, X_CONST, Y_CONST, ModelLine, PitchModel

STATUS_CALIBRATED = "calibrated"
STATUS_TRACKED = "tracked"
STATUS_INTERPOLATED = "interpolated"  # held back by the frame stride: interpolated between fits, no line fit
STATUS_UNCALIBRATED = "uncalibrated"


# ----------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------


@dataclass
class ImageLine:
    """A straight marking in the image: sub-pixel line plus its visible extent."""

    p0: np.ndarray
    p1: np.ndarray
    abc: np.ndarray  # a x + b y + c = 0, with a^2 + b^2 = 1
    n_pixels: int
    bend_px: float = 0.0  # how far the stroke bows away from the line over its extent (see stroke_bend)
    half_width_px: float = float("nan")  # half the stroke width

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def mid(self) -> np.ndarray:
        return (self.p0 + self.p1) / 2.0

    @property
    def angle(self) -> float:
        """Direction in radians, in [0, pi)."""
        d = self.p1 - self.p0
        return float(math.atan2(d[1], d[0]) % math.pi)

    def endpoints(self) -> np.ndarray:
        return np.stack([self.p0, self.p1], axis=0)


@dataclass
class LineMatch:
    image_index: int
    model_name: str
    distance_px: float
    extent_fraction: float


@dataclass
class CalibrationResult:
    status: str
    h_m2px: Optional[np.ndarray] = None
    h_px2m: Optional[np.ndarray] = None
    matches: List[LineMatch] = field(default_factory=list)
    n_lines: int = 0
    fit_error_px: float = float("nan")
    reprojection_error_m: float = float("nan")
    shot_cut: bool = False
    note: str = ""
    # the markings the frame was calibrated from, kept so a caller can re-score a
    # different homography (the tracker's filtered one) without extracting again
    lines: List["ImageLine"] = field(default_factory=list, repr=False, compare=False)
    ellipses: List["ImageEllipse"] = field(default_factory=list, repr=False, compare=False)
    arc_matches: List["ArcMatch"] = field(default_factory=list)
    verify: Dict[str, object] = field(default_factory=dict)  # see verify_solution; on a refused frame, why

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_CALIBRATED, STATUS_TRACKED, STATUS_INTERPOLATED) and self.h_px2m is not None

    @property
    def n_inliers(self) -> int:
        """Matched markings: lines plus circles."""
        return len(self.matches) + len(self.arc_matches)

    def project(self, points_px: np.ndarray) -> np.ndarray:
        """Pixels -> pitch metres; NaN everywhere when the frame is uncalibrated."""
        pts = np.asarray(points_px, float).reshape(-1, 2)
        if not self.ok:
            return np.full((len(pts), 2), np.nan)
        return apply_homography(self.h_px2m, pts)

    def project_boxes(self, xyxy: np.ndarray) -> Dict[str, np.ndarray]:
        """Bottom-left, bottom-right and bottom-middle of each box in metres."""
        b = np.asarray(xyxy, float).reshape(-1, 4)
        left = self.project(np.stack([b[:, 0], b[:, 3]], axis=1))
        right = self.project(np.stack([b[:, 2], b[:, 3]], axis=1))
        return {"left": left, "right": right, "middle": (left + right) / 2.0}

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "homography_px2m": None if self.h_px2m is None else np.round(self.h_px2m, 8).tolist(),
            "n_lines": int(self.n_lines),
            "n_inliers": int(self.n_inliers),
            "fit_error_px": None if math.isnan(self.fit_error_px) else round(self.fit_error_px, 3),
            "reprojection_error_m": (None if math.isnan(self.reprojection_error_m)
                                     else round(self.reprojection_error_m, 3)),
            "shot_cut": bool(self.shot_cut),
            "verify": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.verify.items()},
            "matches": [{"line": m.model_name, "image_index": m.image_index,
                         "distance_px": round(m.distance_px, 2)} for m in self.matches],
            "arcs": [{"arc": m.model_name, "ellipse_index": m.ellipse_index,
                      "distance_px": round(m.distance_px, 2)} for m in self.arc_matches],
            "note": self.note,
        }


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------


def apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ np.asarray(h, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / hom[:, 2:3]


def _normalise_lines(abc: np.ndarray) -> np.ndarray:
    """Scale so that a^2 + b^2 = 1 (works on (..., 3) arrays)."""
    n = np.sqrt(abc[..., 0] ** 2 + abc[..., 1] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return abc / n[..., None]


def project_model_lines(h_px2m: np.ndarray, model_abc: np.ndarray) -> np.ndarray:
    """Model lines (metres, (M, 3)) -> image lines (pixels, (M, 3), normalised).

    A line ``l`` transforms by the inverse transpose of the point homography:
    with ``x_m = H x_px``, ``l_px = H^T l_m``.
    """
    return _normalise_lines(model_abc @ np.asarray(h_px2m, float))


def _dlt_batch(src: np.ndarray, dst: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    """Batched 4-point DLT: ``src`` and ``dst`` are (B, 4, 2); returns (B, 3, 3)
    homographies with ``dst ~ H src`` and ``h33 = 1``.

    Solved through the 8x8 normal equations, each scaled by its own trace and given a
    small ridge, so every system is positive definite and well scaled. It never raises:
    a degenerate hypothesis (collinear points, an intersection far outside the image)
    comes back finite but useless, or NaN, and the scorer discards it. This matters
    because LAPACK builds differ: OpenBLAS (Linux wheels) raises ``LinAlgError`` on an
    exactly singular member and would abort the whole batch, where Accelerate (macOS
    wheels) returned a finite answer for the same system.
    """
    B = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    z = np.zeros_like(x)
    o = np.ones_like(x)
    r1 = np.stack([x, y, o, z, z, z, -u * x, -u * y], axis=-1)
    r2 = np.stack([z, z, z, x, y, o, -v * x, -v * y], axis=-1)
    a = np.concatenate([r1, r2], axis=1)  # (B, 8, 8)
    b = np.concatenate([u, v], axis=1)[..., None]  # (B, 8, 1)
    ata = np.einsum("bij,bik->bjk", a, a)
    atb = np.einsum("bij,bik->bjk", a, b)
    scale = np.trace(ata, axis1=1, axis2=2) / 8.0
    good = np.isfinite(scale) & (scale > 0) & np.isfinite(ata).all(axis=(1, 2)) & np.isfinite(atb).all(axis=(1, 2))
    scale = np.where(good, scale, 1.0)
    ata = np.where(good[:, None, None], ata / scale[:, None, None], 0.0) + ridge * np.eye(8)[None]
    atb = np.where(good[:, None, None], atb / scale[:, None, None], 0.0)
    try:
        sol = np.linalg.solve(ata, atb)[..., 0]
    except np.linalg.LinAlgError:  # safety net only; the scaling above makes every system definite
        sol = np.full((B, 8), np.nan)
        for i in range(B):
            try:
                sol[i] = np.linalg.solve(ata[i], atb[i])[:, 0]
            except np.linalg.LinAlgError:
                pass
    sol[~good] = np.nan
    return np.concatenate([sol, np.ones((B, 1))], axis=1).reshape(B, 3, 3)


def _quad_orientation_ok(pts: np.ndarray, min_area: float = 50.0) -> np.ndarray:
    """Shoelace test on (B, 4, 2) image quads ordered (xa,ya), (xa,yb), (xb,ya), (xb,yb).

    Re-ordered as a ring (xa,ya) -> (xb,ya) -> (xb,yb) -> (xa,yb), the model quad has
    positive signed area in a y-down frame; a non-mirrored camera keeps that sign.
    """
    ring = pts[:, [0, 2, 3, 1], :]
    x, y = ring[..., 0], ring[..., 1]
    area = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    return np.isfinite(area) & (area > min_area)


def _segment_intervals(model: PitchModel) -> Tuple[np.ndarray, np.ndarray, List[ModelLine]]:
    """Per model line: (M, 3) homogeneous form and (M, 2, 2) drawn intervals along
    the free coordinate (second interval NaN when the line is one segment)."""
    lines = model.lines()
    abc = np.stack([ln.homogeneous for ln in lines], axis=0)
    iv = np.full((len(lines), 2, 2), np.nan)
    for i, ln in enumerate(lines):
        for k, (p0, p1) in enumerate(ln.segments[:2]):
            t0, t1 = (p0[1], p1[1]) if ln.kind == X_CONST else (p0[0], p1[0])
            iv[i, k] = (min(t0, t1), max(t0, t1))
    return abc, iv, lines


def extent_fraction(t_lo: np.ndarray, t_hi: np.ndarray, intervals: np.ndarray, tol_m: float) -> np.ndarray:
    """Fraction of ``[t_lo, t_hi]`` covered by the (tolerance-widened) drawn intervals.

    Broadcasts: ``t_lo``/``t_hi`` of shape (...,) against ``intervals`` (..., 2, 2).
    """
    seg = np.maximum(t_hi - t_lo, 1e-9)
    cov = np.zeros_like(seg)
    for k in range(intervals.shape[-2]):
        a = intervals[..., k, 0] - tol_m
        b = intervals[..., k, 1] + tol_m
        ov = np.minimum(t_hi, b) - np.maximum(t_lo, a)
        cov = cov + np.where(np.isnan(a), 0.0, np.clip(ov, 0.0, None))
    return np.clip(cov / seg, 0.0, 1.0)


# ----------------------------------------------------------------------------
# 1 + 2. Marking pixels and lines
# ----------------------------------------------------------------------------


def white_line_mask(frame_bgr: np.ndarray, white_max_sat: int = 70, white_min_val: int = 150,
                    green_hue: Tuple[int, int] = (35, 85), green_min_sat: int = 60,
                    grass_dilate_px: int = 15) -> np.ndarray:
    """Binary mask of marking pixels: bright, unsaturated, and within ``grass_dilate_px`` of grass."""
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (s <= white_max_sat) & (v >= white_min_val)
    green = ((h >= green_hue[0]) & (h <= green_hue[1]) & (s >= green_min_sat)).astype(np.uint8)
    if grass_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grass_dilate_px + 1, 2 * grass_dilate_px + 1))
        green = cv2.dilate(green, k)
    return (white & (green > 0)).astype(np.uint8)


def grass_region(frame_bgr: np.ndarray, green_hue: Tuple[int, int] = (30, 90), green_min_sat: int = 40,
                 green_min_val: int = 30, scale: int = 4) -> np.ndarray:
    """The playing surface: the largest connected green area, with its holes filled (uint8 0/1).

    Found on a ``scale`` times smaller image (the boundary is then accurate to about a
    pixel after the linear upsampling). Filling the holes brings back everything the grass
    surrounds, the markings and the players included, and leaves out what lies beyond the
    grass: stands, advertising boards, the track. On real broadcast frames the boards hold
    white text right above the far touchline, which a mask restricted to *pixels near
    grass* (``white_line_mask``) takes as markings.
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    g = ((hsv[..., 0] >= green_hue[0]) & (hsv[..., 0] <= green_hue[1]) & (hsv[..., 1] >= green_min_sat)
         & (hsv[..., 2] >= green_min_val)).astype(np.uint8)
    g0 = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Close before labelling: a marking that runs to the frame edge would otherwise cut the grass in
    # two. When the largest piece still holds under 60% of the green the markings are thicker than the
    # closing bridges (near-camera strokes), so close harder.
    for k in (5, 11):
        g = cv2.morphologyEx(g0, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=4)
        if n <= 1:
            return np.zeros((h, w), np.uint8)
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[big, cv2.CC_STAT_AREA] >= 0.6 * stats[1:, cv2.CC_STAT_AREA].sum():
            break
    g = (lab == big).astype(np.uint8)
    outside = np.pad(1 - g, 1, constant_values=1)  # non-grass, framed so that the flood starts outside the image
    cv2.floodFill(outside, np.zeros((outside.shape[0] + 2, outside.shape[1] + 2), np.uint8), (0, 0), 2)
    g = (g.astype(bool) | (outside[1:-1, 1:-1] == 1)).astype(np.float32)  # holes: non-grass the flood did not reach
    return (cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)


def person_box_mask(shape: Tuple[int, int], person_boxes: Optional[np.ndarray], pad_px: float = 3.0,
                    pad_frac: float = 0.03) -> np.ndarray:
    """Boolean mask of the pixels inside the given person boxes (``(N, 4)`` x0, y0, x1, y1), each grown by
    ``pad_px + pad_frac * box height``."""
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=bool)
    if person_boxes is None:
        return out
    for x0, y0, x1, y1 in np.asarray(person_boxes, float).reshape(-1, 4):
        if not np.isfinite([x0, y0, x1, y1]).all():
            continue
        p = pad_px + pad_frac * max(0.0, y1 - y0)
        a0, b0 = max(0, int(math.floor(x0 - p))), max(0, int(math.floor(y0 - p)))
        a1, b1 = min(w, int(math.ceil(x1 + p))), min(h, int(math.ceil(y1 + p)))
        if a1 > a0 and b1 > b0:
            out[b0:b1, a0:a1] = True
    return out


def marking_mask(frame_bgr: np.ndarray, person_boxes: Optional[np.ndarray] = None, tophat_px: int = 21,
                 noise_window_px: int = 31, noise_mult: float = 2.5, margin: float = 11.0, blur_sigma: float = 1.0,
                 grass: Optional[np.ndarray] = None, level_cap: int = 25) -> np.ndarray:
    """Marking pixels by local contrast (uint8 0/1): thin structures brighter than the grass around them.

    The luminance is the minimum colour channel (white paint is bright in all three, grass is
    dark in at least one, and so are coloured boards and shirts), lightly blurred against
    JPEG noise. A white top-hat with a ``tophat_px`` square keeps what is brighter than its
    surroundings and narrower than the square: the markings. The threshold is relative to
    the local grass level of that response, its median over ``noise_window_px``, which is
    the grass texture (markings are too sparse to move a median): a pixel is a marking pixel
    when its top-hat reaches ``noise_mult`` times that level plus ``margin``. A fixed
    brightness threshold (``white_line_mask``'s V >= 150) misses the faint distant lines
    of a broadcast frame and takes white shirts and board text instead.

    Only pixels inside :func:`grass_region` count, and the pixels inside ``person_boxes``
    (``(N, 4)`` x0, y0, x1, y1 in pixels, see :func:`person_box_mask`) are removed: a white
    shirt is as bright and as thin as a marking.

    Measured on 69 frames of 15 SoccerNet GSR *train* clips against the markings projected
    with the homography the labels imply (a marking sample is found when a mask pixel is
    within 3 px; a mask pixel is right when a marking is within 5 px; labelled people
    masked): ``white_line_mask`` found 35% of the markings with 6.5% of its pixels right
    (median 63k pixels a frame); this mask finds 66% with 48% right (22k pixels).
    """
    import cv2

    lum = frame_bgr.min(axis=2) if frame_bgr.ndim == 3 else frame_bgr
    if blur_sigma > 0:
        lum = cv2.GaussianBlur(lum, (0, 0), blur_sigma)
    th = cv2.morphologyEx(lum, cv2.MORPH_TOPHAT, np.ones((tophat_px, tophat_px), np.uint8))
    # capped first: where thick markings fill half the window the median would be the marking itself
    level = cv2.medianBlur(np.minimum(th, level_cap).astype(np.uint8), noise_window_px)
    m = th.astype(np.float32) >= noise_mult * level.astype(np.float32) + margin
    m &= (grass if grass is not None else grass_region(frame_bgr)) > 0
    if person_boxes is not None and len(person_boxes):
        m &= ~person_box_mask(m.shape, person_boxes)
    return m.astype(np.uint8)


def line_weights(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Soft whiteness (0-1, float32) of the pixels around the marking mask.

    A binary mask throws away the sub-pixel position of a stroke: a 3 px line whose
    centre sits at row 100.0 and one at row 100.25 threshold to the same rows, so a
    near-horizontal line such as the far touchline keeps a constant bias of up to a
    quarter of a pixel along its whole length, which is half a metre on the far
    side of a wide shot. The minimum colour channel is linear in how much of a pixel
    the white paint covers (grass is dark in at least one channel, paint is not), so
    it is rescaled between the grass level around the mask and the paint level inside
    it and used as the weight of each pixel in the line fit. Zero away from the mask
    (3 px reach). Falls back to the binary mask when the mask is too small to
    calibrate against.
    """
    import cv2

    on = mask > 0
    if int(on.sum()) < 60:
        return mask.astype(np.float32)
    m = np.minimum(np.minimum(frame_bgr[..., 0], frame_bgr[..., 1]), frame_bgr[..., 2])
    near = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
    ring = cv2.dilate(mask, np.ones((17, 17), np.uint8)) > 0
    ring &= ~near
    hi = float(np.percentile(m[on], 90))
    lo = float(np.median(m[ring])) if int(ring.sum()) >= 60 else 0.0
    if hi - lo < 20.0:
        return mask.astype(np.float32)
    w = np.clip((m.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    w[~near] = 0.0
    return w


def _fit_line_pca(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through (N, 2) points: (abc normalised, direction)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    abc = np.array([n[0], n[1], -(n @ c)])
    return abc, d


def _band_select(pix: np.ndarray, ys: np.ndarray, alive: np.ndarray, c: np.ndarray, u: np.ndarray,
                 lo: float, hi: float, band: float) -> np.ndarray:
    """Indices of the live mask pixels within ``band`` of the line (centre ``c``,
    direction ``u``) whose position along it lies in ``[lo, hi]``.

    ``pix`` is row-major (``np.nonzero`` order) so ``ys`` is sorted and the
    candidate rows form one contiguous slice: the cost scales with the rows the
    line crosses, not with the whole mask.
    """
    e0, e1 = c + u * lo, c + u * hi
    r = band + 1.5
    i0 = int(np.searchsorted(ys, int(math.floor(min(e0[1], e1[1]) - r)), side="left"))
    i1 = int(np.searchsorted(ys, int(math.ceil(max(e0[1], e1[1]) + r)), side="right"))
    if i1 <= i0:
        return np.zeros(0, dtype=np.int64)
    rel = pix[i0:i1] - c
    t = rel @ u
    s = rel[:, 1] * u[0] - rel[:, 0] * u[1]
    m = alive[i0:i1] & (np.abs(s) <= band) & (t >= lo) & (t <= hi)
    return np.nonzero(m)[0] + i0


def _fit_profile(pts: np.ndarray, wts: np.ndarray, c: np.ndarray, u: np.ndarray, bin_px: float = 6.0,
                 tukey_px: float = 1.5) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, float, int]]:
    """Robust line fit to the pixels of one stroke through its cross-section centroids.

    The pixels (with soft weights ``wts``, 1 for a binary mask) are binned along the
    seed direction ``u`` (bins of ``bin_px``), each bin is reduced to its centroid, and
    a line is fitted to the centroids by Theil-Sen followed by Tukey-biweight
    reweighting. Unlike a plain PCA over the pixels this (a) is unbiased by how the
    band cuts a tilted stroke, because every centroid is taken across the full
    cross-section, and (b) rejects the bins at a T-junction or a corner where another
    marking adds pixels, which would drag a PCA line off a short marking by one to two
    pixels. A bin's centre is the weighted mean of its pixels within the stroke
    half-width of the bin's weighted median: the median finds the stroke in spite of
    a stub on one side, the weighted mean then resolves its position to a fraction of
    a pixel.

    Returns ``(centre, direction, t_lo, t_hi, thickness_px, n_pixels)`` with the
    extent and pixel count taken from the inlier bins only (``thickness_px`` is the
    pixel support per unit length, used to size the band), or ``None`` when fewer
    than four bins carry data.
    """
    n = np.array([-u[1], u[0]])
    rel = pts - c
    t = rel @ u
    s_ = rel @ n
    k = np.floor((t - t.min()) / bin_px).astype(np.int64)
    nb = int(k.max()) + 1
    cnt = np.bincount(k, minlength=nb)
    wsum = np.bincount(k, weights=wts, minlength=nb)
    ok = (cnt >= 3) & (wsum >= 2.5)
    if ok.sum() < 4:
        return None
    # weighted median of the cross-section coordinate per bin
    order = np.lexsort((s_, k))
    cw = np.cumsum(wts[order])
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    before = np.where(first > 0, cw[np.maximum(first - 1, 0)], 0.0)
    med = s_[order][np.minimum(np.searchsorted(cw, before + 0.5 * wsum, side="left"), len(order) - 1)]
    support = np.median(cnt[ok][1:-1] if ok.sum() > 3 else cnt[ok]) / bin_px
    gate = np.abs(s_ - med[k]) <= max(1.5, 0.5 * support + 1.0)
    gw = wts * gate
    den = np.bincount(k, weights=gw, minlength=nb)
    sm = (np.bincount(k, weights=gw * s_, minlength=nb) / np.maximum(den, 1e-9))[ok]
    tm = (np.bincount(k, weights=gw * t, minlength=nb) / np.maximum(den, 1e-9))[ok]
    wc = den[ok]
    m = len(sm)
    if m > 48:  # Theil-Sen on a thinned set keeps the pair count bounded
        sub = np.linspace(0, m - 1, 48).astype(int)
    else:
        sub = np.arange(m)
    ii, jj = np.triu_indices(len(sub), k=1)
    dt = tm[sub][jj] - tm[sub][ii]
    good = dt > 2.0 * bin_px
    if not good.any():
        return None
    gamma = float(np.median((sm[sub][jj] - sm[sub][ii])[good] / dt[good]))
    alpha = float(np.median(sm - gamma * tm))
    for _ in range(4):
        r = sm - (alpha + gamma * tm)
        wt = wc * np.clip(1.0 - (r / tukey_px) ** 2, 0.0, None) ** 2
        sw = wt.sum()
        if sw <= 0:
            return None
        tbar, sbar = (wt * tm).sum() / sw, (wt * sm).sum() / sw
        den2 = (wt * (tm - tbar) ** 2).sum()
        gamma = float((wt * (tm - tbar) * (sm - sbar)).sum() / den2) if den2 > 1e-9 else gamma
        alpha = float(sbar - gamma * tbar)
    r = sm - (alpha + gamma * tm)
    inl_bin = np.zeros(nb, dtype=bool)
    ok_idx = np.nonzero(ok)[0]
    inl_bin[ok_idx[np.abs(r) <= 0.7 * tukey_px]] = True
    if inl_bin.sum() < 3:
        return None
    nrm = math.sqrt(1.0 + gamma * gamma)
    u2 = (u + gamma * n) / nrm
    c2 = c + alpha * n
    keep = inl_bin[k]
    tt = (pts[keep] - c2) @ u2
    cn = cnt[inl_bin]
    thickness = float(np.median(cn[1:-1] if len(cn) > 3 else cn) / bin_px)
    return c2, u2, float(tt.min()), float(tt.max()), thickness, int(keep.sum())


def stroke_bend(pts: np.ndarray, wts: np.ndarray, c: np.ndarray, u: np.ndarray, lo: float, hi: float,
                half_w: float = 1.5, bin_px: float = 6.0) -> Tuple[float, float, int]:
    """How far a stroke bends away from its fitted straight line.

    The stroke's pixels (``pts``, soft weights ``wts``) between ``lo`` and ``hi`` along the
    line (centre ``c``, unit direction ``u``) are reduced to one cross-section centroid per
    ``bin_px`` bin, as in the line fit, and a parabola ``s = a + b t + q t^2`` is fitted to
    the centroids by least squares with Tukey reweighting (a T-junction or a corner adds an
    outlying bin at one end, an arc bends both ends the same way). Returns ``(sagitta_px,
    sigma_px, bins)``: the parabola's bow over the extent, ``|q| (hi - lo)^2 / 4``, the
    robust scatter of the centroids about it, and the number of bins used. A straight
    marking bows by a fraction of a pixel over its length (lens distortion adds a pixel or
    two over the full frame width); a chord through the centre circle or a penalty arc bows
    by its whole stroke width or more.
    """
    n = np.array([-u[1], u[0]])
    rel = pts - c
    t = rel @ u
    s = rel @ n
    sel = (t >= lo) & (t <= hi)
    t, s, wv = t[sel], s[sel], wts[sel]
    if len(t) < 8 or hi - lo < 4 * bin_px:
        return 0.0, float("nan"), 0
    k = np.floor((t - lo) / bin_px).astype(np.int64)
    nb = int(k.max()) + 1
    wsum = np.bincount(k, weights=wv, minlength=nb)
    ok = wsum >= 1.5
    if ok.sum() < 5:
        return 0.0, float("nan"), int(ok.sum())
    sm = (np.bincount(k, weights=wv * s, minlength=nb) / np.maximum(wsum, 1e-9))[ok]
    tm = (np.bincount(k, weights=wv * t, minlength=nb) / np.maximum(wsum, 1e-9))[ok]
    mid, half = 0.5 * (lo + hi), max(0.5 * (hi - lo), 1e-6)
    x = (tm - mid) / half  # in [-1, 1]: the parabola's q * half^2 is then the coefficient of x^2
    a_mat = np.stack([np.ones_like(x), x, x * x], axis=1)
    w = np.ones_like(x)
    coef = np.zeros(3)
    scale = max(1.0, half_w)
    for _ in range(5):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(a_mat * sw[:, None], sm * sw, rcond=None)[0]
        r = sm - a_mat @ coef
        w = np.clip(1.0 - (r / (2.5 * scale)) ** 2, 0.0, None) ** 2
        if w.sum() < 4:
            break
    r = sm - a_mat @ coef
    good = w > 0
    sigma = float(1.4826 * np.median(np.abs(r[good]))) if good.sum() >= 4 else float("nan")
    return float(abs(coef[2])), sigma, int(good.sum())


# Bend test used by calibrate_frame: a candidate is a piece of a curve when it bows by more than
# DEFAULT_MAX_BEND_PX over its extent and its radius of curvature is under DEFAULT_MIN_BEND_RADIUS frame
# widths. Measured on 1,213 lines of 113 SoccerNet GSR train frames, labelled straight or arc with the
# homography the people labels imply: this drops 62% of the arc chords and 5.6% of the straight lines
# (1.9% of their length: short ones).
DEFAULT_MAX_BEND_PX = 0.7
DEFAULT_MIN_BEND_RADIUS = 1.3
# The contrast mask keeps the anti-aliased fringe on one side of a stroke; a peel band of half the
# stroke plus 1 px left it behind as a short parallel "line" 4 px off the marking (synthetic wide shot).
DEFAULT_PEEL_HALO_PX = 2.0


def extract_lines(mask: np.ndarray, min_length_px: float = 25.0, hough_threshold: int = 30,
                  max_gap_px: int = 10, merge_angle_deg: float = 2.0, merge_dist_px: float = 3.0,
                  band_px: float = 3.5, min_coverage: float = 0.6, max_lines: int = 24,
                  max_candidates: int = 150, bridge: float = 3.0,
                  weights: Optional[np.ndarray] = None, max_bend_px: Optional[float] = None,
                  min_bend_radius_px: float = float("inf"), peel_halo_px: float = 1.0) -> List[ImageLine]:
    """Hough candidates -> one sub-pixel line per marking, found by *peeling*.

    Probabilistic-Hough segments, longest first, are only seeds. Each seed is
    refitted to the mask pixels around it (PCA over a symmetric band, four
    passes), then those pixels are removed from the live mask, so the other
    segments Hough returns for the same thick marking find nothing left and no
    duplicate lines appear. Deciding on pixels instead of on clustering the noisy
    segments is what keeps nearby parallel markings (the far end of a wide shot)
    apart while a marking interrupted by a player still comes out as one line.

    The band must be symmetric and wider than the stroke: a narrow band truncates
    the stroke, which leaves the fitted angle free to wander inside the stroke
    (0.3 degrees over a 400 px goal line, measured). The stroke's half-width is
    estimated from the pixel count per unit length after each pass and the band
    follows it (``band_px`` is the minimum final half-width), so thin far-field
    lines and the thick near touchline are both fitted over their whole stroke.

    A fitted line's extent is trimmed to where white pixels actually are, and a
    line covered for less than ``min_coverage`` of that extent is a chord through
    other markings and is dropped, as are lines shorter than ``min_length_px``.
    Curved markings break into short chords and mostly fall below that length.
    ``weights`` (see :func:`line_weights`) are soft per-pixel whiteness values: the fit then
    uses the fringe pixels around the binary mask with their partial weights, which gives
    the stroke centre to a fraction of a pixel; without them every mask pixel counts 1.
    ``merge_angle_deg`` / ``merge_dist_px`` define a final duplicate test: two
    lines that overlap along their length, are within ``merge_angle_deg`` and whose
    endpoints are within ``merge_dist_px`` of each other's line are one marking.
    """
    import cv2

    h, w = mask.shape[:2]
    segs = cv2.HoughLinesP(mask, rho=1.0, theta=np.pi / 360.0, threshold=int(hough_threshold),
                           minLineLength=int(max(8, min_length_px * 0.4)), maxLineGap=int(max_gap_px))
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(float)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs = segs[np.argsort(-length)][:max_candidates]
    if weights is None:
        ys, xs = np.nonzero(mask)
        wgt = np.ones(len(ys), dtype=np.float32)
    else:
        ys, xs = np.nonzero(weights > 0.04)
        wgt = weights[ys, xs].astype(np.float32)
    if len(ys) == 0:
        return []
    pix = np.stack([xs, ys], axis=1).astype(float)
    alive = np.ones(len(pix), dtype=bool)
    rem = np.pad(mask, 2)  # live mask with a margin so 3x3 probes never leave the array
    thick = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    grow = float(max_gap_px) * bridge
    out: List[ImageLine] = []
    for seg in segs:
        a, b = seg[:2], seg[2:]
        d = b - a
        seg_len = float(np.hypot(d[0], d[1]))
        if seg_len < 4.0:
            continue
        # Quick duplicate test: a seed whose points are mostly on pixels that were
        # already taken by an earlier line is another view of that marking.
        probes = a + np.outer(np.linspace(0.15, 0.85, 5), d)
        px = np.round(probes).astype(int) + 2
        if sum(bool(rem[y - 1:y + 2, x - 1:x + 2].any()) for x, y in px) < 3:
            continue
        u = d / seg_len
        c = (a + b) / 2.0
        lo, hi = -seg_len / 2.0, seg_len / 2.0
        band = band_px + 1.5
        idx = None
        half_w = 0.0
        n_inl = 0
        for margin in (1.5, 1.0, 0.5, 0.0):
            sel = _band_select(pix, ys, alive, c, u, lo - grow, hi + grow, band)
            if len(sel) < 6:
                idx = None
                break
            pts = pix[sel]
            # keep the run with most pixels: bridge player-sized gaps, not whole other markings
            tt = (pts - c) @ u
            order = np.argsort(tt)
            gaps = np.nonzero(np.diff(tt[order]) > grow)[0]
            if len(gaps):
                starts, ends = np.r_[0, gaps + 1], np.r_[gaps + 1, len(tt)]
                k = int(np.argmax(ends - starts))
                sel = sel[order[starts[k]:ends[k]]]
                pts = pix[sel]
            fit = _fit_profile(pts, wgt[sel], c, u)
            if fit is None:  # too short for a profile: plain total least squares
                _abc, dv = _fit_line_pca(pts)
                if dv @ u < 0:
                    dv = -dv
                cc = pts.mean(axis=0)
                tt = (pts - cc) @ dv
                fit = (cc, dv, float(tt.min()), float(tt.max()), float("nan"), len(pts))
            c, u, lo, hi, thickness, n_inl = fit
            if np.isfinite(thickness):
                half_w = 0.5 * thickness
            band = max(band_px, half_w + 1.5 + margin)
            idx = sel
        if idx is None:
            continue
        nrm = np.array([-u[1], u[0]])
        abc = np.array([nrm[0], nrm[1], -(nrm @ c)])
        n = int(max(2, math.ceil(hi - lo)))
        samp = c + np.outer(np.linspace(lo, hi, n), u)
        sx = np.clip(np.round(samp[:, 0]).astype(int), 0, w - 1)
        sy = np.clip(np.round(samp[:, 1]).astype(int), 0, h - 1)
        hit = thick[sy, sx] > 0
        if hit.sum() < 2:
            continue
        hidx = np.nonzero(hit)[0]
        first, last = hidx[0], hidx[-1]
        coverage = float(hit[first:last + 1].mean())
        p0, p1 = samp[first], samp[last]
        if np.linalg.norm(p1 - p0) < min_length_px or coverage < min_coverage:
            continue
        # take this marking's pixels (stroke plus a halo) out of the live mask
        take = _band_select(pix, ys, alive, c, u, lo - 2.0, hi + 2.0, max(band_px, half_w + peel_halo_px))
        bend, _sig, _nb = stroke_bend(pix[take], wgt[take], c, u, lo, hi, half_w=max(half_w, 1.0))
        alive[take] = False
        rem[ys[take] + 2, xs[take] + 2] = 0
        half_len = 0.5 * (hi - lo)
        if max_bend_px is not None and bend > max_bend_px and half_len * half_len / (2.0 * bend) < min_bend_radius_px:
            continue  # a chord of the centre circle or a penalty arc: its pixels are taken, no line is kept
        out.append(ImageLine(p0=p0, p1=p1, abc=abc, n_pixels=int(n_inl), bend_px=bend,
                             half_width_px=float(half_w) if half_w > 0 else float("nan")))
    out.sort(key=lambda ln: -ln.length)
    tol_th = math.radians(merge_angle_deg)
    kept: List[ImageLine] = []
    for ln in out:
        dup = False
        for k in kept:
            dth = abs(ln.angle - k.angle)
            dth = min(dth, math.pi - dth)
            if dth >= tol_th or np.abs(ln.endpoints() @ k.abc[:2] + k.abc[2]).max() >= merge_dist_px:
                continue
            dk = (k.p1 - k.p0) / max(k.length, 1e-9)
            t_ln = np.sort((ln.endpoints() - k.p0) @ dk)
            if min(t_ln[1], k.length) - max(t_ln[0], 0.0) > 0.0:  # they share part of their extent
                dup = True
                break
        if not dup:
            kept.append(ln)
    return kept[:max_lines]


# ----------------------------------------------------------------------------
# 2b. Curved markings: the centre circle and the penalty arcs as image ellipses
# ----------------------------------------------------------------------------


@dataclass
class ImageEllipse:
    """A curved marking in the image: the conic fitted to its pixels, and points on it where it is drawn."""

    conic: np.ndarray  # 3x3 symmetric, x^T C x = 0 for homogeneous pixels, unit Frobenius norm
    centre: np.ndarray
    axes: np.ndarray  # semi-axes (along angle_deg, across), pixels
    angle_deg: float
    points: np.ndarray  # (K, 2) points of the fitted ellipse in the parts its pixels cover
    span_deg: float  # how much of the ellipse its pixels cover, in its own parametric angle
    rms_px: float  # RMS Sampson distance of its pixels
    n_pixels: int

    @property
    def arc_length(self) -> float:
        """Length of the covered part (pixels), from the sample points."""
        if len(self.points) < 2:
            return 0.0
        d = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        return float(d[d < 4.0 * np.median(d)].sum())


def ellipse_conic(centre: Sequence[float], axes: Sequence[float], angle_deg: float) -> np.ndarray:
    """3x3 conic of the ellipse with ``centre``, semi-axes ``axes`` (along ``angle_deg``, across), unit norm."""
    th = math.radians(angle_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    m = rot @ np.diag([1.0 / axes[0] ** 2, 1.0 / axes[1] ** 2]) @ rot.T
    c = np.asarray(centre, float)
    conic = np.zeros((3, 3))
    conic[:2, :2] = m
    conic[:2, 2] = conic[2, :2] = -m @ c
    conic[2, 2] = c @ m @ c - 1.0
    return conic / np.linalg.norm(conic)


def circle_conic(centre: Sequence[float], radius: float) -> np.ndarray:
    """Conic of a circle in the pitch frame (metres)."""
    cx, cy = centre
    return np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [-cx, -cy, cx * cx + cy * cy - radius * radius]])


def conic_distance(conic: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Sampson distance (first-order geometric distance, pixels) of (N, 2) points to a conic."""
    p = np.concatenate([np.asarray(pts, float).reshape(-1, 2), np.ones((len(pts), 1))], axis=1)
    cp = p @ conic  # conic is symmetric
    num = np.abs((cp * p).sum(axis=1))
    den = 2.0 * np.hypot(cp[:, 0], cp[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.inf)


def _ellipse_param(centre: np.ndarray, axes: np.ndarray, angle_deg: float, pts: np.ndarray) -> np.ndarray:
    """Parametric angle (radians, [-pi, pi)) of points around an ellipse."""
    th = math.radians(angle_deg)
    d = pts - centre
    xr = d[:, 0] * math.cos(th) + d[:, 1] * math.sin(th)
    yr = -d[:, 0] * math.sin(th) + d[:, 1] * math.cos(th)
    return np.arctan2(yr / axes[1], xr / axes[0])


def _refine_ellipse(pts: np.ndarray, centre: np.ndarray, axes: np.ndarray, angle_deg: float,
                    max_points: int = 1500) -> Tuple[np.ndarray, np.ndarray, float]:
    """Geometric refit of an ellipse: least squares on the Sampson distances of ``pts``.

    OpenCV's direct fit minimises an algebraic error, which pulls the ellipse inward on a thick stroke
    or a partial arc; minimising the (first-order) geometric distance puts it on the stroke's centre
    line. Returns the input unchanged when the refit fails.
    """
    from scipy.optimize import least_squares

    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
    p0 = np.array([centre[0], centre[1], math.log(axes[0]), math.log(axes[1]), math.radians(angle_deg)])

    def resid(p):
        conic = ellipse_conic(p[:2], np.exp(np.clip(p[2:4], -5.0, 12.0)), math.degrees(p[4]))
        hp = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        cp = hp @ conic
        return (cp * hp).sum(axis=1) / (2.0 * np.hypot(cp[:, 0], cp[:, 1]) + 1e-12)

    try:
        sol = least_squares(resid, p0, method="trf", loss="soft_l1", f_scale=2.0, max_nfev=50)
        p = sol.x
        if np.isfinite(p).all() and (np.abs(p[2:4] - p0[2:4]) < 1.0).all():  # within e^1 of the direct fit
            return p[:2].copy(), np.exp(p[2:4]), math.degrees(p[4])
    except (ValueError, np.linalg.LinAlgError):
        pass
    return centre, axes, angle_deg


def detect_ellipses(mask: np.ndarray, lines: Sequence[ImageLine] = (), max_ellipses: int = 3, min_pixels: int = 60,
                    min_span_deg: float = 90.0, max_rms_px: float = 1.5, straight_len_px: float = 120.0,
                    straight_bend_px: float = 0.5, long_len_px: Optional[float] = None, n_points: int = 72,
                    min_axis_ratio: float = 0.08) -> List[ImageEllipse]:
    """Ellipses through the curved marking pixels: the centre circle and the penalty arcs as the camera sees them.

    The confidently straight markings are erased first, which cuts the centre circle free of the halfway
    line it crosses: ``lines`` at least ``straight_len_px`` long that bow by at most ``straight_bend_px``,
    and every line of at least ``long_len_px`` (default a quarter of the frame width; lens distortion
    bows a long line by a pixel or more, a circle never yields a straight piece that long). Every remaining connected piece of at least ``min_pixels`` pixels then seeds an ellipse
    (OpenCV's direct least-squares fit), which is grown over the remaining pixels within a Sampson
    distance of about the stroke half-width plus two pixels and refitted, three times: a circle cut
    into pieces by the halfway line and by players comes back as one ellipse. An ellipse is kept when
    its pixels cover at least ``min_span_deg`` of it (in its own parametric angle; a short piece of arc
    does not fix a conic), fit it within ``max_rms_px`` or 0.6 times the stroke half-width plus half a pixel,
    whichever is larger (a filled stroke of half-width ``w`` alone has an RMS of ``w / sqrt(3)``: the
    near side of a centre circle measured 1.7 px), and it is not degenerate (major semi-axis at least
    40 px, minor at least 6 px and ``min_axis_ratio`` of the major: thinner ellipses are fits to a
    straight piece;
    centre within a frame size of the image). Its pixels are then taken out of the pool
    and the next seed is tried, up to ``max_ellipses``. Each ellipse carries ``n_points`` points
    on the fitted curve, spread over the parts its pixels cover, for the homography search and fit.
    """
    import cv2

    h, w = mask.shape[:2]
    if long_len_px is None:
        long_len_px = 0.25 * w
    res = mask.copy()
    hws = [ln.half_width_px for ln in lines if np.isfinite(ln.half_width_px)]
    half_w = float(np.median(hws)) if hws else 1.5
    for ln in lines:
        if ln.length >= long_len_px or (ln.length >= straight_len_px and ln.bend_px <= straight_bend_px):
            hw = ln.half_width_px if np.isfinite(ln.half_width_px) else half_w
            cv2.line(res, tuple(int(round(v)) for v in ln.p0), tuple(int(round(v)) for v in ln.p1), 0,
                     thickness=int(2 * math.ceil(hw) + 3))
    ys, xs = np.nonzero(res)
    if len(ys) < min_pixels:
        return []
    pool = np.stack([xs, ys], axis=1).astype(float)
    free = np.ones(len(pool), dtype=bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(res, np.ones((3, 3), np.uint8)), connectivity=8)
    comp = lab[ys, xs]
    order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])
    tol = 2.0 + half_w
    out: List[ImageEllipse] = []
    for k in order:
        if len(out) >= max_ellipses:
            break
        seed = free & (comp == k)
        if seed.sum() < min_pixels:
            continue
        if max(stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]) < 25:
            continue
        sel = seed
        ell = None
        for _ in range(3):
            if sel.sum() < 6:
                ell = None
                break
            (cx, cy), (aw, ah), ang = cv2.fitEllipseDirect(pool[sel].astype(np.float32))
            if not (np.isfinite([cx, cy, aw, ah, ang]).all() and aw > 2 and ah > 2):
                ell = None
                break
            ell = (np.array([cx, cy]), np.array([aw / 2.0, ah / 2.0]), float(ang))
            d = conic_distance(ellipse_conic(*ell), pool)
            sel = free & (d <= tol)
        if ell is None or sel.sum() < min_pixels:
            continue
        centre, axes, ang = _refine_ellipse(pool[sel], *ell)
        if axes[1] > axes[0]:  # major axis first
            axes, ang = axes[::-1].copy(), ang + 90.0
        if (axes[1] / axes[0] < min_axis_ratio or axes[1] < 6.0 or axes[0] < 40.0 or axes[0] > 3.0 * w
                or not (-w <= centre[0] <= 2 * w) or not (-h <= centre[1] <= 2 * h)):
            continue
        conic = ellipse_conic(centre, axes, ang)
        d = conic_distance(conic, pool[sel])
        rms = float(np.sqrt(np.mean(d ** 2)))
        phi = _ellipse_param(centre, axes, ang, pool[sel])
        bins = np.bincount(np.floor((phi + math.pi) / (2 * math.pi) * 36).astype(int).clip(0, 35), minlength=36)
        covered = bins >= 3
        span = 10.0 * covered.sum()
        if span < min_span_deg or rms > max(max_rms_px, 0.6 * half_w + 0.5):  # a filled stroke alone gives half_w / sqrt(3)
            continue
        t = np.linspace(-math.pi, math.pi, n_points, endpoint=False) + math.pi / n_points
        keep = covered[np.floor((t + math.pi) / (2 * math.pi) * 36).astype(int).clip(0, 35)]
        th = math.radians(ang)
        ex = axes[0] * np.cos(t[keep]); ey = axes[1] * np.sin(t[keep])
        pts = np.stack([centre[0] + ex * math.cos(th) - ey * math.sin(th),
                        centre[1] + ex * math.sin(th) + ey * math.cos(th)], axis=1)
        out.append(ImageEllipse(conic=conic, centre=centre, axes=axes, angle_deg=ang, points=pts, span_deg=span,
                                rms_px=rms, n_pixels=int(sel.sum())))
        free &= ~sel
    return out


def drop_chords(lines: Sequence[ImageLine], ellipses: Sequence[ImageEllipse], tol_px: float = 2.5,
                max_len_px: float = float("inf")) -> List[ImageLine]:
    """The lines that do not lie along one of the ``ellipses`` (a straight piece cut out of a curve is not a
    line); lines of at least ``max_len_px`` are kept whatever they touch."""
    if not ellipses:
        return list(lines)
    out = []
    for ln in lines:
        if ln.length >= max_len_px:
            out.append(ln)
            continue
        s = ln.p0 + np.outer(np.linspace(0.0, 1.0, 9), ln.p1 - ln.p0)
        hw = ln.half_width_px if np.isfinite(ln.half_width_px) else 1.5
        if any(float(conic_distance(e.conic, s).max()) <= tol_px + hw for e in ellipses):
            continue
        out.append(ln)
    return out


# ----------------------------------------------------------------------------
# 3. Hypothesis search
# ----------------------------------------------------------------------------


def _score_hypotheses(h_m2px: np.ndarray, lines: Sequence[ImageLine], model_abc: np.ndarray,
                      intervals: np.ndarray, model_kind: np.ndarray, tau_px: float, tol_m: float,
                      min_extent: float = 0.7, half_width: float = 34.0,
                      image_centre: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score (B,) hypotheses on all image lines.

    Returns (score, best_model_index (B, N), inlier mask (B, N)). An image line is
    an inlier of its nearest projected model line when both endpoints are within
    ``tau_px`` and the back-projected segment lies on a drawn part of the line.
    """
    B = h_m2px.shape[0]
    N = len(lines)
    ends = np.stack([ln.endpoints() for ln in lines], axis=0)  # (N, 2, 2)
    lengths = np.array([ln.length for ln in lines])
    h_m2px = np.asarray(h_m2px, float)
    finite = np.isfinite(h_m2px).all(axis=(1, 2))
    h_m2px = np.where(finite[:, None, None], h_m2px, np.eye(3)[None])
    det = np.linalg.det(h_m2px)
    regular = finite & (np.abs(det) > 1e-30 * np.maximum(np.abs(h_m2px).max(axis=(1, 2)) ** 3, 1e-300))
    h_m2px = np.where(regular[:, None, None], h_m2px, np.eye(3)[None])
    h_px2m = np.linalg.inv(h_m2px)
    # A homography is defined up to scale *and sign*, but the checks below look at the
    # sign of the homogeneous coordinate. Fix the sign so that the image centre, which is
    # always in front of the camera, has w > 0 under the pixel -> metre map (hence also
    # the metre -> pixel map is positive at its pre-image). Every check is then
    # independent of how the caller normalised H (h33 = 1, unit norm, an inverse, ...).
    centre = np.array([image_centre[0], image_centre[1], 1.0])
    w_c = h_px2m @ centre  # (B, 3)
    flip = np.where(w_c[:, 2] < 0.0, -1.0, 1.0)
    h_px2m = h_px2m * flip[:, None, None]
    h_m2px = h_m2px * flip[:, None, None]
    # Camera-side prior: the markings are symmetric under a 180-degree turn of the
    # pitch, so lines alone cannot tell the near touchline from the far one. The
    # GSR frame puts the camera on the +y side, so (0, +W/2) must project lower in
    # the image (larger v) than (0, -W/2), and both must be in front of the camera.
    probe = np.array([[0.0, -half_width, 1.0], [0.0, half_width, 1.0]])  # (2, 3)
    pr = np.einsum("bij,kj->bki", h_m2px, probe)  # (B, 2, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = pr[..., 1] / pr[..., 2]
    orient_ok = (pr[..., 2] > 0).all(axis=1) & (v[:, 1] > v[:, 0])
    # Handedness: pixels (x right, y down) and the pitch frame (x right, y toward
    # the camera) have the same orientation, so the Jacobian of the pixel -> metre
    # map must have positive determinant at the image centre, which is always in
    # front of the camera. For a homography det J(p) = det(H) / w(p)^3.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        jac_sign = np.linalg.det(h_px2m) / (w_c[:, 2] * flip) ** 3
    orient_ok &= np.isfinite(jac_sign) & (jac_sign > 0)
    # projected model lines in the image: (B, M, 3)
    l_px = _normalise_lines(np.einsum("mk,bkj->bmj", model_abc, h_px2m))
    ends_h = np.concatenate([ends, np.ones((N, 2, 1))], axis=-1)  # (N, 2, 3)
    dist = np.abs(np.einsum("nek,bmk->bnme", ends_h, l_px))  # (B, N, M, 2)
    dist = dist.max(axis=-1)  # worst endpoint
    best = np.argmin(dist, axis=2)  # (B, N)
    best_d = np.take_along_axis(dist, best[..., None], axis=2)[..., 0]
    # back-project endpoints to metres: (B, N, 2, 3)
    bp = np.einsum("nek,bjk->bnej", ends_h, h_px2m)
    w = bp[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = bp[..., :2] / w[..., None]
    kind = model_kind[best]  # (B, N): True where x_const (free coordinate is y)
    t = np.where(kind[..., None], xy[..., 1], xy[..., 0])  # (B, N, 2)
    t_lo, t_hi = t.min(axis=-1), t.max(axis=-1)
    iv = intervals[best]  # (B, N, 2, 2)
    frac = extent_fraction(t_lo, t_hi, iv, tol_m)
    same_sign = (np.sign(w[..., 0]) == np.sign(w[..., 1])) & np.isfinite(xy).all(axis=(-1, -2))
    inlier = (best_d <= tau_px) & (frac >= min_extent) & same_sign
    score = (inlier * lengths[None, :] * frac).sum(axis=1)
    score = np.where(regular & orient_ok, score, -1.0)
    return score, best, inlier


def _distinct_lines(lines: Sequence[ImageLine], k: int, width: int) -> List[int]:
    """Indices of the ``k`` longest lines that lie on different infinite lines.

    Two drawn segments of one model line (both penalty-area sides on the near side, say,
    or a touchline split by a player) are collinear in the image and add no constraint
    to a hypothesis, so a second one would only waste a candidate slot.
    """
    tol_px = max(2.0, 0.002 * width)
    tol_ang = math.radians(1.5)
    chosen: List[int] = []
    for i, ln in enumerate(lines):  # sorted by length
        dup = False
        for j in chosen:
            d_ang = abs(ln.angle - lines[j].angle)
            d_ang = min(d_ang, math.pi - d_ang)
            if d_ang < tol_ang and np.abs(ln.endpoints() @ lines[j].abc[:2] + lines[j].abc[2]).max() < tol_px:
                dup = True
                break
        if not dup:
            chosen.append(i)
            if len(chosen) == k:
                break
    return chosen


def _camera_ok_batch(h_m2px: np.ndarray, width: int, height: int, focal_range: Tuple[float, float],
                     camera_box) -> np.ndarray:
    """(B,) whether each metre -> pixel homography decomposes into a camera (as in
    :func:`camera_from_homography`) with a focal length in ``focal_range`` frame widths and its centre
    (x, y, height) inside ``camera_box``."""
    t = np.array([[1.0, 0.0, -width / 2.0], [0.0, 1.0, -height / 2.0], [0.0, 0.0, 1.0]])
    hh = np.einsum("ij,bjk->bik", t, np.asarray(h_m2px, float))
    with np.errstate(all="ignore"):
        hh = hh / np.linalg.norm(hh, axis=(1, 2))[:, None, None]
        h1, h2, h3 = hh[:, :, 0], hh[:, :, 1], hh[:, :, 2]
        a0 = h1[:, 0] * h2[:, 0] + h1[:, 1] * h2[:, 1]
        a1 = h1[:, 0] ** 2 + h1[:, 1] ** 2 - h2[:, 0] ** 2 - h2[:, 1] ** 2
        b0 = -h1[:, 2] * h2[:, 2]
        b1 = -(h1[:, 2] ** 2 - h2[:, 2] ** 2)
        inv_f2 = (a0 * b0 + a1 * b1) / (a0 * a0 + a1 * a1)
        f = 1.0 / np.sqrt(inv_f2)
        ok = np.isfinite(f) & (inv_f2 > 0) & (f >= focal_range[0] * width) & (f <= focal_range[1] * width)
        k = np.stack([1.0 / f, 1.0 / f, np.ones_like(f)], axis=1)  # diagonal of K^-1
        r1, r2, tt = h1 * k, h2 * k, h3 * k
        lam = 2.0 / (np.linalg.norm(r1, axis=1) + np.linalg.norm(r2, axis=1))
        sgn = np.where(tt[:, 2] * lam < 0, -1.0, 1.0) * lam
        r1, r2, tt = r1 * sgn[:, None], r2 * sgn[:, None], tt * sgn[:, None]
        rot = np.stack([r1, r2, np.cross(r1, r2)], axis=2)  # columns r1, r2, r3
        centre = -np.einsum("bji,bj->bi", rot, tt)
        cam = np.stack([centre[:, 0], centre[:, 1], -centre[:, 2]], axis=1)
        for i, (lo, hi) in enumerate(camera_box):
            ok &= (cam[:, i] >= lo) & (cam[:, i] <= hi)
    return ok & np.isfinite(cam).all(axis=1)


@dataclass(frozen=True)
class _ModelCircle:
    """A circle of the pitch model and the part of it that is drawn (``x_side`` 0: all of it; -1 / +1: only
    where ``x <= x_limit`` / ``x >= x_limit``, the penalty arcs outside the penalty area)."""

    name: str
    centre: Tuple[float, float]
    radius: float
    x_side: int = 0
    x_limit: float = 0.0


def _model_circles(model: PitchModel) -> List[_ModelCircle]:
    L, r = model.half_length, model.circle_radius
    return [_ModelCircle("centre_circle", (0.0, 0.0), r),
            _ModelCircle("penalty_arc_left", (-L + model.penalty_mark, 0.0), r, +1, -L + model.penalty_depth),
            _ModelCircle("penalty_arc_right", (L - model.penalty_mark, 0.0), r, -1, L - model.penalty_depth)]


@dataclass
class ArcMatch:
    ellipse_index: int
    model_name: str
    distance_px: float  # median Sampson distance of the ellipse's points to the projected model circle
    fraction: float  # share of its points within the tolerance, on the drawn part


def _score_arcs(h_m2px: np.ndarray, ellipses: Sequence[ImageEllipse], model: PitchModel, tau_px: float,
                tol_m: float = 1.0, min_fraction: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score (B,) metre -> pixel homographies on the image ellipses.

    An ellipse is an inlier of a model circle when at least ``min_fraction`` of its points lie within
    ``tau_px`` (Sampson distance) of the projected circle and back-project onto its drawn part (for a
    penalty arc, outside the penalty area, ``tol_m`` of slack). Returns (score = sum of the inlier
    ellipses' covered arc length times their fraction, best circle index (B, E), inlier (B, E),
    median distance (B, E)).
    """
    B, E = len(h_m2px), len(ellipses)
    if E == 0:
        z = np.zeros((B, 0))
        return np.zeros(B), z.astype(int), z.astype(bool), z
    circles = _model_circles(model)
    cm = np.stack([circle_conic(c.centre, c.radius) for c in circles], axis=0)  # (A, 3, 3)
    hm = np.asarray(h_m2px, float)
    finite = np.isfinite(hm).all(axis=(1, 2))
    hm = np.where(finite[:, None, None], hm, np.eye(3)[None])
    with np.errstate(all="ignore"):
        try:
            hp = np.linalg.inv(hm)
        except np.linalg.LinAlgError:
            hp = np.stack([np.linalg.pinv(x) for x in hm])
        # image conics of the model circles: H_px2m^T C_m H_px2m, (B, A, 3, 3)
        ci = np.einsum("bji,ajk,bkl->bail", hp, cm, hp)
    score = np.zeros(B)
    best = np.zeros((B, E), dtype=int)
    inl = np.zeros((B, E), dtype=bool)
    med = np.full((B, E), np.inf)
    for e, ell in enumerate(ellipses):
        p = np.concatenate([ell.points, np.ones((len(ell.points), 1))], axis=1)  # (K, 3)
        with np.errstate(all="ignore"):
            cp = np.einsum("bajk,nk->banj", ci, p)  # (B, A, K, 3)
            num = np.abs((cp * p[None, None]).sum(-1))
            den = 2.0 * np.hypot(cp[..., 0], cp[..., 1])
            d = np.where(den > 0, num / den, np.inf)  # (B, A, K)
            q = np.einsum("bij,nj->bni", hp, p)  # back-projected points (B, K, 3)
            xm = q[..., 0] / q[..., 2]
        drawn = np.ones((B, len(circles), len(p)), dtype=bool)
        for a, c in enumerate(circles):
            if c.x_side > 0:
                drawn[:, a] = xm >= c.x_limit - tol_m
            elif c.x_side < 0:
                drawn[:, a] = xm <= c.x_limit + tol_m
        ok = (d <= tau_px) & drawn
        frac = ok.mean(axis=-1)  # (B, A)
        a_best = np.argmax(frac, axis=1)
        f_best = np.take_along_axis(frac, a_best[:, None], axis=1)[:, 0]
        best[:, e] = a_best
        med[:, e] = np.median(np.take_along_axis(d, a_best[:, None, None], axis=1)[:, 0], axis=-1)
        inl[:, e] = finite & (f_best >= min_fraction)
        score += np.where(inl[:, e], ell.arc_length * f_best, 0.0)
    return score, best, inl, med


def match_arcs(h_m2px: np.ndarray, ellipses: Sequence[ImageEllipse], model: PitchModel = DEFAULT_PITCH,
               tau_px: float = 4.0) -> List[ArcMatch]:
    """Image ellipses that lie on a drawn model circle under ``h_m2px``; each model circle takes one (the best)."""
    if not ellipses:
        return []
    circles = _model_circles(model)
    _s, best, inl, med = _score_arcs(np.asarray(h_m2px, float)[None], ellipses, model, tau_px)
    out: Dict[str, ArcMatch] = {}
    for e in range(len(ellipses)):
        if not inl[0, e]:
            continue
        name = circles[int(best[0, e])].name
        m = ArcMatch(ellipse_index=e, model_name=name, distance_px=float(med[0, e]), fraction=1.0)
        if name not in out or ellipses[e].arc_length > ellipses[out[name].ellipse_index].arc_length:
            out[name] = m
    return sorted(out.values(), key=lambda m: m.ellipse_index)


def _dlt_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Batched DLT from homogeneous point pairs: ``src``, ``dst`` (B, N, 3), N >= 4; returns (B, 3, 3) with
    ``dst ~ H src`` (least squares on the algebraic error, unit-norm rows per pair; NaN where degenerate)."""
    src = src / np.linalg.norm(src, axis=-1, keepdims=True)
    dst = dst / np.linalg.norm(dst, axis=-1, keepdims=True)
    B, N, _ = src.shape
    z = np.zeros((B, N, 3))
    u, v, w = dst[..., 0:1], dst[..., 1:2], dst[..., 2:3]
    r1 = np.concatenate([z, -w * src, v * src], axis=-1)
    r2 = np.concatenate([w * src, z, -u * src], axis=-1)
    a = np.concatenate([r1, r2], axis=1)  # (B, 2N, 9)
    out = np.full((B, 3, 3), np.nan)
    good = np.isfinite(a).all(axis=(1, 2))
    if good.any():
        _u, s, vt = np.linalg.svd(a[good])
        out[good] = vt[:, -1].reshape(-1, 3, 3)
        cond = s[:, -2] / np.maximum(s[:, 0], 1e-300)  # a second small singular value: under-determined
        bad = np.nonzero(good)[0][cond < 1e-6]
        out[bad] = np.nan
    return out


def _conic_line_points(conic: np.ndarray, p0: np.ndarray, d: np.ndarray) -> Optional[Tuple[float, float]]:
    """Parameters ``t`` where the homogeneous line point ``p0 + t d`` lies on the conic (real roots only)."""
    a = d @ conic @ d
    b = 2.0 * (p0 @ conic @ d)
    c = p0 @ conic @ p0
    if abs(a) < 1e-300:
        return None
    disc = b * b - 4 * a * c
    if disc <= 0:
        return None
    sq = math.sqrt(disc)
    return (-b - sq) / (2 * a), (-b + sq) / (2 * a)


def _circle_hypotheses(lines: Sequence[ImageLine], cand: Sequence[int], ellipses: Sequence[ImageEllipse],
                       model: PitchModel, max_vp_angle_deg: float = 2.0,
                       min_span_deg: float = 150.0) -> Tuple[np.ndarray, np.ndarray]:
    """Correspondences for homographies from the centre circle, the halfway line and one touchline.

    For an image ellipse ``E`` taken as the centre circle, an image line ``L`` taken as the halfway line
    and a line ``M`` taken as a touchline (``y = +-W/2``):

    * ``L`` meets ``E`` at the images of (0, -r) and (0, +r);
    * the pole of ``L`` with respect to ``E`` is the vanishing point of the pitch's x direction (the
      pole of a diameter is the point at infinity perpendicular to it), so ``M`` must pass through it
      (within ``max_vp_angle_deg``), which rejects most wrong pairings before any solve;
    * ``L`` meets ``M`` at the image of (0, +-W/2); with the two circle points this fixes the
      projective map along ``L`` and with it the image of the centre mark (0, 0);
    * the line through that centre and the vanishing point is the image of ``y = 0``, which meets
      ``E`` at the images of (-r, 0) and (+r, 0).

    Returns six homogeneous correspondences per hypothesis, model (B, 6, 3) and image (B, 6, 3): four
    circle points, the touchline crossing and the vanishing point, for a DLT. Both orderings of each pair are generated; the scorer's orientation tests
    discard the mirrored ones. Only ellipses that cover at least ``min_span_deg`` are used: a short
    arc does not fix its conic well enough for the pole construction.
    """
    r = model.circle_radius
    W = model.half_width
    sin_max = math.sin(math.radians(max_vp_angle_deg))
    src_all, dst_all = [], []
    for ell in ellipses:
        if ell.span_deg < min_span_deg:
            continue
        C = ell.conic
        try:
            c_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            continue
        for i in cand:
            L = lines[i].abc
            d_l = np.array([-L[1], L[0], 0.0])  # direction of L as a homogeneous point at infinity
            base = np.array([*(lines[i].mid), 1.0])
            ts = _conic_line_points(C, base, d_l)
            if ts is None:
                continue
            P = [base + t * d_l for t in ts]
            V = c_inv @ L  # pole of L
            for j in cand:
                if j == i:
                    continue
                M = lines[j]
                # M passes through V: the direction from M's midpoint to V is along M
                dv = V[:2] - M.mid * V[2]
                nv = float(np.linalg.norm(dv))
                um = (M.p1 - M.p0) / max(M.length, 1e-9)
                if nv < 1e-12 or abs(um[0] * dv[1] - um[1] * dv[0]) / nv > sin_max:
                    continue
                Q = np.cross(L, M.abc)
                if abs(Q[2]) < 1e-12:
                    continue
                tq = ((Q[:2] / Q[2] - base[:2]) @ d_l[:2])
                for sy in (-W, W):
                    for pa, pb in ((0, 1), (1, 0)):
                        # 1-D projective map along L from model y to line parameter t: t = (a y + b) / (c y + 1)
                        ys_ = np.array([-r, r, sy])
                        tt = np.array([ts[pa], ts[pb], tq])
                        a_mat = np.stack([ys_, np.ones(3), -tt * ys_], axis=1)
                        try:
                            a_, b_, c_ = np.linalg.solve(a_mat, tt)
                        except np.linalg.LinAlgError:
                            continue
                        c0 = base + b_ * d_l  # image of the centre mark (y = 0: t = b)
                        dy0 = np.cross(np.cross(c0, V), np.array([0.0, 0.0, 1.0]))  # direction of the line c0-V
                        if not np.isfinite(dy0).all() or np.linalg.norm(dy0[:2]) < 1e-12:
                            continue
                        ss = _conic_line_points(C, c0, np.array([dy0[0], dy0[1], 0.0]) * c0[2])
                        if ss is None:
                            continue
                        S = [c0 + s * np.array([dy0[0], dy0[1], 0.0]) * c0[2] for s in ss]
                        for sa, sb in ((0, 1), (1, 0)):
                            src_all.append([[0.0, -r, 1.0], [0.0, r, 1.0], [0.0, sy, 1.0], [1.0, 0.0, 0.0],
                                            [-r, 0.0, 1.0], [r, 0.0, 1.0]])
                            dst_all.append([P[pa], P[pb], Q, V, S[sa], S[sb]])
    if not src_all:
        return np.zeros((0, 6, 3)), np.zeros((0, 6, 3))
    return np.asarray(src_all, float), np.asarray(dst_all, float)


def search_homography(lines: Sequence[ImageLine], width: int, height: int, model: PitchModel = DEFAULT_PITCH,
                      tau_px: Optional[float] = None, tol_m: float = 2.0, top_k: int = 6, n_score: int = 10,
                      n_rescore: int = 64, chunk: int = 40000,
                      ellipses: Sequence[ImageEllipse] = (), ranked: Optional[List[Tuple[float, np.ndarray]]] = None,
                      camera_prior: Optional["VerifyConfig"] = None) -> Tuple[Optional[np.ndarray], List[LineMatch], float]:
    """Global search: assign four image lines to two x = const and two y = const
    model lines, solve the homography from their four intersections, score on all
    lines; returns ``(H_m2px, matches, score)`` or ``(None, [], 0)``.

    With ``ellipses`` (see :func:`detect_ellipses`), hypotheses from the centre circle, the halfway
    line and a touchline are added (:func:`_circle_hypotheses`; they need only two lines), and every
    hypothesis that reaches the final scoring is also scored on the ellipses (:func:`_score_arcs`).
    A list passed as ``ranked`` receives every positively scored hypothesis of that final scoring as
    ``(score, H_m2px)``, best first. With ``camera_prior`` (a :class:`VerifyConfig`), a hypothesis no
    camera in its ``focal_range`` and ``camera_box`` could produce scores nothing: verification would
    refuse it anyway, and on broadcast frames such hypotheses often outscore the right one.
    """
    if len(lines) < 4 and not (ellipses and len(lines) >= 2):
        return None, [], 0.0
    tau = tau_px if tau_px is not None else max(2.5, 0.006 * width)
    model_abc, intervals, mlines = _segment_intervals(model)
    model_kind = np.array([ln.kind == X_CONST for ln in mlines])
    # Each family sorted by its constant, so every model pair below has xa < xb and
    # ya < yb: the model quad then has positive orientation, matching the positive image
    # quads kept by _quad_orientation_ok. (Unsorted, any pair listed in descending order,
    # such as the near touchline with a penalty-area side, could only ever be paired with
    # a mirrored image quad, so its correct homography was never generated.)
    xs = sorted(((i, ln.value) for i, ln in enumerate(mlines) if ln.kind == X_CONST), key=lambda t: t[1])
    ys = sorted(((i, ln.value) for i, ln in enumerate(mlines) if ln.kind == Y_CONST), key=lambda t: t[1])
    cand = _distinct_lines(lines, top_k, width)  # the longest lines, one per infinite line
    # image assignments: ordered pair A -> x_const, ordered pair B -> y_const (disjoint)
    img_assign = []
    for a in itertools.combinations(cand, 2):
        rest = [c for c in cand if c not in a]
        for b in itertools.combinations(rest, 2):
            for a_ord in (a, a[::-1]):
                for b_ord in (b, b[::-1]):
                    img_assign.append((a_ord, b_ord))
    model_assign = [(xa, xb, ya, yb) for xa, xb in itertools.combinations(xs, 2)
                    for ya, yb in itertools.combinations(ys, 2)]
    # intersections of image lines (A_i with B_j) give the four image points; the model
    # points are (x_a, y_a), (x_a, y_b), (x_b, y_a), (x_b, y_b).
    abcs = np.stack([ln.abc for ln in lines], axis=0)

    def intersect(i, j):
        p = np.cross(abcs[i], abcs[j])
        return p[:2] / p[2] if abs(p[2]) > 1e-12 else np.array([np.nan, np.nan])

    img_pts = []
    for (a1, a2), (b1, b2) in img_assign:
        img_pts.append([intersect(a1, b1), intersect(a1, b2), intersect(a2, b1), intersect(a2, b2)])
    img_pts = np.array(img_pts, float).reshape(-1, 4, 2)  # (IA, 4, 2); none with fewer than four lines
    mod_pts = np.array([[(xa[1], ya[1]), (xa[1], yb[1]), (xb[1], ya[1]), (xb[1], yb[1])]
                        for xa, xb, ya, yb in model_assign])  # (MA, 4, 2)
    IA, MA = len(img_assign), len(model_assign)
    ok_img = np.isfinite(img_pts).all(axis=(1, 2)) & _quad_orientation_ok(img_pts)
    # Real marking intersections lie within a few frame sizes of the image; a corner ten
    # frames away comes from two nearly parallel lines, a wrong pairing whose homography
    # is ill-conditioned anyway.
    with np.errstate(invalid="ignore"):
        ok_img &= (np.abs(img_pts - np.array([width / 2.0, height / 2.0]))
                   <= 10.0 * np.array([width, height], float)).all(axis=(1, 2))
    # normalisation for conditioning
    s_px = np.array([width, height], float)
    s_m = np.array([model.half_length, model.half_width], float)
    pairs = [(i, j) for i in range(IA) if ok_img[i] for j in range(MA)] if len(lines) >= 4 else []
    score_lines = list(lines[:n_score])
    leaders: List[Tuple[float, np.ndarray]] = []
    for start in range(0, len(pairs), chunk):
        sub = pairs[start:start + chunk]
        ii = np.array([p[0] for p in sub]); jj = np.array([p[1] for p in sub])
        src = mod_pts[jj] / s_m
        dst = img_pts[ii] / s_px
        hn = _dlt_batch(src, dst)
        # unnormalise: px = S_px ... H_n ... S_m^-1
        h = np.einsum("ij,bjk,kl->bil", np.diag([s_px[0], s_px[1], 1.0]), hn, np.diag([1 / s_m[0], 1 / s_m[1], 1.0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            h = h / h[:, 2:3, 2:3]
        score, _best, _inl = _score_hypotheses(h, score_lines, model_abc, intervals, model_kind, tau, tol_m,
                                               half_width=model.half_width, image_centre=(width / 2.0, height / 2.0))
        if camera_prior is not None:
            score = np.where(_camera_ok_batch(h, width, height, camera_prior.focal_range, camera_prior.camera_box),
                             score, -1.0)
        top = np.argsort(-score)[:n_rescore]
        leaders.extend((float(score[b]), h[b].copy()) for b in top if score[b] > 0)
    leaders.sort(key=lambda t: -t[0])
    hyps = [h for _, h in leaders[:n_rescore]]
    if ellipses:
        src, dst = _circle_hypotheses(lines, cand if len(cand) else _distinct_lines(lines, top_k, width), ellipses, model)
        if len(src):
            t_m = np.diag([1.0 / model.half_length, 1.0 / model.half_width, 1.0])
            t_px = np.array([[1.0 / width, 0.0, -0.5], [0.0, 1.0 / height, -0.5], [0.0, 0.0, 1.0]])
            hn = _dlt_points(src @ t_m.T, dst @ t_px.T)
            h = np.einsum("ij,bjk,kl->bil", np.linalg.inv(t_px), hn, t_m)
            with np.errstate(divide="ignore", invalid="ignore"):
                h = h / h[:, 2:3, 2:3]
            hyps.extend(h[np.isfinite(h).all(axis=(1, 2))])
    if not hyps:
        return None, [], 0.0
    hs = np.stack(hyps, axis=0)
    full, _b, _i = _score_hypotheses(hs, lines, model_abc, intervals, model_kind, tau, tol_m,
                                     half_width=model.half_width, image_centre=(width / 2.0, height / 2.0))
    if ellipses:
        arc_score, _ab, _ai, _am = _score_arcs(hs, ellipses, model, tau)
        full = np.where(full >= 0, full + arc_score, full)
    if camera_prior is not None:
        full = np.where(_camera_ok_batch(hs, width, height, camera_prior.focal_range, camera_prior.camera_box),
                        full, -1.0)
    b = int(np.argmax(full))
    if ranked is not None:
        ranked.extend((float(full[i]), hs[i]) for i in np.argsort(-full) if full[i] > 0)
    if full[b] <= 0:
        return None, [], 0.0
    best_h = hs[b]
    matches = match_lines(best_h, lines, model, tau_px=tau, tol_m=tol_m)
    return best_h, matches, float(full[b])


def match_lines(h_m2px: np.ndarray, lines: Sequence[ImageLine], model: PitchModel = DEFAULT_PITCH,
                tau_px: float = 4.0, tol_m: float = 2.0, min_extent: float = 0.7,
                image_centre: Optional[Tuple[float, float]] = None) -> List[LineMatch]:
    """Image lines that lie on a drawn model segment under ``h_m2px`` (one model line
    may take several image segments; each image line takes at most one model line)."""
    if len(lines) == 0:
        return []
    model_abc, intervals, mlines = _segment_intervals(model)
    model_kind = np.array([ln.kind == X_CONST for ln in mlines])
    if image_centre is None:  # the mean of the line midpoints is always in front of the camera
        image_centre = tuple(np.mean([ln.mid for ln in lines], axis=0))
    score, best, inl = _score_hypotheses(h_m2px[None], lines, model_abc, intervals, model_kind, tau_px, tol_m,
                                         min_extent=min_extent, half_width=model.half_width,
                                         image_centre=image_centre)
    if score[0] <= 0:
        return []
    out: List[LineMatch] = []
    h_px2m = np.linalg.inv(h_m2px)
    l_px = project_model_lines(h_px2m, model_abc)
    for n, ln in enumerate(lines):
        if not inl[0, n]:
            continue
        m = int(best[0, n])
        e = np.concatenate([ln.endpoints(), np.ones((2, 1))], axis=1)
        d = float(np.abs(e @ l_px[m]).max())
        xy = apply_homography(h_px2m, ln.endpoints())
        t = xy[:, 1] if model_kind[m] else xy[:, 0]
        fr = float(extent_fraction(np.array(t.min()), np.array(t.max()), intervals[m][None], tol_m)[0])
        out.append(LineMatch(image_index=n, model_name=mlines[m].name, distance_px=d, extent_fraction=fr))
    return out


# ----------------------------------------------------------------------------
# 4. Refinement
# ----------------------------------------------------------------------------


def _fit_residuals(lines: Sequence[ImageLine], matches: Sequence[LineMatch], model: PitchModel,
                   ellipses: Sequence[ImageEllipse] = (), arc_matches: Sequence[ArcMatch] = (),
                   arc_weight: float = 1.0):
    """The weighted residual function of a metre -> pixel homography that the refinement minimises (see
    :func:`refine_homography`), or None when nothing is matched."""
    if len(matches) == 0 and len(arc_matches) == 0:
        return None
    model_lines = {ln.name: ln for ln in model.lines()}
    circles = {c.name: c for c in _model_circles(model)}
    pts, mabc, wts = [], [], []
    for m in matches:
        ln = lines[m.image_index]
        ml = model_lines[m.model_name].homogeneous
        for p in (ln.p0, ln.mid, ln.p1):
            pts.append(p); mabc.append(ml); wts.append(math.sqrt(max(ln.length, 1.0)))
    arc_pts, arc_cm, arc_w = [], [], []
    for am in arc_matches:
        ell = ellipses[am.ellipse_index]
        c = circles[am.model_name]
        k = len(ell.points)
        arc_pts.append(ell.points)
        arc_cm.extend([circle_conic(c.centre, c.radius)] * k)
        arc_w.extend([arc_weight * math.sqrt(3.0 * max(ell.arc_length, 1.0) / max(k, 1))] * k)
    pts = np.concatenate([np.array(pts).reshape(-1, 2), np.ones((len(pts), 1))], axis=1)
    mabc = np.array(mabc).reshape(-1, 3)
    apts = np.concatenate(arc_pts, axis=0) if arc_pts else np.zeros((0, 2))
    apts = np.concatenate([apts, np.ones((len(apts), 1))], axis=1)
    acm = np.array(arc_cm).reshape(-1, 3, 3)
    all_w = np.array(wts + arc_w)
    all_w = all_w / np.sqrt(np.mean(all_w ** 2))
    wts, arc_wts = all_w[:len(wts)], all_w[len(wts):]

    def residuals_h(h):
        with np.errstate(divide="ignore", invalid="ignore"):
            h_inv = np.linalg.inv(h)
        l_px = _normalise_lines(mabc @ h_inv)
        r = wts * (pts * l_px).sum(axis=1)
        if len(apts):
            with np.errstate(divide="ignore", invalid="ignore"):
                ci = np.einsum("ji,njk,kl->nil", h_inv, acm, h_inv)  # image conic per point
                cp = np.einsum("nij,nj->ni", ci, apts)
                d = (cp * apts).sum(axis=1) / (2.0 * np.hypot(cp[:, 0], cp[:, 1]))
            r = np.concatenate([r, arc_wts * np.where(np.isfinite(d), d, 1e3)])
        return r

    return residuals_h


def _h_param(h_m2px: np.ndarray, width: int, height: int, model: PitchModel, camera: bool):
    """``(theta0, to_h)``: a parametrisation of a metre -> pixel homography (the camera's seven parameters,
    or the eight free entries of the normalised matrix) and the map back; None if ``camera`` and there
    is no camera."""
    if camera:
        q0 = _camera_params(h_m2px, width, height)
        if q0 is None:
            return None
        return q0, lambda q: _camera_h(q, width, height)
    s_px = np.diag([width, height, 1.0])
    s_m = np.diag([model.half_length, model.half_width, 1.0])
    s_m_inv = np.linalg.inv(s_m)
    hn0 = np.linalg.inv(s_px) @ h_m2px @ s_m  # normalised parametrisation
    hn0 = hn0 / hn0[2, 2]
    return hn0.reshape(-1)[:8], lambda p: s_px @ np.append(p, 1.0).reshape(3, 3) @ s_m_inv


def refine_homography(h_m2px: np.ndarray, lines: Sequence[ImageLine], matches: Sequence[LineMatch],
                      model: PitchModel = DEFAULT_PITCH, width: int = 1920, height: int = 1080,
                      ellipses: Sequence[ImageEllipse] = (), arc_matches: Sequence[ArcMatch] = (),
                      arc_weight: float = 1.0, camera: bool = False) -> np.ndarray:
    """Levenberg-Marquardt on the 8 free parameters: minimise the pixel distances of
    the matched segments' endpoints and midpoints to the projected model lines and, for matched
    ellipses, the Sampson distances of their points to the projected model circles (each ellipse
    weighted like a line of its covered arc length).

    With ``camera`` the homography is refined as a pinhole camera instead (square pixels, principal
    point at the image centre: focal length, rotation and position, seven parameters), which a real
    broadcast camera is; a homography that does not decompose into such a camera is returned unrefined."""
    from scipy.optimize import least_squares

    residuals_h = _fit_residuals(lines, matches, model, ellipses, arc_matches, arc_weight)
    if residuals_h is None:
        return h_m2px
    par = _h_param(h_m2px, width, height, model, camera)
    if par is None:
        return h_m2px
    theta0, to_h = par
    try:
        sol = least_squares(lambda t: residuals_h(to_h(t)), theta0, method="lm", x_scale="jac" if camera else 1.0,
                            xtol=1e-12, ftol=1e-12, max_nfev=400)
        h = to_h(sol.x)
        if np.isfinite(h).all():
            return h / h[2, 2]
    except (ValueError, np.linalg.LinAlgError):
        pass
    return h_m2px


def fit_uncertainty_m(h_m2px: np.ndarray, residuals_h, pts_px: np.ndarray, width: int, height: int,
                      model: PitchModel = DEFAULT_PITCH, camera: bool = False, min_sigma_px: float = 0.5) -> np.ndarray:
    """Standard error (metres) of the pitch position the fit gives each of ``pts_px``.

    The parameter covariance is ``s^2 (J^T J)^-1`` at the solution (``J`` the Jacobian of the weighted
    residuals, ``s`` their RMS with ``n - p`` degrees of freedom, at least ``min_sigma_px``), carried to
    each point's back-projection with that point's own Jacobian. It is large where the markings that
    fixed the fit say little: far from them, or along the one direction a near-degenerate set leaves
    free. Infinite when the fit has no redundancy.
    """
    pts_px = np.asarray(pts_px, float).reshape(-1, 2)
    par = _h_param(h_m2px, width, height, model, camera)
    if par is None or residuals_h is None or len(pts_px) == 0:
        return np.full(len(pts_px), np.inf)
    theta, to_h = par
    r0 = residuals_h(to_h(theta))
    n, k = len(r0), len(theta)
    if n <= k or not np.isfinite(r0).all():
        return np.full(len(pts_px), np.inf)
    steps = 1e-6 * np.maximum(np.abs(theta), 1e-3)

    def back(t):
        return apply_homography(np.linalg.inv(to_h(t)), pts_px).ravel()

    m0 = back(theta)
    jr = np.empty((n, k))
    jm = np.empty((len(m0), k))
    for i in range(k):
        t = theta.copy()
        t[i] += steps[i]
        jr[:, i] = (residuals_h(to_h(t)) - r0) / steps[i]
        jm[:, i] = (back(t) - m0) / steps[i]
    s2 = max(float(r0 @ r0) / (n - k), min_sigma_px ** 2)
    try:
        cov = s2 * np.linalg.inv(jr.T @ jr)
    except np.linalg.LinAlgError:
        return np.full(len(pts_px), np.inf)
    var = np.einsum("ij,jk,ik->i", jm, cov, jm).reshape(-1, 2).sum(axis=1)
    return np.sqrt(np.clip(var, 0.0, None))


def _camera_params(h_m2px: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    """``(f, rx, ry, rz, cx, cy, cz)``: focal length, Rodrigues rotation (pitch -> camera) and camera centre
    (pitch frame, ``z`` into the ground) of the pinhole camera closest to ``h_m2px``; None if there is none."""
    import cv2

    t = np.array([[1.0, 0.0, -width / 2.0], [0.0, 1.0, -height / 2.0], [0.0, 0.0, 1.0]])
    hh = t @ np.asarray(h_m2px, float)
    f, _inc, _c = camera_from_homography(h_m2px, width, height)
    if not np.isfinite(f):
        return None
    m = np.diag([1.0 / f, 1.0 / f, 1.0]) @ hh
    lam = 2.0 / (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1]))
    m = m * lam
    if m[2, 2] < 0:  # pitch in front of the camera
        m = -m
    r = np.stack([m[:, 0], m[:, 1], np.cross(m[:, 0], m[:, 1])], axis=1)
    u, _s, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        return None
    rvec, _ = cv2.Rodrigues(r)
    centre = -r.T @ m[:, 2]
    return np.concatenate([[f], rvec.ravel(), centre])


def _camera_h(q: np.ndarray, width: int, height: int) -> np.ndarray:
    """Metre -> pixel homography of the camera ``q`` (see :func:`_camera_params`)."""
    import cv2

    r, _ = cv2.Rodrigues(np.asarray(q[1:4], float).reshape(3, 1))
    k = np.array([[q[0], 0.0, width / 2.0], [0.0, q[0], height / 2.0], [0.0, 0.0, 1.0]])
    return k @ np.stack([r[:, 0], r[:, 1], -r @ np.asarray(q[4:7], float)], axis=1)


def _errors(h_m2px: np.ndarray, lines: Sequence[ImageLine], matches: Sequence[LineMatch],
            model: PitchModel, ellipses: Sequence[ImageEllipse] = (),
            arc_matches: Sequence[ArcMatch] = ()) -> Tuple[float, float]:
    """RMS pixel and metre distances of the matched lines' endpoints (and the matched ellipses' points)
    to their model markings."""
    if len(matches) == 0 and len(arc_matches) == 0:
        return float("nan"), float("nan")
    model_lines = {ln.name: ln for ln in model.lines()}
    h_px2m = np.linalg.inv(h_m2px)
    d_px, d_m = [], []
    for m in matches:
        ln = lines[m.image_index]
        ml = model_lines[m.model_name].homogeneous
        l_px = _normalise_lines(ml @ h_px2m)
        e = np.concatenate([ln.endpoints(), np.ones((2, 1))], axis=1)
        d_px.extend(np.abs(e @ l_px).tolist())
        xy = apply_homography(h_px2m, ln.endpoints())
        d_m.extend(np.abs(np.concatenate([xy, np.ones((2, 1))], axis=1) @ ml).tolist())
    circles = {c.name: c for c in _model_circles(model)}
    for am in arc_matches:
        ell, c = ellipses[am.ellipse_index], circles[am.model_name]
        cm = circle_conic(c.centre, c.radius)
        d_px.extend(conic_distance(h_px2m.T @ cm @ h_px2m, ell.points).tolist())
        xy = apply_homography(h_px2m, ell.points)
        d_m.extend(np.abs(np.linalg.norm(xy - np.asarray(c.centre), axis=1) - c.radius).tolist())
    return float(np.sqrt(np.mean(np.square(d_px)))), float(np.sqrt(np.mean(np.square(d_m))))


def _families_ok(matches: Sequence[LineMatch], model: PitchModel, min_inliers: int,
                 arc_matches: Sequence[ArcMatch] = ()) -> bool:
    """Enough matched markings to fix the homography: two lines of each family (and ``min_inliers``
    lines), or a matched circle (centre circle or penalty arc, worth two lines) with at least one line
    of each family."""
    kinds = {ln.name: ln.kind for ln in model.lines()}
    names = {m.model_name for m in matches}
    nx = sum(1 for n in names if kinds[n] == X_CONST)
    ny = sum(1 for n in names if kinds[n] == Y_CONST)
    if len(matches) >= min_inliers and nx >= 2 and ny >= 2:
        return True
    return len(arc_matches) > 0 and nx >= 1 and ny >= 1 and len(matches) + 2 * len(arc_matches) >= min_inliers


def camera_from_homography(h_m2px: np.ndarray, width: int, height: int) -> Tuple[float, float, Optional[np.ndarray]]:
    """The camera behind a metre -> pixel homography, assuming square pixels and the principal point
    at the image centre: ``(focal_px, inconsistency, centre_xyz)``.

    The first two columns of ``K^-1 H`` are the pitch axes in camera coordinates, so they must be
    orthogonal and of equal length; each condition gives ``1 / f^2`` and the least-squares value of
    the two is used. ``inconsistency`` is how far the two columns then are from orthogonal and equal
    (the larger of ``|cos angle|`` and ``|log length ratio|``; 0.024 median, 0.13 at the 95th
    percentile on homographies fitted to SoccerNet GSR labels). ``centre_xyz`` is the camera position
    in pitch metres, ``z`` its height above the pitch. NaN / None when no positive focal length fits.
    """
    t = np.array([[1.0, 0.0, -width / 2.0], [0.0, 1.0, -height / 2.0], [0.0, 0.0, 1.0]])
    hh = t @ np.asarray(h_m2px, float)
    if not np.isfinite(hh).all():
        return float("nan"), float("inf"), None
    hh = hh / np.linalg.norm(hh)
    h1, h2, h3 = hh[:, 0], hh[:, 1], hh[:, 2]
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = -np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    if float(a @ a) < 1e-30:
        return float("nan"), float("inf"), None
    inv_f2 = float(a @ b) / float(a @ a)
    if not inv_f2 > 0:
        return float("nan"), float("inf"), None
    f = 1.0 / math.sqrt(inv_f2)
    k_inv = np.diag([1.0 / f, 1.0 / f, 1.0])
    r1, r2, tt = k_inv @ h1, k_inv @ h2, k_inv @ h3
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    incons = max(abs(float(r1 @ r2)) / (n1 * n2), abs(math.log(n1 / n2)))
    lam = 2.0 / (n1 + n2)
    r1, r2, tt = r1 * lam, r2 * lam, tt * lam
    if tt[2] < 0:  # the pitch must be in front of the camera
        r1, r2, tt = -r1, -r2, -tt
    rot = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    centre = -rot.T @ tt
    centre[2] = -centre[2]  # x right and y toward the camera make the third axis point into the ground
    return f, incons, centre


@dataclass(frozen=True)
class VerifyConfig:
    """When a solution is refused (see :func:`verify_solution` and :func:`verify_failures`).

    The defaults were chosen on 225 frames (every 50th) of 15 SoccerNet GSR *train* clips, with the
    labelled person boxes as the people mask. There the calibrator claims 89 frames, 83 of them within
    1 m of the labels (precision 0.93) and none over 5 m; without the refusals it claimed 168 with 29
    over 5 m. The same box (``focal_range``, ``camera_box``) also prunes the global search.
    """

    min_support: float = 0.0
    max_missing: int = 99
    max_unexplained: float = 0.3
    max_loo_shift_m: float = 5.0
    max_critical: int = 99
    max_inconsistency: float = 0.2
    max_sigma_people_m: float = 0.11
    max_sigma_grid_m: float = 0.5  # used instead when no person boxes are given
    focal_range: Tuple[float, float] = (0.3, 8.0)  # in frame widths
    camera_box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-70.0, 70.0), (15.0, 150.0), (3.0, 50.0))  # camera centre x, y, height in pitch metres (GSR: main camera, +y side)
    enabled: bool = True
    refuse: bool = True  # False: record the evidence, refuse nothing


def _marking_elements(model: PitchModel, step_m: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Points every ``step_m`` along every drawn marking (each segment of a line, the centre circle, the drawn
    part of each penalty arc), (S, 2), with the index of their element, (S,), and the element names."""
    pts, idx, names = [], [], []
    for ln in model.lines():
        for k, (p0, p1) in enumerate(ln.segments):
            a, b = np.asarray(p0, float), np.asarray(p1, float)
            n = max(2, int(math.ceil(np.linalg.norm(b - a) / step_m)) + 1)
            pts.append(np.linspace(a, b, n)); idx.append(np.full(n, len(names))); names.append(f"{ln.name}#{k}")
    for c in _model_circles(model):
        n = int(math.ceil(2 * math.pi * c.radius / step_m))
        t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        p = np.stack([c.centre[0] + c.radius * np.cos(t), c.centre[1] + c.radius * np.sin(t)], axis=1)
        if c.x_side > 0:
            p = p[p[:, 0] >= c.x_limit]
        elif c.x_side < 0:
            p = p[p[:, 0] <= c.x_limit]
        pts.append(p); idx.append(np.full(len(p), len(names))); names.append(c.name)
    return np.concatenate(pts, axis=0), np.concatenate(idx), names


_ELEMENTS: Dict[int, Tuple[np.ndarray, np.ndarray, List[str]]] = {}


def verify_solution(h_m2px: np.ndarray, mask: np.ndarray, grass: np.ndarray, lines: Sequence[ImageLine],
                    matches: Sequence[LineMatch], ellipses: Sequence[ImageEllipse], arc_matches: Sequence[ArcMatch],
                    model: PitchModel = DEFAULT_PITCH, person_boxes: Optional[np.ndarray] = None,
                    min_inliers: int = 4, reach_px: Optional[float] = None, with_loo: bool = True,
                    arc_weight: float = 1.0, camera: bool = False) -> Dict[str, object]:
    """Evidence for and against a solution, from the frame itself.

    * ``support``: share of the model's marking points in view (inside the grass, outside the people) that
      land within ``reach_px`` of a marking pixel; ``n_visible`` how many there are.
    * ``missing``: drawn markings with at least 40 points in view of which under a quarter land on marking
      pixels (a wrong homography puts lines where the grass is empty).
    * ``unexplained``: share of the detected line length (lines of at least 60 px) that matches no model line.
    * ``loo_shift_m``: the largest move, in metres, of the visible pitch when the fit is redone without one of
      the matched model markings (all image pieces matched to it); ``critical`` counts the markings without
      which the rest no longer fixes the homography. Independent subsets of a right solution agree.
    * ``focal_ratio`` (focal length in frame widths), ``inconsistency`` and ``camera`` (centre x, y, z),
      from :func:`camera_from_homography`.
    * ``sigma_grid_m``: the 90th percentile of the fit's standard error (:func:`fit_uncertainty_m`) over a
      grid of the visible pitch; ``sigma_people_m``: the median over the feet of ``person_boxes``.
    """
    import cv2

    h, w = mask.shape[:2]
    out: Dict[str, object] = {}
    key = id(model)
    if key not in _ELEMENTS:
        _ELEMENTS[key] = _marking_elements(model)
    pts, elem, _names = _ELEMENTS[key]
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ np.asarray(h_m2px, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        px = hom[:, :2] / hom[:, 2:3]
    # the side of w > 0 is the side of the image centre's pre-image
    hc = np.linalg.inv(h_m2px) @ np.array([w / 2.0, h / 2.0, 1.0])
    front = hom[:, 2] * np.sign(hc[2]) > 0
    vis = front & np.isfinite(px).all(axis=1) & (px[:, 0] >= 1) & (px[:, 0] < w - 2) & (px[:, 1] >= 1) & (px[:, 1] < h - 2)
    pi = np.zeros((len(px), 2), dtype=np.int64)
    pi[vis] = np.round(px[vis]).astype(np.int64)
    vis &= grass[pi[:, 1], pi[:, 0]] > 0
    if person_boxes is not None and len(person_boxes):
        vis &= ~person_box_mask((h, w), person_boxes)[pi[:, 1], pi[:, 0]]
    reach = reach_px if reach_px is not None else max(3.0, 0.002 * w)
    r = int(math.ceil(reach))
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    hit = np.zeros(len(px), dtype=bool)
    hit[vis] = near[pi[vis, 1], pi[vis, 0]] > 0
    n_vis = int(vis.sum())
    out["n_visible"] = n_vis
    out["support"] = float(hit[vis].mean()) if n_vis else 0.0
    nv = np.bincount(elem[vis], minlength=int(elem.max()) + 1)
    nh = np.bincount(elem[hit], minlength=int(elem.max()) + 1)
    big = nv >= 40
    out["missing"] = int((big & (nh < 0.25 * np.maximum(nv, 1))).sum())
    out["present"] = int((big & (nh >= 0.5 * np.maximum(nv, 1))).sum())
    matched = {m.image_index for m in matches}
    tot = sum(ln.length for ln in lines if ln.length >= 60.0)
    un = sum(ln.length for i, ln in enumerate(lines) if ln.length >= 60.0 and i not in matched)
    out["unexplained"] = float(un / tot) if tot > 0 else 0.0
    f, inc, cc = camera_from_homography(h_m2px, w, h)
    out["focal_ratio"] = float(f / w) if np.isfinite(f) else float("nan")
    out["inconsistency"] = float(inc)
    out["camera"] = None if cc is None else [round(float(v), 2) for v in cc]
    gx, gy = np.meshgrid(np.linspace(0.05, 0.95, 12) * w, np.linspace(0.1, 0.95, 8) * h)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    gi = np.round(grid).astype(np.int64)
    grid = grid[grass[gi[:, 1], gi[:, 0]] > 0]
    h_inv = np.linalg.inv(h_m2px)
    base = apply_homography(h_inv, grid)
    on = model.contains(base, margin=5.0) if len(grid) else np.zeros(0, bool)
    grid, base = grid[on], base[on]
    res_fn = _fit_residuals(lines, matches, model, ellipses, arc_matches, arc_weight)
    sg = fit_uncertainty_m(h_m2px, res_fn, grid, w, h, model, camera) if len(grid) else np.array([np.inf])
    out["sigma_grid_m"] = float(np.percentile(sg, 90)) if np.isfinite(sg).all() else float("inf")
    if person_boxes is not None and len(person_boxes):
        b = np.asarray(person_boxes, float).reshape(-1, 4)
        sp = fit_uncertainty_m(h_m2px, res_fn, np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1), w, h, model, camera)
        out["sigma_people_m"] = float(np.median(sp)) if np.isfinite(sp).all() else float("inf")
    if with_loo:
        worst, critical = 0.0, 0
        names = sorted({m.model_name for m in matches}) + sorted({a.model_name for a in arc_matches})
        for name in names:
            m2 = [m for m in matches if m.model_name != name]
            a2 = [a for a in arc_matches if a.model_name != name]
            if not _families_ok(m2, model, min_inliers, a2):
                critical += 1
                continue
            if len(grid) == 0:
                continue
            h2 = refine_homography(h_m2px, lines, m2, model, width=w, height=h, ellipses=ellipses, arc_matches=a2,
                                   arc_weight=arc_weight, camera=camera)
            try:
                moved = np.linalg.norm(apply_homography(np.linalg.inv(h2), grid) - base, axis=1)
            except np.linalg.LinAlgError:
                moved = np.array([np.inf])
            worst = max(worst, float(np.nanmax(moved)) if np.isfinite(moved).any() else float("inf"))
        out["loo_shift_m"] = worst
        out["critical"] = critical
    return out


def verify_failures(v: Dict[str, object], cfg: VerifyConfig, width: int) -> List[str]:
    """The reasons ``cfg`` refuses a solution with verification record ``v`` (empty: accepted)."""
    why = []
    if v["support"] < cfg.min_support:
        why.append(f"support {v['support']:.2f} < {cfg.min_support}")
    if v["missing"] > cfg.max_missing:
        why.append(f"{v['missing']} markings in view are missing")
    if v["unexplained"] > cfg.max_unexplained:
        why.append(f"{v['unexplained']:.2f} of the line length unexplained")
    if v.get("loo_shift_m", 0.0) > cfg.max_loo_shift_m:
        why.append(f"leave-one-out shift {v['loo_shift_m']:.2f} m")
    if v.get("critical", 0) > cfg.max_critical:
        why.append(f"{v['critical']} markings are each indispensable")
    fr = v["focal_ratio"]
    if not (np.isfinite(fr) and cfg.focal_range[0] <= fr <= cfg.focal_range[1]):
        why.append(f"focal length {fr:.2f} frame widths")
    if v["inconsistency"] > cfg.max_inconsistency:
        why.append(f"camera inconsistency {v['inconsistency']:.2f}")
    if "sigma_people_m" in v:
        if not v["sigma_people_m"] <= cfg.max_sigma_people_m:
            why.append(f"standard error at the people {v['sigma_people_m']:.2f} m")
    elif not v.get("sigma_grid_m", 0.0) <= cfg.max_sigma_grid_m:
        why.append(f"standard error over the pitch {v['sigma_grid_m']:.2f} m")
    cam = v["camera"]
    if cam is None or not all(lo <= c <= hi for c, (lo, hi) in zip(cam, cfg.camera_box)):
        why.append(f"camera at {cam}")
    return why


DEFAULT_VERIFY = VerifyConfig()


def default_tau_px(width: int, tau_px: Optional[float] = None) -> float:
    """Inlier distance for the global search and the final match (pixels)."""
    return tau_px if tau_px is not None else max(2.5, 0.006 * width)


def calibrate_frame(frame_bgr: np.ndarray, model: PitchModel = DEFAULT_PITCH, prior_m2px: Optional[np.ndarray] = None,
                    min_inliers: int = 4, tau_px: Optional[float] = None, track_tau_px: Optional[float] = None,
                    mask_kw: Optional[dict] = None, line_kw: Optional[dict] = None,
                    extra_priors: Sequence[np.ndarray] = (), min_tracked_inliers: int = 0,
                    person_boxes: Optional[np.ndarray] = None, mask: str = "contrast",
                    use_arcs: bool = True, ellipse_kw: Optional[dict] = None,
                    verify: Optional[VerifyConfig] = None, arc_weight: float = 0.3,
                    camera_fit: bool = True, max_alternatives: int = 3, search_camera_prior: bool = True) -> CalibrationResult:
    """Calibrate one frame from scratch, or from ``prior_m2px`` (tracking) with a
    global search as the fallback. ``extra_priors`` are tried after ``prior_m2px``
    (for instance the previous solution when the first prior is an extrapolation)
    and before the global search. A tracked solution with fewer than
    ``min_tracked_inliers`` matched lines is rejected as a false lock (when the camera
    moves further than the tracking tolerance between frames the lines can snap onto
    their neighbours and still fit: 6 m of error with a plausible residual, measured) and
    the global search decides. The result carries the extracted ``lines``.

    ``person_boxes`` ((N, 4) pixels) are left out of the marking mask (see :func:`marking_mask`;
    ``mask="white"`` uses the old fixed-threshold mask). Straight candidates that bow like a piece
    of a curve are dropped (``line_kw`` ``max_bend_px`` / ``min_bend_radius_px``), and with
    ``use_arcs`` the curved markings are fitted as ellipses (:func:`detect_ellipses`, ``ellipse_kw``)
    and used in the search, the matching and the refinement, so a frame with the halfway line, one
    touchline and the centre circle can be calibrated.

    Every solution is checked against the frame (:func:`verify_solution`) and refused when ``verify``
    (default :data:`DEFAULT_VERIFY`) says so: a refused prior falls through to the next prior or to
    the global search, a refused global solution makes the frame ``uncalibrated``, with the reasons in
    ``note`` and the evidence in ``verify``."""
    h, w = frame_bgr.shape[:2]
    tau = default_tau_px(w, tau_px)
    tau_track = track_tau_px if track_tau_px is not None else 4.0 * tau
    cfg = verify if verify is not None else DEFAULT_VERIFY
    grass = grass_region(frame_bgr)
    if mask == "white":
        mpx = white_line_mask(frame_bgr, **(mask_kw or {}))
        if person_boxes is not None and len(person_boxes):
            mpx[person_box_mask(mpx.shape, person_boxes)] = 0
    else:
        mpx = marking_mask(frame_bgr, person_boxes=person_boxes, grass=grass, **(mask_kw or {}))
    weights = line_weights(frame_bgr, mpx)
    if person_boxes is not None and len(person_boxes):
        # the soft weights reach 3 px past the mask: keep them out of the people too, or the fit sees half
        # a stroke at every box edge
        weights[person_box_mask(mpx.shape, person_boxes)] = 0.0
    line_kw = {"weights": weights, "max_bend_px": DEFAULT_MAX_BEND_PX,
               "min_bend_radius_px": DEFAULT_MIN_BEND_RADIUS * w, "peel_halo_px": DEFAULT_PEEL_HALO_PX,
               **(line_kw or {})}
    lines = extract_lines(mpx, **line_kw)
    ellipses = detect_ellipses(mpx, lines, **(ellipse_kw or {})) if use_arcs else []
    lines = drop_chords(lines, ellipses, max_len_px=0.25 * w)
    if len(lines) < 4 and not (ellipses and len(lines) >= 2):
        return CalibrationResult(status=STATUS_UNCALIBRATED, n_lines=len(lines), lines=lines, ellipses=ellipses,
                                 note=f"{len(lines)} straight markings and {len(ellipses)} curves found, need 4 lines "
                                      "or 2 lines and a curve")

    def finish(h0: np.ndarray, status: str, note: str) -> Optional[CalibrationResult]:
        hh = h0
        for round_tau in (tau_track if status == STATUS_TRACKED else tau * 2.0, tau):
            matches = match_lines(hh, lines, model, tau_px=round_tau)
            arcs = match_arcs(hh, ellipses, model, tau_px=round_tau)
            if not _families_ok(matches, model, min_inliers, arcs):
                return None
            aw = arc_weight if _families_ok(matches, model, min_inliers) else 1.0
            hh = refine_homography(hh, lines, matches, model, width=w, height=h, ellipses=ellipses, arc_matches=arcs,
                                   arc_weight=aw)
            incons = camera_from_homography(hh, w, h)[1]  # how far the free fit is from a camera
            if camera_fit:
                hh = refine_homography(hh, lines, matches, model, width=w, height=h, ellipses=ellipses,
                                       arc_matches=arcs, arc_weight=aw, camera=True)
        matches = match_lines(hh, lines, model, tau_px=tau)
        arcs = match_arcs(hh, ellipses, model, tau_px=tau)
        if not _families_ok(matches, model, min_inliers, arcs):
            return None
        fit_px, rep_m = _errors(hh, lines, matches, model, ellipses, arcs)
        h_inv = np.linalg.inv(hh)
        res = CalibrationResult(status=status, h_m2px=hh, h_px2m=h_inv / h_inv[2, 2],
                                matches=matches, n_lines=len(lines), fit_error_px=fit_px,
                                reprojection_error_m=rep_m, note=note, lines=lines, ellipses=ellipses,
                                arc_matches=arcs)
        if cfg.enabled:
            res.verify = verify_solution(hh, mpx, grass, lines, matches, ellipses, arcs, model, person_boxes,
                                         min_inliers=min_inliers, camera=camera_fit,
                                         arc_weight=arc_weight if _families_ok(matches, model, min_inliers) else 1.0)
            res.verify["inconsistency"] = float(incons)
            why = verify_failures(res.verify, cfg, w) if cfg.refuse else []
            if why:
                refused.append((status, why, res.verify))
                return None
        return res

    refused: List[Tuple[str, List[str], Dict[str, object]]] = []

    for prior in ([prior_m2px] if prior_m2px is not None else []) + list(extra_priors):
        res = finish(prior, STATUS_TRACKED, "tracked from the previous frame")
        if res is not None and res.n_inliers >= min_tracked_inliers:
            return res
    ranked: List[Tuple[float, np.ndarray]] = []
    h_best, matches, score = search_homography(lines, w, h, model, tau_px=tau, ellipses=ellipses, ranked=ranked,
                                               camera_prior=cfg if (cfg.enabled and cfg.refuse and search_camera_prior) else None)
    arcs = match_arcs(h_best, ellipses, model, tau_px=tau) if h_best is not None else []
    if h_best is None or not _families_ok(matches, model, min_inliers, arcs):
        return CalibrationResult(status=STATUS_UNCALIBRATED, n_lines=len(lines), lines=lines, ellipses=ellipses,
                                 note=f"global search: {len(matches)} consistent lines and {len(arcs)} curves, "
                                      f"need {min_inliers} lines across both families or a curve and a line of each")
    note = f"global search over {len(lines)} lines and {len(ellipses)} curves"
    res = finish(h_best, STATUS_CALIBRATED, note)
    # A refused best hypothesis may have beaten a right one on the score: try the next distinct ones
    # (those that are not within the matching tolerance of one already tried, at the image anchors).
    tried = [h_best]
    for _score, h_alt in ranked:
        if res is not None or len(tried) > max_alternatives:
            break
        if any(homography_shift_px(t, h_alt, w, h) <= 2.0 * tau for t in tried):
            continue
        tried.append(h_alt)
        m_alt = match_lines(h_alt, lines, model, tau_px=tau)
        if _families_ok(m_alt, model, min_inliers, match_arcs(h_alt, ellipses, model, tau_px=tau)):
            res = finish(h_alt, STATUS_CALIBRATED, note + f" (hypothesis {len(tried)})")
    if res is None:
        last = refused[-1] if refused and refused[-1][0] == STATUS_CALIBRATED else None
        return CalibrationResult(status=STATUS_UNCALIBRATED, n_lines=len(lines), lines=lines, ellipses=ellipses,
                                 note=("refused: " + "; ".join(last[1])) if last else
                                 "global search found a hypothesis that did not survive refinement",
                                 verify=dict(last[2]) if last else {})
    return res


# ----------------------------------------------------------------------------
# 5. Per-frame state
# ----------------------------------------------------------------------------


def frame_difference(prev_small: Optional[np.ndarray], frame_bgr: np.ndarray, size: Tuple[int, int] = (64, 36)) -> Tuple[float, np.ndarray]:
    """Mean absolute grey difference against the previous downscaled frame (0-255)."""
    import cv2

    small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(float)
    if prev_small is None:
        return 0.0, small
    return float(np.abs(small - prev_small).mean()), small


# Shot-cut detection. A grey frame difference alone misses a cut between two shots
# whose brightness happens to be alike (wide camera on one grass colour to a medium
# camera on another measured 27 on a 0-255 scale at 1280x720, and a fast pan of 6 m per
# frame measures 5.8), so it is paired with a hue-saturation histogram distance, which
# sees the colour change but not the camera moving. Measured on the synthetic clips:
#   within a shot  grey 2.0 (0.8 m/frame pan) .. 5.8 (6 m/frame),  HS distance 0.02 .. 0.05
#   stress pairs   grey up to 21 (+20 % exposure), HS distance up to 0.17 (mowing stripes)
#   cuts           grey 9 .. 27,                                    HS distance 0.33 .. 0.89
# The thresholds below sit between those groups. Real footage is unmeasured.
CUT_GREY_THRESHOLD = 25.0
CUT_HIST_THRESHOLD = 0.2
_SIG_SIZE = (160, 90)  # downscale for both features (width, height)
_HS_BINS = (30, 32)
_HS_BLUR_BINS = 1.0  # smooths the histogram so a small hue drift cannot flip a bin and read as a cut


@dataclass
class ShotSignature:
    """What the cut detector keeps from a frame: a 64x36 grey thumbnail and a smoothed HS histogram."""

    grey: np.ndarray
    hs: np.ndarray


def shot_signature(frame_bgr: np.ndarray) -> ShotSignature:
    import cv2

    small = cv2.resize(frame_bgr, _SIG_SIZE, interpolation=cv2.INTER_AREA)
    grey = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, list(_HS_BINS), [0, 180, 0, 256])
    hist = cv2.GaussianBlur(hist.astype(np.float32), (0, 0), _HS_BLUR_BINS, borderType=cv2.BORDER_REFLECT)
    return ShotSignature(grey=grey.astype(np.float32), hs=hist / max(float(hist.sum()), 1e-9))


def shot_change(prev: ShotSignature, cur: ShotSignature) -> Tuple[float, float]:
    """(mean absolute grey difference 0-255, Bhattacharyya distance of the HS histograms 0-1)."""
    grey = float(np.abs(prev.grey - cur.grey).mean())
    bc = float(np.sqrt(prev.hs * cur.hs).sum())
    return grey, math.sqrt(max(0.0, 1.0 - bc))


# Homography filtering. The 8 free entries of the normalised pixel-to-pitch map are
# filtered independently with an alpha-beta (constant-velocity) filter. A linear blend of
# consecutive homographies lags a panning camera by (1 - alpha) / alpha frames of motion
# (measured: up to 1.8 m on the 1.5 m/frame pan clip at alpha 0.6, against 0.1 m for the
# per-frame fit itself), whereas a constant-velocity filter has no lag on a steady pan and
# still averages the fit noise on a static shot.


def _h_to_params(h_m2px: np.ndarray, width: int, height: int, model: PitchModel) -> np.ndarray:
    hn = np.diag([1.0 / width, 1.0 / height, 1.0]) @ np.asarray(h_m2px, float) @ np.diag([model.half_length, model.half_width, 1.0])
    return (hn / hn[2, 2]).reshape(-1)[:8]


def _params_to_h(p: np.ndarray, width: int, height: int, model: PitchModel) -> np.ndarray:
    hn = np.append(np.asarray(p, float), 1.0).reshape(3, 3)
    h = np.diag([float(width), float(height), 1.0]) @ hn @ np.diag([1.0 / model.half_length, 1.0 / model.half_width, 1.0])
    return h / h[2, 2]


def homography_shift_px(h_a_m2px: np.ndarray, h_b_m2px: np.ndarray, width: int, height: int) -> float:
    """How far apart two metre -> pixel maps put the same pitch points, in pixels: the largest
    distance over four anchor pixels in the lower half of the image (where the pitch is) of
    the point ``h_a`` sees there against where ``h_b`` puts the same pitch point."""
    anchors = np.array([[0.25 * width, 0.6 * height], [0.75 * width, 0.6 * height],
                        [0.2 * width, 0.9 * height], [0.8 * width, 0.9 * height]])
    try:
        pitch = apply_homography(np.linalg.inv(h_a_m2px), anchors)
        d = np.linalg.norm(apply_homography(h_b_m2px, pitch) - anchors, axis=1).max()
    except np.linalg.LinAlgError:
        return float("inf")
    return float(d) if np.isfinite(d) else float("inf")


# Frame stride. During steady tracking only one frame in up to ``frame_stride`` is fitted to
# the markings; the frames between are held back and, once the next fit is in, interpolated
# between the fits. Measured back to back on the Mac (Apple M5) on a 150-frame 1920x1080 pan,
# fitting every frame costs 82 ms per frame, nearly all of it line extraction; a held-back
# frame costs about 1.5 ms (the shot signature and the alignment check below), so a stride of
# 6 brings the clip to 17 ms per frame (on the Colab L4 VM's CPU: 0.27 s to 0.056 s). Interpolation, not extrapolation: the image motion of
# a panning camera is not linear in the homography, so even with the exact velocity a
# constant-velocity prediction is 0.5 px off after two frames and 1.1 px after three on a
# 0.8 m/frame pan (1080p), and a broadcast pan also accelerates. A quadratic through the fit
# before the stretch, the fit that opens it and the fit that closes it follows both: on 0.8
# and 2 m/frame pans at 1080p the interpolated frames were no less accurate than the fitted
# ones (0.077 m worst against 0.105 m). What decides that a frame may be interpolated:
#   * the shot-cut test still runs on every frame; a cut closes the open stretch with a fit of
#     its last frame and then starts the new shot with a global search;
#   * frames are held back only after ``steady_frames`` one-frame tracked fits in a row whose
#     innovation (``last_shift_px``) is within ``hold_tol_px``, and the stretch grows by one
#     frame after each stretch that closed cleanly, up to ``frame_stride - 1`` held frames;
#   * the fit that closes a stretch must be tracked and land within ``close_tol_px`` of the
#     prediction it was tracked from; otherwise the state is rewound to the opening fit and
#     the held frames are fitted one by one, exactly as ``update`` would have done;
#   * every interpolated frame is checked against its own markings with
#     :func:`marking_alignment`: its median offset may exceed that of the two fits around it
#     by at most ``align_tol_px`` and its share of usable samples must be at least
#     ``support_keep`` times theirs, or the frame is fitted after all. On rendered 1080p
#     frames the median offset is 0.03 to 0.05 px for the true homography and 0.13 px for one
#     0.37 px off.
# Any failure drops back to fitting every frame until the camera is steady again. Measured on
# rendered 1080p clips (steady, fast, accelerating and oscillating pans, start-stop, a whip, a
# zoom, camera shake): no frame was less accurate than when fitting every frame, and on the
# moving cameras the stride falls back to about the cost of fitting every frame.
DEFAULT_FRAME_STRIDE = 6


def marking_samples(model: PitchModel = DEFAULT_PITCH, step_m: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Points every ``step_m`` metres along the drawn straight markings, (S, 2), and the index
    of each one's model line in ``model.lines()``, (S,)."""
    pts, idx = [], []
    for i, ln in enumerate(model.lines()):
        for p0, p1 in ln.segments:
            a, b = np.asarray(p0, float), np.asarray(p1, float)
            n = max(2, int(math.ceil(np.linalg.norm(b - a) / step_m)) + 1)
            pts.append(np.linspace(a, b, n))
            idx.append(np.full(n, i))
    return np.concatenate(pts, axis=0), np.concatenate(idx)


def marking_alignment(frame_bgr: np.ndarray, h_m2px: np.ndarray, samples: np.ndarray, sample_line: np.ndarray,
                      model_abc: np.ndarray, half_window_px: int = 5, min_contrast: float = 30.0) -> Tuple[float, float, int]:
    """How well ``h_m2px`` lines up with the frame's markings, without extracting them.

    At every marking sample in view the brightness profile across the projected line
    (``2 * half_window_px + 1`` pixels along its normal; the minimum colour channel, as in
    :func:`line_weights`, minus the profile's own minimum) is reduced to its centroid, the
    signed offset of the stroke from where the homography puts it. A sample is usable when
    the profile has ``min_contrast`` and its peak is not at either end (a player, or a
    stroke outside the window). Returns ``(median |offset| px over usable samples, usable
    share of the samples in view, samples in view)``; the median is NaN with fewer than 20
    usable samples. A few thousand pixel reads, against a full mask and line fit.
    """
    h, w = frame_bgr.shape[:2]
    m_px = half_window_px + 1
    hom = np.concatenate([samples, np.ones((len(samples), 1))], axis=1) @ np.asarray(h_m2px, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        px = hom[:, :2] / hom[:, 2:3]
        vis = ((hom[:, 2] * np.sign(h_m2px[2, 2]) > 0) & np.isfinite(px).all(axis=1)
               & (px[:, 0] >= m_px) & (px[:, 0] <= w - 1 - m_px) & (px[:, 1] >= m_px) & (px[:, 1] <= h - 1 - m_px))
    n_vis = int(vis.sum())
    if n_vis < 20:
        return float("nan"), 0.0, n_vis
    try:
        l_px = project_model_lines(np.linalg.inv(h_m2px), model_abc)
    except np.linalg.LinAlgError:
        return float("nan"), 0.0, n_vis
    nrm = l_px[sample_line[vis], :2]
    p = px[vis]
    offs = np.arange(-half_window_px, half_window_px + 1, dtype=float)
    q = np.rint(p[:, None, :] + offs[None, :, None] * nrm[:, None, :]).astype(np.int64)  # (S, taps, 2)
    d = ((q - p[:, None, :]) * nrm[:, None, :]).sum(axis=-1)  # signed distance of each tap from the line
    m = frame_bgr[q[..., 1], q[..., 0]].min(axis=-1).astype(np.float32)
    wgt = m - m.min(axis=1, keepdims=True)
    peak = wgt.argmax(axis=1)
    usable = (wgt.max(axis=1) >= min_contrast) & (peak > 0) & (peak < len(offs) - 1)
    if int(usable.sum()) < 20:
        return float("nan"), float(usable.mean()), n_vis
    off = (wgt[usable] * d[usable]).sum(axis=1) / np.maximum(wgt[usable].sum(axis=1), 1e-6)
    return float(np.median(np.abs(off))), float(usable.mean()), n_vis


# what a rewind restores (see FrameCalibrator._close_with)
_TRACK_STATE = ("_prior", "_p", "_v", "_n_inliers", "_missed", "_fit_no", "_steady", "last_shift_px")


class FrameCalibrator:
    """Stateful per-frame calibration with tracking, filtering, shot-cut resets and a frame stride.

    Two ways to feed it a clip's frames, in order:

    * ``update(frame)`` returns that frame's :class:`CalibrationResult` and fits every frame;
    * ``push(frame)`` applies the frame stride and returns the results that became final,
      in frame order: none while a frame is held back, several when a stretch closes;
      ``flush()`` at the end of the clip returns the rest, so every frame gets exactly one
      result. Held-back frames are kept by reference (at most ``frame_stride - 1`` of them)
      until their stretch closes, so they must not be modified in place.

    A result's ``project`` maps pixels to metres (NaN when uncalibrated). ``name`` lets
    the pipeline report it as its projector backend.

    Per fitted frame: the shot-cut test (grey difference ``>= cut_threshold`` or HS histogram
    distance ``>= hist_threshold``; ``cut_threshold >= 255`` switches the test off) clears
    the state; the frame is tracked from the filter's prediction (falling back to the
    previous solution, then to the global search); a tracked fit of the frame right after
    the last one is passed through the alpha-beta filter (``smoothing`` is alpha, 0 or 1
    turns the filter off) unless it disagrees with the prediction by more than ``guard_px``
    at the anchor pixels, which means the camera did something other than continue its
    motion and the new fit is trusted as it is.

    Frame stride (``push`` only; the rules are in the comment above ``DEFAULT_FRAME_STRIDE``):
    once tracking is steady (``steady_frames``, ``hold_tol_px``, default ``0.0016 * width``),
    up to ``frame_stride - 1`` frames are held back between fits and then interpolated
    (status ``interpolated``, no fit error, no inliers) if the closing fit is clean
    (``close_tol_px``, default the matching tolerance) and each frame passes
    :func:`marking_alignment` (``align_tol_px``, ``support_keep``); otherwise they are fitted.
    ``n_replayed`` counts the held frames that had to be fitted. ``frame_stride=1`` makes
    ``push`` fit every frame.
    """

    name = "line-calibration"

    def __init__(self, model: PitchModel = DEFAULT_PITCH, cut_threshold: float = CUT_GREY_THRESHOLD, smoothing: float = 0.6,
                 hold_frames: int = 5, min_inliers: int = 4, hist_threshold: float = CUT_HIST_THRESHOLD,
                 guard_px: Optional[float] = None, track_keep: float = 0.6, frame_stride: int = DEFAULT_FRAME_STRIDE,
                 steady_frames: int = 2, hold_tol_px: Optional[float] = None, close_tol_px: Optional[float] = None,
                 align_tol_px: float = 0.05, support_keep: float = 0.8, **calibrate_kw):
        self.model = model
        self.cut_threshold = cut_threshold
        self.hist_threshold = hist_threshold
        self.smoothing = smoothing
        self.hold_frames = hold_frames
        self.min_inliers = min_inliers
        self.guard_px = guard_px
        self.track_keep = track_keep
        self.frame_stride = max(1, int(frame_stride))
        self.steady_frames = steady_frames
        self.hold_tol_px = hold_tol_px
        self.close_tol_px = close_tol_px
        self.align_tol_px = align_tol_px
        self.support_keep = support_keep
        self.calibrate_kw = calibrate_kw
        self._samples, self._sample_line = marking_samples(model)
        self._model_abc = np.stack([ln.homogeneous for ln in model.lines()], axis=0)
        self.reset()

    def reset(self) -> None:
        self._prior: Optional[np.ndarray] = None  # last output homography (metres -> pixels)
        self._p: Optional[np.ndarray] = None  # filtered parameters of the last fitted frame
        self._v: Optional[np.ndarray] = None  # per-frame velocity of the parameters
        self._prev_sig: Optional[ShotSignature] = None
        self._size: Optional[Tuple[int, int]] = None  # (width, height) of the frames so far
        self._n_inliers = 0  # inliers of the last solved frame
        self._missed = 0
        self._frame_no = 0  # frames seen since the reset
        self._fit_no = 0  # frame number of the last fitted frame
        # recent fits of this shot, the interpolation nodes: (frame number, fitted parameters before the
        # alpha-beta filter, so that an accelerating pan does not drag the filter's lag into them,
        # alignment median px, usable share)
        self._fits: List[Tuple[int, np.ndarray, float, float]] = []
        self._pending: List[Tuple[int, np.ndarray]] = []  # held-back frames: (frame number, image)
        self._boxes: Dict[int, np.ndarray] = {}  # person boxes of the recent frames, by frame number
        self._steady = 0  # one-frame tracked fits in a row that continued the filter's motion (within hold_tol_px)
        self._hold_max = 1  # frames the next stretch may hold back: grows after each clean stretch, up to frame_stride - 1
        self._snapshot: Optional[dict] = None  # tracking state at the fit that opened the held stretch
        self.history: List[str] = []
        self.last_shot_change: Tuple[float, float] = (0.0, 0.0)
        self.last_shift_px: float = float("nan")
        self.last_prior_shift_px: float = float("nan")  # distance of the last fit from the prior it was tracked from
        self.n_replayed = 0  # held frames refitted one by one because their stretch did not close cleanly

    @property
    def n_pending(self) -> int:
        """Frames pushed whose results ``push`` has not returned yet."""
        return len(self._pending)

    def _drop_track(self) -> None:
        self._prior = None
        self._p = None
        self._v = None
        self._n_inliers = 0
        self._fits = []
        self._steady = 0
        self._hold_max = 1

    def _priors(self, width: int, height: int, dt: int = 1) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        if self._p is None:
            return None, []
        h_prev = _params_to_h(self._p, width, height, self.model)
        if self._v is None:
            return h_prev, []
        return _params_to_h(self._p + dt * self._v, width, height, self.model), [h_prev]

    def _filter(self, res: CalibrationResult, width: int, height: int, dt: int = 1) -> None:
        """Replace the fit in ``res`` by the filtered homography and update the filter state.

        ``dt`` is the number of frames since the last fit (1 unless frames were held back).
        Over a held-back stretch a constant-velocity prediction is off by a pixel or more on
        a smooth pan, so such a fit is taken as it is and only restarts the velocity.
        """
        p_raw = _h_to_params(res.h_m2px, width, height, self.model)
        alpha = self.smoothing
        have_prev = self._p is not None
        active = 0.0 < alpha < 1.0 and res.status == STATUS_TRACKED and have_prev and dt == 1
        p_new, v_new = p_raw, None
        if have_prev and not active:  # a fresh global search right after a solved frame, or a stretch: learn the motion
            v_new = (p_raw - self._p) / dt
        if active and self._v is not None:
            p_pred = self._p + self._v
            self.last_shift_px = homography_shift_px(_params_to_h(p_pred, width, height, self.model), res.h_m2px, width, height)
            guard = self.guard_px if self.guard_px is not None else 0.0008 * width
            if self.last_shift_px <= guard:
                beta = alpha * alpha / (2.0 - alpha)  # critically damped (Benedict-Bordner)
                innov = p_raw - p_pred
                p_new, v_new = p_pred + alpha * innov, self._v + beta * innov
            else:
                v_new = p_raw - self._p
        elif active:  # first tracked frame after a (re)start: no velocity yet, take the fit and start one
            v_new = p_raw - self._p
        if p_new is not p_raw:
            h_f = _params_to_h(p_new, width, height, self.model)
            tau = default_tau_px(width, self.calibrate_kw.get("tau_px"))
            matches = match_lines(h_f, res.lines, self.model, tau_px=tau)
            arcs = match_arcs(h_f, res.ellipses, self.model, tau_px=tau)
            if _families_ok(matches, self.model, self.min_inliers, arcs):
                res.h_m2px = h_f
                h_inv = np.linalg.inv(h_f)
                res.h_px2m = h_inv / h_inv[2, 2]
                res.matches, res.arc_matches = matches, arcs
                res.fit_error_px, res.reprojection_error_m = _errors(h_f, res.lines, matches, self.model,
                                                                     res.ellipses, arcs)
            else:  # the filtered solution no longer explains the markings: fall back to the fit
                p_new = p_raw
        self._p, self._v = p_new, v_new
        self._prior = res.h_m2px

    def _alignment(self, frame_bgr: np.ndarray, h_m2px: np.ndarray) -> Tuple[float, float]:
        med, usable, _n = marking_alignment(frame_bgr, h_m2px, self._samples, self._sample_line, self._model_abc)
        return med, usable

    def _cut_test(self, frame_bgr: np.ndarray) -> bool:
        sig = shot_signature(frame_bgr)
        cut = False
        if self._prev_sig is not None and self.cut_threshold < 255:
            self.last_shot_change = shot_change(self._prev_sig, sig)
            cut = self.last_shot_change[0] >= self.cut_threshold or self.last_shot_change[1] >= self.hist_threshold
        self._prev_sig = sig
        return cut

    def _fit(self, frame_bgr: np.ndarray, cut: bool) -> CalibrationResult:
        """Line fit of frame number ``self._frame_no``, updating the tracking state."""
        h, w = frame_bgr.shape[:2]
        dt = max(1, self._frame_no - self._fit_no)
        prior, extra = self._priors(w, h, dt)
        floor = max(self.min_inliers, int(math.ceil(self.track_keep * self._n_inliers))) if prior is not None else 0
        res = calibrate_frame(frame_bgr, self.model, prior_m2px=prior, extra_priors=extra, min_inliers=self.min_inliers,
                              min_tracked_inliers=floor, person_boxes=self._boxes.get(self._frame_no),
                              **self.calibrate_kw)
        res.shot_cut = bool(cut)
        self.last_prior_shift_px = (homography_shift_px(prior, res.h_m2px, w, h) if res.ok and prior is not None
                                    else float("nan"))
        if res.ok:
            p_raw = _h_to_params(res.h_m2px, w, h, self.model)
            self.last_shift_px = float("nan")
            self._filter(res, w, h, dt)
            hold_tol = self.hold_tol_px if self.hold_tol_px is not None else 0.0016 * w
            if dt == 1:
                self._steady = self._steady + 1 if (res.status == STATUS_TRACKED and self.last_shift_px <= hold_tol) else 0
                if self._steady == 0:
                    self._hold_max = 1
            med, usable = self._alignment(frame_bgr, res.h_m2px) if self.frame_stride > 1 else (float("nan"), 0.0)
            self._fits = (self._fits + [(self._frame_no, p_raw, med, usable)])[-3:]
            self._n_inliers = res.n_inliers
            self._missed = 0
        else:
            self._missed += 1
            self._v = None  # a velocity from before a gap says nothing about after it
            self._fits = []
            self._steady = 0
            if self._missed > self.hold_frames:
                self._drop_track()
        self._fit_no = self._frame_no
        self.history.append(res.status)
        return res

    def _may_hold(self) -> bool:
        return (self.frame_stride > 1 and self._steady >= self.steady_frames and self._p is not None
                and len(self._fits) >= 2 and len(self._pending) < min(self._hold_max, self.frame_stride - 1))

    def _interpolated(self, frame_no: int, width: int, height: int) -> np.ndarray:
        """Homography (metres -> pixels) at ``frame_no`` from the last three fits (two if that is
        all there is): Lagrange interpolation of the parameters, quadratic so that it follows a
        pan's curvature and acceleration."""
        p = np.zeros(8)
        for i, (ti, pi, _m, _u) in enumerate(self._fits):
            li = 1.0
            for j, (tj, _pj, _mj, _uj) in enumerate(self._fits):
                if j != i:
                    li *= (frame_no - tj) / (ti - tj)
            p = p + li * pi
        return _params_to_h(p, width, height, self.model)

    def _close_with(self, frame_bgr: np.ndarray, cut: bool) -> List[CalibrationResult]:
        """Fit frame number ``self._frame_no`` and return the results of the held-back frames before it, then its own.

        The stretch is interpolated only when this fit closes it cleanly: tracked, bracketed by
        two fits of this shot and within ``close_tol_px`` (default: the matching tolerance,
        ``0.006 * width``) of the prediction it was tracked from. Otherwise the camera did
        something the stretch cannot vouch for (a whip pan, a zoom, a false lock from a stale
        prior), and the tracking state is rewound to the fit that opened the stretch and every
        held frame, then this one, is fitted in order, exactly as ``update`` would have.
        """
        res = self._fit(frame_bgr, cut)
        if not self._pending:
            self._snapshot = None
            return [res]
        w = frame_bgr.shape[1]
        close_tol = self.close_tol_px if self.close_tol_px is not None else default_tau_px(w, self.calibrate_kw.get("tau_px"))
        fits = self._fits
        clean = (res.status == STATUS_TRACKED and len(fits) >= 2 and fits[-1][0] == self._frame_no
                 and fits[-2][0] < self._pending[0][0] and self.last_prior_shift_px <= close_tol)
        if clean:
            return self._interpolate_stretch(res)
        return self._replay(frame_bgr, cut)

    def _replay(self, frame_bgr: np.ndarray, cut: bool) -> List[CalibrationResult]:
        snap, pend, frame_no = self._snapshot, self._pending, self._frame_no
        self._pending, self._snapshot = [], None
        for k in _TRACK_STATE + ("_fits",):
            setattr(self, k, snap[k])
        del self.history[snap["_history_len"]:]
        self.n_replayed += len(pend)
        self._steady, self._hold_max = 0, 1
        out: List[CalibrationResult] = []
        for n, frame in pend:
            self._frame_no = n
            out.append(self._fit(frame, cut=False))
        self._frame_no = frame_no
        out.append(self._fit(frame_bgr, cut))
        return out

    def _interpolate_stretch(self, res: CalibrationResult) -> List[CalibrationResult]:
        """Interpolated results for the held-back frames (each checked against its markings, and
        fitted after all when the check fails), then ``res``, the fit that closed the stretch."""
        pend, self._pending, self._snapshot = self._pending, [], None
        h, w = pend[0][1].shape[:2]
        (ta, _pa, ma, ua), (tb, _pb, mb, ub) = self._fits[-2], self._fits[-1]
        out: List[CalibrationResult] = []
        for n, frame in pend:
            hm = self._interpolated(n, w, h)
            med, usable = self._alignment(frame, hm)
            if med <= max(ma, mb) + self.align_tol_px and usable >= self.support_keep * min(ua, ub):  # NaN fails
                h_inv = np.linalg.inv(hm)
                out.append(CalibrationResult(
                    status=STATUS_INTERPOLATED, h_m2px=hm, h_px2m=h_inv / h_inv[2, 2],
                    note=f"interpolated between the fits of frames {ta} and {tb}; marking offset {med:.2f} px"))
                continue
            # the interpolation does not line up with this frame's markings: fit it, starting from the interpolation
            self._steady, self._hold_max = 0, 1
            self.n_replayed += 1
            floor = max(self.min_inliers, int(math.ceil(self.track_keep * self._n_inliers)))
            out.append(calibrate_frame(frame, self.model, prior_m2px=hm, min_inliers=self.min_inliers,
                                       min_tracked_inliers=floor, person_boxes=self._boxes.get(n), **self.calibrate_kw))
        self.history[-1:-1] = [r.status for r in out]  # in frame order, before the closing fit
        if all(r.status == STATUS_INTERPOLATED for r in out):
            self._hold_max = min(self._hold_max + 1, self.frame_stride - 1)
        return out + [res]

    def _close_open_stretch(self) -> List[CalibrationResult]:
        """Close the held stretch with a fit of its last frame (before a cut, a new clip or flush)."""
        if not self._pending:
            return []
        n, frame = self._pending.pop()
        frame_no, self._frame_no = self._frame_no, n
        try:
            return self._close_with(frame, cut=False)
        finally:
            self._frame_no = frame_no

    def push(self, frame_bgr: np.ndarray, person_boxes: Optional[np.ndarray] = None) -> List[CalibrationResult]:
        """Feed the next frame (and the people detected in it, (N, 4) pixels, left out of the marking
        mask); returns the results that are final now, in frame order."""
        h, w = frame_bgr.shape[:2]
        out: List[CalibrationResult] = []
        if self._size is not None and self._size != (w, h):  # a different video: nothing carries over
            out += self.flush()
            self.reset()
        self._size = (w, h)
        cut = self._cut_test(frame_bgr)
        self._frame_no += 1
        self._boxes = {n: b for n, b in self._boxes.items() if n > self._frame_no - 2 * self.frame_stride - 2}
        if person_boxes is not None:
            self._boxes[self._frame_no] = np.asarray(person_boxes, float).reshape(-1, 4)
        if cut:
            out += self._close_open_stretch()
            self._drop_track()
            self._missed = 0
        elif self._may_hold():
            if not self._pending:
                self._snapshot = {k: getattr(self, k) for k in _TRACK_STATE}
                self._snapshot["_fits"] = list(self._fits)
                self._snapshot["_history_len"] = len(self.history)
            self._pending.append((self._frame_no, frame_bgr))
            return out
        return out + self._close_with(frame_bgr, cut)

    def flush(self) -> List[CalibrationResult]:
        """Results for the frames still held back (call once at the end of a clip)."""
        return self._close_open_stretch()

    def update(self, frame_bgr: np.ndarray, person_boxes: Optional[np.ndarray] = None) -> CalibrationResult:
        """One frame in, its result out: fits every frame (the frame stride is ``push``'s)."""
        if self._pending:
            raise RuntimeError("frames pushed earlier are still held back: call flush() before update()")
        stride, self.frame_stride = self.frame_stride, 1
        try:
            return self.push(frame_bgr, person_boxes)[-1]
        finally:
            self.frame_stride = stride

    # PitchProjector-compatible static use: project with the last good homography.
    def project(self, points_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_px, float).reshape(-1, 2)
        if self._prior is None:
            return np.full((len(pts), 2), np.nan)
        return apply_homography(np.linalg.inv(self._prior), pts)
