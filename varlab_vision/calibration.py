"""Camera-to-pitch calibration from the pitch markings, written from scratch.

Classical and licence-free: no learned keypoint detector, no code from
PnLCalib (GPL-2) or TrackLab / sn-gamestate (GPL-3). The chain is

1. ``white_line_mask``: bright, low-saturation pixels next to grass.
2. ``extract_lines``: probabilistic Hough segments as seeds; each seed is refitted to the
   mask pixels around it by a robust cross-section-centroid fit (Theil-Sen + Tukey, with a
   band that follows the stroke thickness) and then *peeled*: its pixels are removed from
   the live mask so no second line can be found on the same marking.
3. ``search_homography``: a RANSAC-style hypothesis search over assignments of
   four image lines to two ``x = const`` and two ``y = const`` lines of the
   :class:`~varlab_vision.pitch_model.PitchModel`. Every hypothesis is scored on
   all detected lines: an image line is an inlier of a model line when its
   endpoints lie within ``tau_px`` of the projected model line *and* its
   back-projection lies on a drawn segment of that line (not on the invisible
   part of the infinite line). The search is vectorised (batched DLT) and the scorer is
   invariant to the scale and sign a caller gives the homography.
4. ``refine_homography``: Levenberg-Marquardt on the eight homography
   parameters, minimising the pixel distance between the inlier segments'
   endpoints and the projected model lines.
5. ``FrameCalibrator``: per-frame state. A shot-cut test (grey thumbnail difference or
   hue-saturation histogram distance) clears all state. Otherwise frame t is *tracked*
   from the filter's one-step prediction (then from the previous solution), falling back
   to the global search when tracking has too few correspondences or keeps clearly fewer
   inliers than the last frame (a false lock). A tracked fit goes through an alpha-beta
   (constant-velocity) filter on the eight homography parameters, so a steady pan is
   followed without lag while fit noise is averaged on a static shot.

A frame is **calibrated** only when at least ``min_inliers`` lines match with at
least two from each family (otherwise the homography is unconstrained along one
pitch axis); anything else is reported as ``uncalibrated`` with NaN pitch
coordinates, never as a guess.

Errors reported per frame: ``fit_error_px`` (RMS pixel distance of inlier
endpoints to their projected model lines, the quantity minimised) and
``reprojection_error_m`` (RMS metre distance of the same endpoints,
back-projected, to the model lines: how the pixel error looks on the pitch).

Limits, stated once: single camera, planar pitch (no height, no lens
distortion), straight markings only (the circle and arcs are not used yet),
and a view that shows at least four markings. Accuracy on real broadcast
frames is unmeasured until SoccerNet clips are available; the synthetic clip in
``synthetic.py`` has perfect lines and no crowd, so numbers from it are an upper
bound on what to expect.
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

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_CALIBRATED, STATUS_TRACKED) and self.h_px2m is not None

    @property
    def n_inliers(self) -> int:
        return len(self.matches)

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
            "matches": [{"line": m.model_name, "image_index": m.image_index,
                         "distance_px": round(m.distance_px, 2)} for m in self.matches],
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


def _dlt_batch(src: np.ndarray, dst: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Batched 4-point DLT: ``src`` and ``dst`` are (B, 4, 2); returns (B, 3, 3)
    homographies with ``dst ~ H src`` and ``h33 = 1``.

    Solved through the 8x8 normal equations with a tiny ridge so that a degenerate
    hypothesis (three collinear points) yields a finite but useless matrix instead
    of raising; the scorer then discards it. Much faster than a batched SVD.
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
    ata = np.einsum("bij,bik->bjk", a, a) + ridge * np.eye(8)[None]
    atb = np.einsum("bij,bik->bjk", a, b)
    sol = np.linalg.solve(ata, atb)[..., 0]
    h = np.concatenate([sol, np.ones((B, 1))], axis=1).reshape(B, 3, 3)
    return h


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


def extract_lines(mask: np.ndarray, min_length_px: float = 25.0, hough_threshold: int = 30,
                  max_gap_px: int = 10, merge_angle_deg: float = 2.0, merge_dist_px: float = 3.0,
                  band_px: float = 3.5, min_coverage: float = 0.6, max_lines: int = 24,
                  max_candidates: int = 150, bridge: float = 3.0,
                  weights: Optional[np.ndarray] = None) -> List[ImageLine]:
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
        out.append(ImageLine(p0=p0, p1=p1, abc=abc, n_pixels=int(n_inl)))
        # take this marking's pixels (stroke plus a halo) out of the live mask
        take = _band_select(pix, ys, alive, c, u, lo - 2.0, hi + 2.0, max(band_px, half_w + 1.0))
        alive[take] = False
        rem[ys[take] + 2, xs[take] + 2] = 0
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


def search_homography(lines: Sequence[ImageLine], width: int, height: int, model: PitchModel = DEFAULT_PITCH,
                      tau_px: Optional[float] = None, tol_m: float = 2.0, top_k: int = 6, n_score: int = 10,
                      n_rescore: int = 64, chunk: int = 40000) -> Tuple[Optional[np.ndarray], List[LineMatch], float]:
    """Global search: assign four image lines to two x = const and two y = const
    model lines, solve the homography from their four intersections, score on all
    lines; returns ``(H_m2px, matches, score)`` or ``(None, [], 0)``.
    """
    if len(lines) < 4:
        return None, [], 0.0
    tau = tau_px if tau_px is not None else max(2.5, 0.006 * width)
    model_abc, intervals, mlines = _segment_intervals(model)
    model_kind = np.array([ln.kind == X_CONST for ln in mlines])
    xs = [(i, ln.value) for i, ln in enumerate(mlines) if ln.kind == X_CONST]
    ys = [(i, ln.value) for i, ln in enumerate(mlines) if ln.kind == Y_CONST]
    k = min(top_k, len(lines))
    cand = list(range(k))  # lines are sorted by length
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
    img_pts = np.array(img_pts)  # (IA, 4, 2)
    mod_pts = np.array([[(xa[1], ya[1]), (xa[1], yb[1]), (xb[1], ya[1]), (xb[1], yb[1])]
                        for xa, xb, ya, yb in model_assign])  # (MA, 4, 2)
    IA, MA = len(img_assign), len(model_assign)
    ok_img = np.isfinite(img_pts).all(axis=(1, 2)) & _quad_orientation_ok(img_pts)
    # normalisation for conditioning
    s_px = np.array([width, height], float)
    s_m = np.array([model.half_length, model.half_width], float)
    pairs = [(i, j) for i in range(IA) if ok_img[i] for j in range(MA)]
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
        top = np.argsort(-score)[:n_rescore]
        leaders.extend((float(score[b]), h[b].copy()) for b in top if score[b] > 0)
    if not leaders:
        return None, [], 0.0
    leaders.sort(key=lambda t: -t[0])
    hs = np.stack([h for _, h in leaders[:n_rescore]], axis=0)
    full, _b, _i = _score_hypotheses(hs, lines, model_abc, intervals, model_kind, tau, tol_m,
                                     half_width=model.half_width, image_centre=(width / 2.0, height / 2.0))
    b = int(np.argmax(full))
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


def refine_homography(h_m2px: np.ndarray, lines: Sequence[ImageLine], matches: Sequence[LineMatch],
                      model: PitchModel = DEFAULT_PITCH, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Levenberg-Marquardt on the 8 free parameters: minimise the pixel distances of
    the matched segments' endpoints and midpoints to the projected model lines."""
    from scipy.optimize import least_squares

    if len(matches) == 0:
        return h_m2px
    model_lines = {ln.name: ln for ln in model.lines()}
    pts, mabc, wts = [], [], []
    for m in matches:
        ln = lines[m.image_index]
        ml = model_lines[m.model_name].homogeneous
        for p in (ln.p0, ln.mid, ln.p1):
            pts.append(p); mabc.append(ml); wts.append(math.sqrt(max(ln.length, 1.0)))
    pts = np.concatenate([np.array(pts), np.ones((len(pts), 1))], axis=1)
    mabc = np.array(mabc)
    wts = np.array(wts) / np.sqrt(np.mean(np.array(wts) ** 2))
    s_px = np.diag([width, height, 1.0])
    s_px_inv = np.linalg.inv(s_px)
    s_m = np.diag([model.half_length, model.half_width, 1.0])
    hn0 = s_px_inv @ h_m2px @ s_m  # normalised parametrisation
    hn0 = hn0 / hn0[2, 2]

    def unpack(p):
        hn = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]])
        return s_px @ hn @ np.linalg.inv(s_m)

    def residuals(p):
        h = unpack(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_inv = np.linalg.inv(h)
        l_px = _normalise_lines(mabc @ h_inv)
        return wts * (pts * l_px).sum(axis=1)

    p0 = hn0.reshape(-1)[:8]
    try:
        sol = least_squares(residuals, p0, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=400)
        h = unpack(sol.x)
        if np.isfinite(h).all():
            return h / h[2, 2]
    except (ValueError, np.linalg.LinAlgError):
        pass
    return h_m2px


def _errors(h_m2px: np.ndarray, lines: Sequence[ImageLine], matches: Sequence[LineMatch],
            model: PitchModel) -> Tuple[float, float]:
    if len(matches) == 0:
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
    return float(np.sqrt(np.mean(np.square(d_px)))), float(np.sqrt(np.mean(np.square(d_m))))


def _families_ok(matches: Sequence[LineMatch], model: PitchModel, min_inliers: int) -> bool:
    kinds = {ln.name: ln.kind for ln in model.lines()}
    names = {m.model_name for m in matches}
    nx = sum(1 for n in names if kinds[n] == X_CONST)
    ny = sum(1 for n in names if kinds[n] == Y_CONST)
    return len(matches) >= min_inliers and nx >= 2 and ny >= 2


def default_tau_px(width: int, tau_px: Optional[float] = None) -> float:
    """Inlier distance for the global search and the final match (pixels)."""
    return tau_px if tau_px is not None else max(2.5, 0.006 * width)


def calibrate_frame(frame_bgr: np.ndarray, model: PitchModel = DEFAULT_PITCH, prior_m2px: Optional[np.ndarray] = None,
                    min_inliers: int = 4, tau_px: Optional[float] = None, track_tau_px: Optional[float] = None,
                    mask_kw: Optional[dict] = None, line_kw: Optional[dict] = None,
                    extra_priors: Sequence[np.ndarray] = (), min_tracked_inliers: int = 0) -> CalibrationResult:
    """Calibrate one frame from scratch, or from ``prior_m2px`` (tracking) with a
    global search as the fallback. ``extra_priors`` are tried after ``prior_m2px``
    (for instance the previous solution when the first prior is an extrapolation)
    and before the global search. A tracked solution with fewer than
    ``min_tracked_inliers`` matched lines is rejected as a false lock (when the camera
    moves further than the tracking tolerance between frames the lines can snap onto
    their neighbours and still fit: 6 m of error with a plausible residual, measured) and
    the global search decides. The result carries the extracted ``lines``."""
    h, w = frame_bgr.shape[:2]
    tau = default_tau_px(w, tau_px)
    tau_track = track_tau_px if track_tau_px is not None else 4.0 * tau
    mask = white_line_mask(frame_bgr, **(mask_kw or {}))
    lines = extract_lines(mask, **{"weights": line_weights(frame_bgr, mask), **(line_kw or {})})
    if len(lines) < 4:
        return CalibrationResult(status=STATUS_UNCALIBRATED, n_lines=len(lines), lines=lines,
                                 note=f"{len(lines)} straight markings found, need 4")

    def finish(h0: np.ndarray, status: str, note: str) -> Optional[CalibrationResult]:
        hh = h0
        for round_tau in (tau_track if status == STATUS_TRACKED else tau * 2.0, tau):
            matches = match_lines(hh, lines, model, tau_px=round_tau)
            if not _families_ok(matches, model, min_inliers):
                return None
            hh = refine_homography(hh, lines, matches, model, width=w, height=h)
        matches = match_lines(hh, lines, model, tau_px=tau)
        if not _families_ok(matches, model, min_inliers):
            return None
        fit_px, rep_m = _errors(hh, lines, matches, model)
        h_inv = np.linalg.inv(hh)
        return CalibrationResult(status=status, h_m2px=hh, h_px2m=h_inv / h_inv[2, 2],
                                 matches=matches, n_lines=len(lines), fit_error_px=fit_px,
                                 reprojection_error_m=rep_m, note=note, lines=lines)

    for prior in ([prior_m2px] if prior_m2px is not None else []) + list(extra_priors):
        res = finish(prior, STATUS_TRACKED, "tracked from the previous frame")
        if res is not None and res.n_inliers >= min_tracked_inliers:
            return res
    h_best, matches, score = search_homography(lines, w, h, model, tau_px=tau)
    if h_best is None or not _families_ok(matches, model, min_inliers):
        return CalibrationResult(status=STATUS_UNCALIBRATED, n_lines=len(lines), lines=lines,
                                 note=f"global search: {len(matches)} consistent lines, need {min_inliers} across both families")
    res = finish(h_best, STATUS_CALIBRATED, f"global search over {len(lines)} lines")
    if res is None:
        return CalibrationResult(status=STATUS_UNCALIBRATED, n_lines=len(lines), lines=lines,
                                 note="global search found a hypothesis that did not survive refinement")
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


class FrameCalibrator:
    """Stateful per-frame calibration with tracking, filtering and shot-cut resets.

    ``update(frame)`` returns a :class:`CalibrationResult` whose ``project`` maps
    pixels to metres (NaN when uncalibrated). ``name`` lets the pipeline report it
    as its projector backend.

    Per frame: the shot-cut test (grey difference ``>= cut_threshold`` or HS histogram
    distance ``>= hist_threshold``; ``cut_threshold >= 255`` switches the test off) clears
    the state; the frame is tracked from the filter's one-step prediction (falling back to
    the previous solution, then to the global search); a tracked fit is passed through the
    alpha-beta filter (``smoothing`` is alpha, 0 or 1 turns the filter off) unless it
    disagrees with the prediction by more than ``guard_px`` at the anchor pixels, which
    means the camera did something other than continue its motion and the new fit is
    trusted as it is.
    """

    name = "line-calibration"

    def __init__(self, model: PitchModel = DEFAULT_PITCH, cut_threshold: float = CUT_GREY_THRESHOLD, smoothing: float = 0.6,
                 hold_frames: int = 5, min_inliers: int = 4, hist_threshold: float = CUT_HIST_THRESHOLD,
                 guard_px: Optional[float] = None, track_keep: float = 0.6, **calibrate_kw):
        self.model = model
        self.cut_threshold = cut_threshold
        self.hist_threshold = hist_threshold
        self.smoothing = smoothing
        self.hold_frames = hold_frames
        self.min_inliers = min_inliers
        self.guard_px = guard_px
        self.track_keep = track_keep
        self.calibrate_kw = calibrate_kw
        self.reset()

    def reset(self) -> None:
        self._prior: Optional[np.ndarray] = None  # last output homography (metres -> pixels)
        self._p: Optional[np.ndarray] = None  # its filtered parameters
        self._v: Optional[np.ndarray] = None  # per-frame velocity of the parameters
        self._prev_sig: Optional[ShotSignature] = None
        self._size: Optional[Tuple[int, int]] = None  # (width, height) of the frames so far
        self._n_inliers = 0  # inliers of the last solved frame
        self._missed = 0
        self.history: List[str] = []
        self.last_shot_change: Tuple[float, float] = (0.0, 0.0)
        self.last_shift_px: float = float("nan")

    def _drop_track(self) -> None:
        self._prior = None
        self._p = None
        self._v = None
        self._n_inliers = 0

    def _priors(self, width: int, height: int) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        if self._p is None:
            return None, []
        h_prev = _params_to_h(self._p, width, height, self.model)
        if self._v is None:
            return h_prev, []
        return _params_to_h(self._p + self._v, width, height, self.model), [h_prev]

    def _filter(self, res: CalibrationResult, width: int, height: int) -> None:
        """Replace the fit in ``res`` by the filtered homography and update the filter state."""
        p_raw = _h_to_params(res.h_m2px, width, height, self.model)
        alpha = self.smoothing
        have_prev = self._p is not None
        active = 0.0 < alpha < 1.0 and res.status == STATUS_TRACKED and have_prev
        p_new, v_new = p_raw, None
        if have_prev and not active:  # a fresh global search right after a solved frame: learn the motion from the pair
            v_new = p_raw - self._p
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
            matches = match_lines(h_f, res.lines, self.model, tau_px=default_tau_px(width, self.calibrate_kw.get("tau_px")))
            if _families_ok(matches, self.model, self.min_inliers):
                res.h_m2px = h_f
                h_inv = np.linalg.inv(h_f)
                res.h_px2m = h_inv / h_inv[2, 2]
                res.matches = matches
                res.fit_error_px, res.reprojection_error_m = _errors(h_f, res.lines, matches, self.model)
            else:  # the filtered solution no longer explains the markings: fall back to the fit
                p_new = p_raw
        self._p, self._v = p_new, v_new
        self._prior = res.h_m2px

    def update(self, frame_bgr: np.ndarray) -> CalibrationResult:
        h, w = frame_bgr.shape[:2]
        if self._size is not None and self._size != (w, h):  # a different video: nothing carries over
            self.reset()
        self._size = (w, h)
        sig = shot_signature(frame_bgr)
        cut = False
        if self._prev_sig is not None and self.cut_threshold < 255:
            self.last_shot_change = shot_change(self._prev_sig, sig)
            cut = self.last_shot_change[0] >= self.cut_threshold or self.last_shot_change[1] >= self.hist_threshold
        self._prev_sig = sig
        if cut:
            self._drop_track()
            self._missed = 0
        prior, extra = self._priors(w, h)
        floor = max(self.min_inliers, int(math.ceil(self.track_keep * self._n_inliers))) if prior is not None else 0
        res = calibrate_frame(frame_bgr, self.model, prior_m2px=prior, extra_priors=extra, min_inliers=self.min_inliers,
                              min_tracked_inliers=floor, **self.calibrate_kw)
        res.shot_cut = bool(cut)
        if res.ok:
            self._filter(res, w, h)
            self._n_inliers = res.n_inliers
            self._missed = 0
        else:
            self._missed += 1
            self._v = None  # a velocity from before a gap says nothing about after it
            if self._missed > self.hold_frames:
                self._drop_track()
        self.history.append(res.status)
        return res

    # PitchProjector-compatible static use: project with the last good homography.
    def project(self, points_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_px, float).reshape(-1, 2)
        if self._prior is None:
            return np.full((len(pts), 2), np.nan)
        return apply_homography(np.linalg.inv(self._prior), pts)
