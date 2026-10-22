"""Readers for SoccerNet GSR clips, written from the public format description.

Format source: Figure S3 of the supplementary material of arXiv:2404.11335
(https://arxiv.org/html/2404.11335) and the sn-gamestate README
(https://github.com/SoccerNet/sn-gamestate); field names are documented in
``gsr_format.py``. No code from sn-gamestate or TrackLab (GPL-3) was read or
reused.

Everything here takes explicit paths and works on hand-written fixtures; nothing
assumes the SoccerNet download exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .gsr_format import (
    LABELS_FILE,
    SPLITS,
    ClipInfo,
    GSRClip,
    ImageRecord,
    ObjectAnnotation,
    frame_index_from_file_name,
    parse_categories,
)

# ----------------------------------------------------------------------------
# Annotations
# ----------------------------------------------------------------------------


def read_gsr_clip(path: Path | str) -> GSRClip:
    """Parse one ``Labels-GameState.json`` (or the clip folder that contains it).

    Object annotations become :class:`ObjectAnnotation`; pitch, camera and any
    unknown supercategory are kept verbatim in ``other_annotations`` so a
    round-trip through :func:`gsr_format.write_labels_json` loses nothing.
    """
    p = Path(path)
    if p.is_dir():
        p = p / LABELS_FILE
    with open(p, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or "info" not in raw:
        raise ValueError(f"{p}: not a Labels-GameState.json (no 'info' key)")
    categories = parse_categories(raw.get("categories"))
    info = ClipInfo.from_json(raw["info"])
    images = [ImageRecord.from_json(d) for d in raw.get("images", [])]
    objects: List[ObjectAnnotation] = []
    others: List[Dict[str, Any]] = []
    for d in raw.get("annotations", []):
        sc = d.get("supercategory")
        if sc is None:
            sc = categories.get(int(d.get("category_id", -1)), ("", "object"))[1]
        if sc == "object" and "bbox_image" in d:
            objects.append(ObjectAnnotation.from_json(d, categories))
        else:
            others.append(d)
    return GSRClip(path=p, info=info, images=images, objects=objects, categories=categories, other_annotations=others)


def _float_or_nan(v: Any) -> float:
    """``float(v)`` for a number, NaN for None / missing / anything that is not numeric."""
    if v is None or isinstance(v, bool):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def objects_frame(clip: GSRClip, labeled_only: bool = True) -> pd.DataFrame:
    """Flatten the object annotations to one row per (frame, object).

    Columns: ``frame`` (1-based index parsed from the file name), ``image_id``,
    ``track_id``, ``category_id``, ``role``, ``team``, ``jersey``, ``x``, ``y``,
    ``w``, ``h`` (pixels, top-left), ``x_pitch``, ``y_pitch`` (metres, bottom-middle,
    NaN when ``bbox_pitch`` is null). ``labeled_only`` drops frames whose image
    record says ``is_labeled`` is false.

    Two optional columns come last and are read from ``ObjectAnnotation.extra``,
    where the vision pipeline puts them when it ran a per-frame calibrator:
    ``calib_error`` (the frame's RMS line-fit error in pixels; NaN when the key is
    absent or null, and on interpolated frames, which have no fit of their own) and
    ``calib_status`` (``calibrated`` / ``tracked`` / ``interpolated`` /
    ``uncalibrated``; None when absent). Annotations that never went through a
    calibrator, such as the ground truth, have NaN / None in both.
    """
    by_id = clip.image_by_id()
    rows: List[Dict[str, Any]] = []
    for o in clip.objects:
        im = by_id.get(o.image_id)
        if im is None:
            raise ValueError(f"{clip.path}: annotation {o.id} refers to unknown image_id {o.image_id}")
        if labeled_only and not im.is_labeled:
            continue
        xp, yp = (o.bbox_pitch.xy if o.bbox_pitch is not None else (np.nan, np.nan))
        rows.append(
            {
                "frame": im.frame_index,
                "image_id": o.image_id,
                "track_id": o.track_id,
                "category_id": o.category_id,
                "role": o.role,
                "team": o.team,
                "jersey": o.jersey,
                "x": o.bbox_image.x,
                "y": o.bbox_image.y,
                "w": o.bbox_image.w,
                "h": o.bbox_image.h,
                "x_pitch": xp,
                "y_pitch": yp,
                "calib_error": _float_or_nan(o.extra.get("calib_error")),
                "calib_status": o.extra.get("calib_status"),
            }
        )
    cols = ["frame", "image_id", "track_id", "category_id", "role", "team", "jersey",
            "x", "y", "w", "h", "x_pitch", "y_pitch", "calib_error", "calib_status"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)
    return df


CAMERA_COLUMNS = ["frame", "image_id", "status", "n_inliers", "fit_error_px", "reprojection_error_m", "shot_cut"]


def camera_frame(clip: GSRClip) -> pd.DataFrame:
    """One row per camera record (``supercategory == "camera"``), ordered by frame.

    Columns: ``frame`` (1-based index of the image the record belongs to),
    ``image_id``, ``status`` (``calibrated`` / ``tracked`` / ``interpolated`` /
    ``uncalibrated``; an interpolated frame has 0 inliers and NaN errors),
    ``n_inliers`` (markings matched to the pitch model), ``fit_error_px``,
    ``reprojection_error_m`` (NaN when null) and ``shot_cut``. These are the
    diagnostics under the record's ``calibration`` key, which the vision pipeline
    writes for every frame it calibrates. A camera record without that key (the
    ground truth keeps pan / tilt / position under ``camera`` instead) yields a
    row with ``status`` None, NaN numbers and ``shot_cut`` False; a clip with no
    camera records gives an empty frame with these columns.
    """
    by_id = clip.image_by_id()
    rows: List[Dict[str, Any]] = []
    for d in clip.other_annotations:
        if d.get("supercategory") != "camera":
            continue
        image_id = str(d.get("image_id"))
        im = by_id.get(image_id)
        if im is None:
            raise ValueError(f"{clip.path}: camera record {d.get('id')} refers to unknown image_id {image_id}")
        cal = d.get("calibration")
        cal = cal if isinstance(cal, dict) else {}
        n_in = cal.get("n_inliers")
        rows.append(
            {
                "frame": im.frame_index,
                "image_id": image_id,
                "status": cal.get("status"),
                "n_inliers": float("nan") if n_in is None else int(n_in),
                "fit_error_px": _float_or_nan(cal.get("fit_error_px")),
                "reprojection_error_m": _float_or_nan(cal.get("reprojection_error_m")),
                "shot_cut": bool(cal.get("shot_cut", False)),
            }
        )
    df = pd.DataFrame(rows, columns=CAMERA_COLUMNS)
    if len(df):
        df = df.sort_values("frame", kind="stable").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Image sequences
# ----------------------------------------------------------------------------


@dataclass
class FrameSequence:
    """The ordered frame files of one clip, validated against ``info``."""

    clip_dir: Path
    paths: List[Path]
    frame_indices: List[int]
    im_ext: str

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Tuple[int, Path]]:
        return iter(zip(self.frame_indices, self.paths))


def list_frames(clip_dir: Path | str, info: Optional[ClipInfo] = None, strict: bool = True) -> FrameSequence:
    """List ``<clip_dir>/<im_dir>/<nnnnnn><im_ext>`` sorted by frame index.

    With ``strict`` the count must equal ``info.seq_length`` (when known) and the
    indices must be 1..N without gaps, which is what the GSR clips promise.
    """
    clip_dir = Path(clip_dir)
    im_dir = clip_dir / (info.im_dir if info else "img1")
    im_ext = info.im_ext if info else ".jpg"
    if not im_dir.is_dir():
        raise FileNotFoundError(f"image folder missing: {im_dir}")
    files = [p for p in im_dir.iterdir() if p.is_file() and p.suffix.lower() == im_ext.lower()]
    indexed: List[Tuple[int, Path]] = []
    for p in files:
        try:
            indexed.append((frame_index_from_file_name(p.name), p))
        except ValueError:
            if strict:
                raise
    indexed.sort(key=lambda t: t[0])
    idx = [i for i, _ in indexed]
    if strict:
        if info is not None and info.seq_length and len(idx) != info.seq_length:
            raise ValueError(f"{clip_dir}: {len(idx)} frames on disk, info.seq_length says {info.seq_length}")
        if idx and idx != list(range(idx[0], idx[0] + len(idx))):
            raise ValueError(f"{clip_dir}: frame indices are not contiguous")
        if idx and idx[0] != 1:
            raise ValueError(f"{clip_dir}: frames start at {idx[0]}, expected 1")
    return FrameSequence(clip_dir=clip_dir, paths=[p for _, p in indexed], frame_indices=idx, im_ext=im_ext)


def iter_frames(seq: FrameSequence, color: str = "bgr") -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_index, image)`` with OpenCV (BGR by default, or ``"rgb"``)."""
    import cv2  # local import keeps the annotation readers free of OpenCV

    for idx, p in seq:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"could not read frame {p}")
        if color == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        yield idx, img


# ----------------------------------------------------------------------------
# Dataset discovery and the game-disjoint split check
# ----------------------------------------------------------------------------


def find_clips(root: Path | str, split: str) -> List[Path]:
    """Clip folders under ``<root>/<split>/`` that contain a ``Labels-GameState.json``."""
    base = Path(root) / split
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / LABELS_FILE).is_file())


def clip_game_ids(root: Path | str, splits: Sequence[str] = SPLITS) -> Dict[str, Dict[str, str]]:
    """``{split: {clip_name: game_id}}`` read from each clip's ``info``.

    Challenge clips have no public labels; a split folder that is absent or has
    no labelled clips simply maps to an empty dict.
    """
    out: Dict[str, Dict[str, str]] = {}
    for split in splits:
        games: Dict[str, str] = {}
        for clip_dir in find_clips(root, split):
            with open(clip_dir / LABELS_FILE, "r", encoding="utf-8") as fh:
                info = json.load(fh).get("info", {})
            games[str(info.get("name") or clip_dir.name)] = str(info.get("game_id", ""))
        out[split] = games
    return out


@dataclass
class SplitCheck:
    """Result of :func:`check_game_disjoint`."""

    ok: bool
    overlaps: Dict[str, List[str]] = field(default_factory=dict)  # game_id -> splits it appears in
    missing_game_id: Dict[str, List[str]] = field(default_factory=dict)  # split -> clips with empty game_id
    games_per_split: Dict[str, int] = field(default_factory=dict)
    clips_per_split: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"game-disjoint: {'OK' if self.ok else 'VIOLATED'}"]
        for s, n in self.clips_per_split.items():
            lines.append(f"  {s}: {n} clips, {self.games_per_split.get(s, 0)} games")
        for g, splits in sorted(self.overlaps.items()):
            lines.append(f"  game {g} appears in {', '.join(splits)}")
        for s, clips in self.missing_game_id.items():
            lines.append(f"  {s}: {len(clips)} clips without game_id ({', '.join(clips[:5])}{'...' if len(clips) > 5 else ''})")
        return "\n".join(lines)


def check_game_disjoint(split_games: Mapping[str, Mapping[str, str]],
                        pairs: Optional[Iterable[Tuple[str, str]]] = None) -> SplitCheck:
    """Verify that no match contributes clips to two different splits.

    ``split_games`` is ``{split: {clip_name: game_id}}`` as returned by
    :func:`clip_game_ids`. By default every pair of splits is checked; pass
    ``pairs`` to restrict (e.g. ``[("train", "valid")]``). Clips with an empty
    ``game_id`` cannot be checked and are reported, and they make the check fail
    because a missing id hides a possible leak.
    """
    splits = list(split_games.keys())
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(splits) for b in splits[i + 1:]]
    games_by_split: Dict[str, set] = {}
    missing: Dict[str, List[str]] = {}
    for s in splits:
        ids = set()
        for clip, g in split_games[s].items():
            if g in ("", None):
                missing.setdefault(s, []).append(clip)
            else:
                ids.add(str(g))
        games_by_split[s] = ids
    overlaps: Dict[str, List[str]] = {}
    for a, b in pairs:
        for g in games_by_split.get(a, set()) & games_by_split.get(b, set()):
            lst = overlaps.setdefault(g, [])
            for s in (a, b):
                if s not in lst:
                    lst.append(s)
    ok = not overlaps and not missing
    return SplitCheck(
        ok=ok,
        overlaps=overlaps,
        missing_game_id=missing,
        games_per_split={s: len(v) for s, v in games_by_split.items()},
        clips_per_split={s: len(split_games[s]) for s in splits},
    )
