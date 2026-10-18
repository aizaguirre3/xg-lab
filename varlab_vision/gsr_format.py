"""The SoccerNet Game State Reconstruction (GSR) annotation format, written from
the public description, not from the GPL-3 sn-gamestate / TrackLab code.

Sources read for this module (2026-10-17):

* Somers et al., "SoccerNet Game State Reconstruction: End-to-End Athlete
  Tracking and Identification on a Minimap", arXiv:2404.11335, Figure S3 of the
  supplementary material (https://arxiv.org/html/2404.11335), which prints an
  abridged ``Labels-GameState.json`` and states the pitch frame: origin at the
  centre mark, x toward the right goal, y along the halfway line toward the
  camera, metres, pitch 105 x 68 m.
* The sn-gamestate README (https://github.com/SoccerNet/sn-gamestate): data
  lives under ``SoccerNetGS/{train,valid,test,challenge}/`` and the current
  annotation version is 1.3 (``info.version``); the README also defines the
  GS-HOTA similarity (see ``gshota.py``).

Layout on disk (one folder per clip)::

    SoccerNetGS/<split>/SNGS-<nnn>/Labels-GameState.json
    SoccerNetGS/<split>/SNGS-<nnn>/img1/000001.jpg ... 000750.jpg

``Labels-GameState.json`` has four top-level keys:

``info``
    ``version``, ``game_id``, ``id``, ``num_tracklets``, ``action_position``,
    ``action_class``, ``visibility``, ``game_time_start``, ``game_time_stop``,
    ``clip_start``, ``clip_stop``, ``name`` (e.g. ``SNGS-200``), ``im_dir``
    (``img1``), ``frame_rate`` (25), ``seq_length`` (750), ``im_ext`` (``.jpg``).
``images``
    one record per frame: ``is_labeled``, ``image_id``, ``file_name``
    (``000001.jpg``), ``height``, ``width``, ``has_labeled_person``,
    ``has_labeled_pitch``, ``has_labeled_camera``, ``ignore_regions_x``,
    ``ignore_regions_y``.
``annotations``
    object records (``supercategory`` ``"object"``): ``id``, ``image_id``,
    ``track_id``, ``category_id``, ``attributes`` {``role``, ``jersey``,
    ``team``}, ``bbox_image`` {``x``, ``y``, ``x_center``, ``y_center``, ``w``,
    ``h``} in pixels with (x, y) the top-left corner, and ``bbox_pitch``
    {``x_bottom_left``, ``y_bottom_left``, ``x_bottom_right``,
    ``y_bottom_right``, ``x_bottom_middle``, ``y_bottom_middle``} in metres,
    the image box's bottom edge projected on the pitch plane (version 1.3 adds
    a temporally smoothed athlete location; ``bbox_pitch_raw`` carries the
    unsmoothed one when present). Pitch records (``supercategory``
    ``"pitch"``) carry ``lines``: named polylines of normalised image points.
    Camera records (``supercategory`` ``"camera"``) carry the calibration.
``categories``
    the id -> name table. Figure S3 shows ``player`` = 1 and ``pitch`` = 5; the
    remaining ids below are the expected values and the loader always prefers
    the table found in the file itself.

Anything not listed above is passed through untouched by the loader.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FORMAT_SOURCE_URL = "https://arxiv.org/html/2404.11335"  # Figure S3, supplementary
README_SOURCE_URL = "https://github.com/SoccerNet/sn-gamestate"

LABELS_FILE = "Labels-GameState.json"
DEFAULT_IM_DIR = "img1"
DEFAULT_IM_EXT = ".jpg"
DEFAULT_FRAME_RATE = 25
SPLITS: Tuple[str, ...] = ("train", "valid", "test", "challenge")

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

ROLE_PLAYER = "player"
ROLE_GOALKEEPER = "goalkeeper"
ROLE_REFEREE = "referee"
ROLE_BALL = "ball"
ROLE_OTHER = "other"
ROLES: Tuple[str, ...] = (ROLE_PLAYER, ROLE_GOALKEEPER, ROLE_REFEREE, ROLE_BALL, ROLE_OTHER)
TEAMS: Tuple[str, ...] = ("left", "right")

# Expected id -> (name, supercategory). Only 1 and 5 are confirmed by Figure S3;
# the loader overrides this table with the file's own ``categories`` list.
DEFAULT_CATEGORIES: Dict[int, Tuple[str, str]] = {
    1: ("player", "object"),
    2: ("goalkeeper", "object"),
    3: ("referee", "object"),
    4: ("ball", "object"),
    5: ("pitch", "pitch"),
    6: ("camera", "camera"),
    7: ("other", "object"),
}


def default_categories_list() -> List[Dict[str, Any]]:
    """The ``categories`` array as written into a new annotation file."""
    return [
        {"supercategory": sc, "id": cid, "name": name}
        for cid, (name, sc) in sorted(DEFAULT_CATEGORIES.items())
    ]


def category_id_for_role(role: str, categories: Optional[Dict[int, Tuple[str, str]]] = None) -> int:
    table = categories or DEFAULT_CATEGORIES
    for cid, (name, _sc) in table.items():
        if name == role:
            return cid
    raise KeyError(f"no category for role {role!r}")


@dataclass(frozen=True)
class BBoxImage:
    """Pixel box, (x, y) = top-left corner, as in ``bbox_image``."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x_center(self) -> float:
        return self.x + self.w / 2.0

    @property
    def y_center(self) -> float:
        return self.y + self.h / 2.0

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def bottom_middle(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h)

    def to_json(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "w": self.w,
            "h": self.h,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "BBoxImage":
        return cls(x=float(d["x"]), y=float(d["y"]), w=float(d["w"]), h=float(d["h"]))


@dataclass(frozen=True)
class BBoxPitch:
    """Metres on the pitch plane, as in ``bbox_pitch`` (bottom edge of the image box)."""

    x_bottom_left: float
    y_bottom_left: float
    x_bottom_right: float
    y_bottom_right: float
    x_bottom_middle: float
    y_bottom_middle: float

    @property
    def xy(self) -> Tuple[float, float]:
        """The athlete's position used by GS-HOTA: the bottom-middle point."""
        return (self.x_bottom_middle, self.y_bottom_middle)

    def to_json(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "BBoxPitch":
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]

    @classmethod
    def from_points(cls, left: Tuple[float, float], right: Tuple[float, float]) -> "BBoxPitch":
        return cls(
            x_bottom_left=left[0],
            y_bottom_left=left[1],
            x_bottom_right=right[0],
            y_bottom_right=right[1],
            x_bottom_middle=(left[0] + right[0]) / 2.0,
            y_bottom_middle=(left[1] + right[1]) / 2.0,
        )


@dataclass
class ClipInfo:
    name: str
    game_id: str
    clip_id: str = ""
    version: str = "1.3"
    frame_rate: float = DEFAULT_FRAME_RATE
    seq_length: int = 0
    im_dir: str = DEFAULT_IM_DIR
    im_ext: str = DEFAULT_IM_EXT
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        d = dict(self.extra)
        d.update(
            {
                "version": self.version,
                "game_id": self.game_id,
                "id": self.clip_id,
                "name": self.name,
                "im_dir": self.im_dir,
                "frame_rate": self.frame_rate,
                "seq_length": self.seq_length,
                "im_ext": self.im_ext,
            }
        )
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ClipInfo":
        known = {"version", "game_id", "id", "name", "im_dir", "frame_rate", "seq_length", "im_ext"}
        return cls(
            name=str(d.get("name", "")),
            game_id=str(d.get("game_id", "")),
            clip_id=str(d.get("id", "")),
            version=str(d.get("version", "")),
            frame_rate=float(d.get("frame_rate", DEFAULT_FRAME_RATE)),
            seq_length=int(d.get("seq_length", 0)),
            im_dir=str(d.get("im_dir", DEFAULT_IM_DIR)),
            im_ext=str(d.get("im_ext", DEFAULT_IM_EXT)),
            extra={k: v for k, v in d.items() if k not in known},
        )


@dataclass
class ImageRecord:
    image_id: str
    file_name: str
    width: int
    height: int
    is_labeled: bool = True
    has_labeled_person: bool = True
    has_labeled_pitch: bool = False
    has_labeled_camera: bool = False
    ignore_regions_x: List[Any] = field(default_factory=list)
    ignore_regions_y: List[Any] = field(default_factory=list)

    @property
    def frame_index(self) -> int:
        return frame_index_from_file_name(self.file_name)

    def to_json(self) -> Dict[str, Any]:
        return {
            "is_labeled": self.is_labeled,
            "image_id": self.image_id,
            "file_name": self.file_name,
            "height": self.height,
            "width": self.width,
            "has_labeled_person": self.has_labeled_person,
            "has_labeled_pitch": self.has_labeled_pitch,
            "has_labeled_camera": self.has_labeled_camera,
            "ignore_regions_y": list(self.ignore_regions_y),
            "ignore_regions_x": list(self.ignore_regions_x),
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=str(d["image_id"]),
            file_name=str(d["file_name"]),
            width=int(d["width"]),
            height=int(d["height"]),
            is_labeled=bool(d.get("is_labeled", True)),
            has_labeled_person=bool(d.get("has_labeled_person", True)),
            has_labeled_pitch=bool(d.get("has_labeled_pitch", False)),
            has_labeled_camera=bool(d.get("has_labeled_camera", False)),
            ignore_regions_x=list(d.get("ignore_regions_x", [])),
            ignore_regions_y=list(d.get("ignore_regions_y", [])),
        )


@dataclass
class ObjectAnnotation:
    """One athlete / ball box in one frame (``supercategory == "object"``)."""

    id: str
    image_id: str
    track_id: Optional[int]
    category_id: int
    role: str
    team: Optional[str]
    jersey: Optional[str]
    bbox_image: BBoxImage
    bbox_pitch: Optional[BBoxPitch] = None
    bbox_pitch_raw: Optional[BBoxPitch] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        d: Dict[str, Any] = dict(self.extra)
        d.update(
            {
                "id": self.id,
                "image_id": self.image_id,
                "track_id": self.track_id,
                "supercategory": "object",
                "category_id": self.category_id,
                "attributes": {"role": self.role, "jersey": self.jersey, "team": self.team},
                "bbox_image": self.bbox_image.to_json(),
                "bbox_pitch": self.bbox_pitch.to_json() if self.bbox_pitch else None,
            }
        )
        if self.bbox_pitch_raw is not None:
            d["bbox_pitch_raw"] = self.bbox_pitch_raw.to_json()
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any], categories: Dict[int, Tuple[str, str]]) -> "ObjectAnnotation":
        attrs = d.get("attributes") or {}
        cid = int(d["category_id"])
        role = attrs.get("role")
        if role is None:
            role = categories.get(cid, (ROLE_OTHER, "object"))[0]
        jersey = attrs.get("jersey")
        team = attrs.get("team")
        known = {
            "id", "image_id", "track_id", "supercategory", "category_id", "attributes",
            "bbox_image", "bbox_pitch", "bbox_pitch_raw",
        }
        return cls(
            id=str(d["id"]),
            image_id=str(d["image_id"]),
            track_id=None if d.get("track_id") is None else int(d["track_id"]),
            category_id=cid,
            role=str(role),
            team=None if team in (None, "") else str(team),
            jersey=None if jersey in (None, "") else str(jersey),
            bbox_image=BBoxImage.from_json(d["bbox_image"]),
            bbox_pitch=BBoxPitch.from_json(d["bbox_pitch"]) if d.get("bbox_pitch") else None,
            bbox_pitch_raw=BBoxPitch.from_json(d["bbox_pitch_raw"]) if d.get("bbox_pitch_raw") else None,
            extra={k: v for k, v in d.items() if k not in known},
        )


@dataclass
class GSRClip:
    """A parsed ``Labels-GameState.json`` plus the folder it came from."""

    path: Path
    info: ClipInfo
    images: List[ImageRecord]
    objects: List[ObjectAnnotation]
    categories: Dict[int, Tuple[str, str]]
    other_annotations: List[Dict[str, Any]] = field(default_factory=list)  # pitch, camera, unknown

    @property
    def clip_dir(self) -> Path:
        return self.path.parent

    @property
    def image_dir(self) -> Path:
        return self.clip_dir / self.info.im_dir

    def image_by_id(self) -> Dict[str, ImageRecord]:
        return {im.image_id: im for im in self.images}

    def to_json(self) -> Dict[str, Any]:
        return {
            "info": self.info.to_json(),
            "images": [im.to_json() for im in self.images],
            "annotations": [o.to_json() for o in self.objects] + list(self.other_annotations),
            "categories": [
                {"supercategory": sc, "id": cid, "name": name}
                for cid, (name, sc) in sorted(self.categories.items())
            ],
        }


def frame_index_from_file_name(file_name: str) -> int:
    """``000001.jpg`` -> 1. Raises ValueError for anything that is not ``<digits><ext>``."""
    stem = Path(file_name).stem
    if not stem.isdigit():
        raise ValueError(f"frame file name without a numeric stem: {file_name!r}")
    return int(stem)


def frame_file_name(frame_index: int, im_ext: str = DEFAULT_IM_EXT, width: int = 6) -> str:
    return f"{frame_index:0{width}d}{im_ext}"


def parse_categories(raw: Optional[Iterable[Dict[str, Any]]]) -> Dict[int, Tuple[str, str]]:
    """Prefer the file's own table; fall back to the expected defaults."""
    if not raw:
        return dict(DEFAULT_CATEGORIES)
    table: Dict[int, Tuple[str, str]] = {}
    for c in raw:
        table[int(c["id"])] = (str(c["name"]), str(c.get("supercategory", "object")))
    return table


def validate_attributes(role: str, team: Optional[str], jersey: Optional[str]) -> List[str]:
    """Return the format violations (empty list when valid).

    The GS-HOTA README requires null team and jersey for non-player roles, and a
    null jersey when it is not visible. Goalkeepers keep a team.
    """
    problems: List[str] = []
    if role not in ROLES:
        problems.append(f"unknown role {role!r}")
    if team is not None and team not in TEAMS:
        problems.append(f"unknown team {team!r}")
    if role in (ROLE_REFEREE, ROLE_BALL, ROLE_OTHER):
        if team is not None:
            problems.append(f"{role} must have team null, got {team!r}")
        if jersey is not None:
            problems.append(f"{role} must have jersey null, got {jersey!r}")
    if jersey is not None and not str(jersey).isdigit():
        problems.append(f"jersey must be digits or null, got {jersey!r}")
    return problems


def write_labels_json(clip: GSRClip, path: Optional[Path] = None, indent: Optional[int] = None) -> Path:
    """Write a loader-compatible ``Labels-GameState.json``."""
    out = Path(path) if path is not None else clip.path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(clip.to_json(), fh, indent=indent)
    return out


def new_image_id(clip_id: str, frame_index: int) -> str:
    """Mimic the ``<clip><frame:06d>``-style ids seen in Figure S3 (``3200000001``)."""
    return f"{clip_id}{frame_index:06d}"


def new_annotation_id(image_id: str, k: int) -> str:
    return f"{image_id}{k:04d}"
