"""Named pitch lines from a segmentation network, to tell the calibration which marking is which.

The classical calibration (``calibration.py``) finds the markings precisely but not their
names: on real broadcast frames it fails mostly because it cannot work out which pitch line a
segment is, not for lack of segments (``analysis/vision_env.md``). SoccerNet's Game State
Reconstruction labels name every visible marking on every frame (``Labels-GameState.json``,
``supercategory == "pitch"``: polylines in normalised image coordinates under the 26 names of
:data:`CLASSES`), so a network can be trained to label each pixel with the marking it belongs
to. Here it only names the lines the classical extractor found (:func:`name_lines`); their
geometry stays the extractor's, which is sub-pixel, and every solution still goes through the
same verification.

The network is a small U-Net on a torchvision ResNet-18 encoder (BSD-3; ImageNet weights),
written here rather than taken from a segmentation library. Weights trained on the GSR train
split stay out of the repository like the data (the dataset card says GPL-3.0).

The mapping from SoccerNet names to :class:`~varlab_vision.pitch_model.PitchModel` lines was
checked on train clips SNGS-060 and SNGS-099: projected through cameras fitted to the labelled
feet, every annotated line of each name lies on the model line given here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CLASSES: Tuple[str, ...] = (
    "Big rect. left bottom", "Big rect. left main", "Big rect. left top",
    "Big rect. right bottom", "Big rect. right main", "Big rect. right top",
    "Circle central", "Circle left", "Circle right",
    "Goal left crossbar", "Goal left post left", "Goal left post right",
    "Goal right crossbar", "Goal right post left", "Goal right post right",
    "Middle line",
    "Side line bottom", "Side line left", "Side line right", "Side line top",
    "Small rect. left bottom", "Small rect. left main", "Small rect. left top",
    "Small rect. right bottom", "Small rect. right main", "Small rect. right top",
)
N_CLASSES = len(CLASSES) + 1  # label 0 is background, label k is CLASSES[k - 1]

# SoccerNet name -> PitchModel line (straight markings on the ground) or circle; the goal frames
# stand off the ground and name nothing.
MODEL_NAME: Dict[str, str] = {
    "Big rect. left bottom": "penalty_side_near", "Big rect. left main": "penalty_front_left",
    "Big rect. left top": "penalty_side_far", "Big rect. right bottom": "penalty_side_near",
    "Big rect. right main": "penalty_front_right", "Big rect. right top": "penalty_side_far",
    "Circle central": "centre_circle", "Circle left": "penalty_arc_left", "Circle right": "penalty_arc_right",
    "Middle line": "halfway",
    "Side line bottom": "touchline_near", "Side line left": "goal_line_left",
    "Side line right": "goal_line_right", "Side line top": "touchline_far",
    "Small rect. left bottom": "goal_area_side_near", "Small rect. left main": "goal_area_front_left",
    "Small rect. left top": "goal_area_side_far", "Small rect. right bottom": "goal_area_side_near",
    "Small rect. right main": "goal_area_front_right", "Small rect. right top": "goal_area_side_far",
}
CIRCLES = ("centre_circle", "penalty_arc_left", "penalty_arc_right")
INPUT_SIZE = (640, 352)  # (width, height) the network sees; both divisible by 32


def rasterise_lines(lines: Dict[str, Sequence[dict]], width: int, height: int, thickness: int = 3) -> np.ndarray:
    """A (height, width) uint8 label map from GSR pitch polylines (``{name: [{"x", "y"}, ...]}``,
    normalised coordinates). Circles are drawn first so a straight line crossing one keeps its
    pixels; unknown names are skipped."""
    import cv2

    out = np.zeros((height, width), np.uint8)
    order = sorted(lines, key=lambda n: (not n.startswith("Circle"), n))
    for name in order:
        if name not in CLASSES:
            continue
        pts = np.array([[p["x"] * width, p["y"] * height] for p in lines[name]], float)
        if len(pts) < 2 or not np.isfinite(pts).all():
            continue
        fp = np.round(pts * 16.0).astype(np.int64).reshape(-1, 1, 2)
        cv2.polylines(out, [fp], False, int(CLASSES.index(name) + 1), thickness=thickness, lineType=cv2.LINE_8, shift=4)
    return out


def name_lines(lines: Sequence, labels: np.ndarray, frame_size: Tuple[int, int], step_px: float = 3.0,
               min_votes: int = 8, min_share: float = 0.6) -> Dict[int, str]:
    """``{index of an ImageLine: PitchModel line name}`` for the extracted lines the label map names.

    Each line is sampled every ``step_px`` frame pixels between its endpoints, and the label map
    (any resolution; ``frame_size`` is the frame's (width, height)) is read at those points. A line
    is named when at least ``min_votes`` samples carry a straight-marking class and one model line
    has ``min_share`` of them. Two image lines given the same name keep it only if they lie on one
    infinite line in the frame (within the label map's precision); otherwise neither is named.
    """
    lh, lw = labels.shape[:2]
    sx, sy = lw / float(frame_size[0]), lh / float(frame_size[1])
    straight = {k + 1: MODEL_NAME[c] for k, c in enumerate(CLASSES) if c in MODEL_NAME and MODEL_NAME[c] not in CIRCLES}
    out: Dict[int, str] = {}
    for i, ln in enumerate(lines):
        p0, p1 = np.asarray(ln.p0, float), np.asarray(ln.p1, float)
        n = max(2, int(np.linalg.norm(p1 - p0) / step_px) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts = p0 + t * (p1 - p0)
        c = np.clip(np.round(pts[:, 0] * sx).astype(int), 0, lw - 1)
        r = np.clip(np.round(pts[:, 1] * sy).astype(int), 0, lh - 1)
        votes = [straight[v] for v in labels[r, c].tolist() if v in straight]
        if len(votes) < min_votes:
            continue
        names, counts = np.unique(votes, return_counts=True)
        k = int(np.argmax(counts))
        if counts[k] >= min_share * len(votes):
            out[i] = str(names[k])
    by_name: Dict[str, List[int]] = {}
    for i, nm in out.items():
        by_name.setdefault(nm, []).append(i)
    tol = 2.0 / min(sx, sy)  # two label pixels, in frame pixels
    for nm, idx in by_name.items():
        if len(idx) < 2:
            continue
        ref = lines[max(idx, key=lambda j: lines[j].length)]
        far = [j for j in idx if max(abs(float(ref.abc @ np.append(p, 1.0))) for p in (lines[j].p0, lines[j].p1)) > tol]
        if far:
            for j in idx:
                del out[j]
    return out


def build_model(n_classes: int = N_CLASSES, pretrained: bool = True):
    """A U-Net on a torchvision ResNet-18: encoder strides 2 to 32, a decoder with skip connections
    back to stride 2, and a last upsampling to the input size."""
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # stride 2, 64
            self.pool = enc.maxpool
            self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4  # 4, 8, 16, 32
            self.d3 = block(512 + 256, 256)
            self.d2 = block(256 + 128, 128)
            self.d1 = block(128 + 64, 64)
            self.d0 = block(64 + 64, 64)
            self.head = nn.Conv2d(64, n_classes, 1)

        def forward(self, x):
            up = nn.functional.interpolate
            s0 = self.stem(x)
            s1 = self.l1(self.pool(s0))
            s2 = self.l2(s1)
            s3 = self.l3(s2)
            s4 = self.l4(s3)
            y = self.d3(torch.cat([up(s4, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3], 1))
            y = self.d2(torch.cat([up(y, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
            y = self.d1(torch.cat([up(y, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1))
            y = self.d0(torch.cat([up(y, size=s0.shape[-2:], mode="bilinear", align_corners=False), s0], 1))
            return up(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)

    return UNet()


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_input(frame_bgr: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 network input at :data:`INPUT_SIZE`, ImageNet-normalised RGB."""
    import cv2

    img = cv2.resize(frame_bgr, INPUT_SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((img - MEAN) / STD).transpose(2, 0, 1))


class LineSegmenter:
    """Inference wrapper: ``labels(frame_bgr)`` returns the label map at :data:`INPUT_SIZE`
    (pixels whose best class has probability under ``min_prob`` are background)."""

    name = "line-seg-unet-r18"

    def __init__(self, weights: str, device: Optional[str] = None, min_prob: float = 0.5):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = build_model(pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.to(self.device).eval()
        self.min_prob = min_prob

    def labels(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(to_input(frame_bgr))[None].to(self.device)
        with torch.inference_mode():
            p = torch.softmax(self.model(x).float(), dim=1)[0]
        conf, lab = p.max(dim=0)
        lab[conf < self.min_prob] = 0
        return lab.to(torch.uint8).cpu().numpy()
