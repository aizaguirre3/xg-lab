"""Jersey numbers from player crops, read per crop and decided per track.

SoccerNet GSR labels the number per player, not per frame: on the train clips every box of a
numbered track carries the number (85 of 118 tracks have one), including the frames where the back
is turned away. So a crop's label says which number the player wears, not that it can be seen,
and nothing can be decided from one crop. The reader scores every crop and a track's number is the
one with the highest summed log-probability over its crops, kept only when it wins by a margin.

The network (a torchvision ResNet-18, BSD-3, ImageNet weights) predicts the number of digits (0, 1
or 2) and each digit separately, so a number never seen in training can still be read from its
digits. It sees the torso: the upper middle of the box, where the number is printed.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

CROP_SIZE = (96, 96)  # (width, height) of the torso crop the network sees
NUMBERS = tuple(str(n) for n in range(100))


def torso(crop_bgr: np.ndarray) -> np.ndarray:
    """The part of a person crop where the number is: rows 12-60%, columns 10-90%, resized."""
    import cv2

    h, w = crop_bgr.shape[:2]
    y0, y1 = int(0.12 * h), max(int(0.12 * h) + 1, int(0.60 * h))
    x0, x1 = int(0.10 * w), max(int(0.10 * w) + 1, int(0.90 * w))
    return cv2.resize(crop_bgr[y0:y1, x0:x1], CROP_SIZE, interpolation=cv2.INTER_AREA)


def encode(number: Optional[str]) -> Tuple[int, int, int]:
    """(digit count, first digit, second digit) for a label; -1 where a digit does not exist."""
    if number is None or not str(number).isdigit() or not 0 <= int(number) <= 99:
        return 0, -1, -1
    s = str(int(number))
    return (1, int(s), -1) if len(s) == 1 else (2, int(s[0]), int(s[1]))


def number_log_probs(log_len: np.ndarray, log_d1: np.ndarray, log_d2: np.ndarray) -> np.ndarray:
    """(N, 101) log-probabilities of "no number" and of 0..99 from the three heads' log-softmax outputs
    ((N, 3), (N, 10), (N, 10))."""
    n = len(log_len)
    out = np.empty((n, 101))
    out[:, 0] = log_len[:, 0]
    out[:, 1:11] = log_len[:, 1:2] + log_d1  # one digit: 0..9
    two = log_len[:, 2:3, None] + log_d1[:, :, None] + log_d2[:, None, :]  # (N, 10, 10): tens, units
    for t in range(1, 10):  # 10..99: a leading zero is not a two-digit number
        out[:, 1 + 10 * t:1 + 10 * t + 10] = two[:, t, :]
    return out


def decide(track_log_probs: np.ndarray, margin: float = 2.0, min_crops: int = 3) -> Optional[str]:
    """The track's number from its crops' (N, 101) log-probabilities: the summed best, if it beats the
    runner-up by ``margin`` nats over at least ``min_crops`` crops and is not "no number"."""
    if len(track_log_probs) < min_crops:
        return None
    total = track_log_probs.sum(axis=0)
    order = np.argsort(-total)
    if order[0] == 0 or total[order[0]] - total[order[1]] < margin:
        return None
    return NUMBERS[order[0] - 1]


def build_model(pretrained: bool = True):
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            enc.fc = nn.Identity()
            self.enc = enc
            self.n_len, self.d1, self.d2 = nn.Linear(512, 3), nn.Linear(512, 10), nn.Linear(512, 10)

        def forward(self, x):
            f = self.enc(x)
            return self.n_len(f), self.d1(f), self.d2(f)

    _ = torch  # the import above is what matters
    return Reader()


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_input(torso_bgr: np.ndarray) -> np.ndarray:
    img = torso_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((img - MEAN) / STD).transpose(2, 0, 1))


class JerseyReader:
    """``read(crops_by_track)`` -> ``{track: number}`` for the tracks it can decide."""

    name = "jersey-r18"

    def __init__(self, weights: str, device: Optional[str] = None, margin: float = 2.0, min_crops: int = 3,
                 batch_size: int = 128):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.to(self.device).eval()
        self.margin, self.min_crops, self.batch_size = margin, min_crops, batch_size

    def log_probs(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for i in range(0, len(crops_bgr), self.batch_size):
                x = torch.from_numpy(np.stack([to_input(torso(c)) for c in crops_bgr[i:i + self.batch_size]])).to(self.device)
                ln, d1, d2 = (torch.log_softmax(t.float(), dim=1).cpu().numpy() for t in self.model(x))
                out.append(number_log_probs(ln, d1, d2))
        return np.concatenate(out) if out else np.zeros((0, 101))

    def read(self, crops_by_track: Dict[int, Sequence[np.ndarray]]) -> Dict[int, str]:
        tids = [t for t, cs in crops_by_track.items() for _ in cs]
        lp = self.log_probs([c for cs in crops_by_track.values() for c in cs])
        tids = np.asarray(tids)
        out: Dict[int, str] = {}
        for t in crops_by_track:
            num = decide(lp[tids == t], self.margin, self.min_crops)
            if num is not None:
                out[int(t)] = num
        return out
