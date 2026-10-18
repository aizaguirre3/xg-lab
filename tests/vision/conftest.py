"""Fixtures for the vision tests: hand-written GSR JSON plus tiny rendered frames.

Nothing here touches SoccerNet data; everything is generated under tmp_path.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _write_frames(clip_dir: Path, n: int, width: int = 160, height: int = 90) -> None:
    cv2 = pytest.importorskip("cv2")
    img_dir = clip_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        frame = np.full((height, width, 3), (40, 150, 40), dtype=np.uint8)
        cv2.rectangle(frame, (10 + 2 * i, 20), (19 + 2 * i, 49), (30, 30, 220), -1)
        assert cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), frame)


@pytest.fixture
def gsr_root(tmp_path: Path) -> Path:
    """``<root>/train/SNGS-001``, ``<root>/train/SNGS-002``, ``<root>/valid/SNGS-003`` with frames."""
    root = tmp_path / "SoccerNetGS"
    layout = {"train": ["SNGS-001", "SNGS-002"], "valid": ["SNGS-003"]}
    for split, clips in layout.items():
        for clip in clips:
            dst = root / split / clip
            dst.mkdir(parents=True)
            shutil.copy(FIXTURES / clip / "Labels-GameState.json", dst / "Labels-GameState.json")
            n = {"SNGS-001": 3, "SNGS-002": 2, "SNGS-003": 1}[clip]
            _write_frames(dst, n)
    return root


@pytest.fixture
def clip_001(gsr_root: Path) -> Path:
    return gsr_root / "train" / "SNGS-001"
