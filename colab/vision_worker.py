"""VM-side worker for colab/run_clips.py: executed once with ``colab exec -f`` to define
``setup`` and ``process_clip`` in the kernel; the runner then calls them through
``colab exec`` with piped code. Kernel state persists between calls, so models are
loaded once per session.

Everything is parameterised; nothing here assumes the SoccerNet files exist, and
the layout of clips inside the Hugging Face dataset repository (``clip_pattern``)
is unverified until access arrives. When the data is already on the VM (uploaded
or mounted) pass ``local_root`` to ``process_clip`` and no download happens.

Safe handling: HTTPS to github.com / pypi.org / huggingface.co / storage.googleapis.com
/ download.openmmlab.com only; the repo is checked out at a pinned commit; the
token is read from Colab Secrets (``HF_TOKEN``) or from ``/content/.hf_token``,
which is deleted after reading; no downloaded content is executed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path("/content/xg-lab")
DATA_DIR = Path("/content/gsr_data")
OUT_DIR = Path("/content/varlab_out")
TOKEN_FILE = Path("/content/.hf_token")

_STATE: dict = {}


def _sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


def _read_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
        if token:
            return token
    except Exception:  # noqa: BLE001  (not in Colab, secret absent, or notebook access denied)
        pass
    if TOKEN_FILE.is_file():
        token = TOKEN_FILE.read_text().strip()
        try:
            TOKEN_FILE.unlink()
        except OSError:
            pass
        return token or None
    return None


def setup(ref: str, repo: str = "https://github.com/aizaguirre3/xg-lab.git", detector_size: str = "medium",
          with_pose: bool = True, hf_dataset: str = "SoccerNet/SN-GSR-2025", hf_revision: str = "main",
          clip_pattern: str = "{split}/{clip}/*", device: str | None = None) -> dict:
    """Clone at ``ref``, install the pinned requirements, load the models. Idempotent per kernel."""
    t0 = time.time()
    if not REPO_DIR.is_dir():
        _sh(["git", "clone", "--quiet", repo, str(REPO_DIR)])
    _sh(["git", "-C", str(REPO_DIR), "fetch", "--quiet", "origin", ref])
    _sh(["git", "-C", str(REPO_DIR), "checkout", "--quiet", ref])
    head = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], check=True, text=True,
                          capture_output=True).stdout.strip()
    print("repo HEAD", head)
    if not _STATE.get("installed"):
        _sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-vision.txt")])
        if sys.platform.startswith("linux"):
            _sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "onnxruntime-gpu==1.31.0"])
        _STATE["installed"] = True
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    token = _read_token()
    if token:
        os.environ["HF_TOKEN"] = token  # huggingface_hub reads it; never printed
        print("HF token: present")
    else:
        print("HF token: ABSENT (gated dataset downloads will fail)")

    import torch
    from varlab_vision.pipeline import production_pipeline

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pose_dev = "cpu"
    if with_pose and dev == "cuda":
        try:
            import onnxruntime as ort

            pose_dev = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        except Exception:  # noqa: BLE001
            pose_dev = "cpu"
    _STATE["pipeline"] = production_pipeline(detector_size=detector_size, device=dev, with_pose=with_pose,
                                             pose_device=pose_dev)
    _STATE.update({"head": head, "hf_dataset": hf_dataset, "hf_revision": hf_revision, "clip_pattern": clip_pattern,
                   "device": dev, "pose_device": pose_dev})
    info = {"head": head, "device": dev, "pose_device": pose_dev, "setup_seconds": round(time.time() - t0, 1),
            "backends": _STATE["pipeline"].backends()}
    print("setup", json.dumps(info))
    return info


def _fetch_clip(clip: str, split: str) -> Path:
    """Download one clip's folder from the dataset repo at the pinned revision; return its directory."""
    from huggingface_hub import snapshot_download

    pattern = _STATE["clip_pattern"].format(split=split, clip=clip)
    local = DATA_DIR / split
    snapshot_download(repo_id=_STATE["hf_dataset"], repo_type="dataset", revision=_STATE["hf_revision"],
                      allow_patterns=[pattern], local_dir=str(local))
    for cand in local.rglob(clip):
        if cand.is_dir() and (cand / "img1").is_dir():
            return cand
    raise FileNotFoundError(f"{clip}: no folder with img1/ under {local} after downloading {pattern!r}; "
                            "check --clip-pattern against the dataset layout")


def process_clip(clip: str, split: str = "valid", local_root: str | None = None, delete_frames: bool = True) -> dict:
    """Run the pipeline on one clip and write ``/content/varlab_out/<split>/<clip>/{Labels-GameState.json,report.json}``."""
    from varlab_vision.gshota import gs_hota
    from varlab_vision.gsr_format import LABELS_FILE
    from varlab_vision.loaders import list_frames, objects_frame, read_gsr_clip

    pipe = _STATE["pipeline"]
    t0 = time.time()
    clip_dir = Path(local_root) / split / clip if local_root else _fetch_clip(clip, split)
    info = None
    if (clip_dir / LABELS_FILE).is_file():
        info = read_gsr_clip(clip_dir / LABELS_FILE).info
    seq = list_frames(clip_dir, info, strict=info is not None)
    out_dir = OUT_DIR / split / clip
    out_dir.mkdir(parents=True, exist_ok=True)
    report = pipe.run_clip(clip_dir, out_dir / LABELS_FILE, clip_name=clip, game_id=(info.game_id if info else ""),
                           frames=seq, progress=lambda i, n: print(f"  {clip} {i}/{n}", flush=True) if i % 100 == 0 else None)
    rep = report.as_dict()
    rep["head"] = _STATE.get("head"); rep["wall_seconds"] = round(time.time() - t0, 1)
    if info is not None:
        gt = objects_frame(read_gsr_clip(clip_dir / LABELS_FILE))
        pred = objects_frame(read_gsr_clip(out_dir / LABELS_FILE))
        if pred["x_pitch"].notna().any():
            r = gs_hota(gt, pred)
            rep["gs_hota"] = {"hota": r.hota, "deta": r.deta, "assa": r.assa, "loca": r.loca}
        else:
            rep["gs_hota"] = None
            rep["notes"].append("GS-HOTA skipped: no pitch coordinates (calibration stage not implemented)")
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    if delete_frames and not local_root:
        shutil.rmtree(clip_dir / (info.im_dir if info else "img1"), ignore_errors=True)
    print(f"{clip}: {rep['frames']} frames, {rep['detections']} detections, {rep['tracks']} tracks, "
          f"{rep['wall_seconds']} s", flush=True)
    return rep
