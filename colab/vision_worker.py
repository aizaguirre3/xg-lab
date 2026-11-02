"""VM-side worker for colab/run_clips.py: executed once with ``colab exec -f`` to define
``setup`` and ``process_clip`` in the kernel; the runner then calls them through
``colab exec`` with piped code. Kernel state persists between calls, so models are
loaded once per session.

Data layout (checked 2026-10-26): ``SoccerNet/SN-GSR-2025`` on Hugging Face is
public and not gated, and ships one zip per split (``valid.zip`` 11.2 GB, 58 clips,
each clip a top-level ``SNGS-xxx/`` folder with ``Labels-GameState.json`` and
``img1/``). With ``hf_layout="zip"`` (the default) a clip is read straight out of
the remote zip with HTTP range requests (``varlab_vision.remote_zip``): about
200 MB per clip, never the whole split. ``hf_layout="folders"`` keeps the old
per-clip ``snapshot_download`` with ``clip_pattern``. When the data is already on
the VM pass ``local_root`` to ``process_clip`` and no download happens.

Camera calibration: ``production_pipeline`` builds a ``FrameCalibrator`` by default
(classical line detection and a homography search on the CPU, about 0.2 s per
1080p frame on a laptop), so every clip comes back with pitch coordinates, a
``camera`` record per frame and a ``calibration`` block in ``report.json``.
``setup(..., with_calibration=False)`` turns it off (run_clips.py: ``--no-calibration``);
the output then has null ``bbox_pitch`` and GS-HOTA is skipped.

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


def _check_numpy_consistent() -> None:
    """Refuse to continue when the imported numpy is not the one on disk (kernel needs a restart)."""
    import sys as _sys
    from importlib.metadata import version

    mod = _sys.modules.get("numpy")
    if mod is not None and mod.__version__ != version("numpy"):
        raise RuntimeError(f"numpy {mod.__version__} is imported but {version('numpy')} is installed: "
                           "run `colab restart-kernel -s <session>` and call setup() again")


def setup(ref: str, repo: str = "https://github.com/aizaguirre3/xg-lab.git", detector_size: str = "medium",
          with_pose: bool = True, hf_dataset: str = "SoccerNet/SN-GSR-2025", hf_revision: str = "main",
          clip_pattern: str = "{split}/{clip}/*", device: str | None = None, install: bool = False,
          gpu_onnxruntime: bool = True, with_calibration: bool = True, hf_layout: str = "zip") -> dict:
    """Clone at ``ref``, (optionally) install the pinned requirements, load the models. Idempotent per kernel.

    ``install=False`` because colab/run_clips.py installs with ``colab install`` and
    restarts the kernel first; installing inside a kernel that already imported
    numpy mixes versions, which ``_check_numpy_consistent`` catches.
    """
    t0 = time.time()
    if not REPO_DIR.is_dir():
        _sh(["git", "clone", "--quiet", repo, str(REPO_DIR)])
    _sh(["git", "-C", str(REPO_DIR), "fetch", "--quiet", "origin", ref])
    _sh(["git", "-C", str(REPO_DIR), "checkout", "--quiet", ref])
    head = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], check=True, text=True,
                          capture_output=True).stdout.strip()
    print("repo HEAD", head)
    if install and not _STATE.get("installed"):
        _sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-vision.txt")])
        _STATE["installed"] = True
    if gpu_onnxruntime and sys.platform.startswith("linux") and not _STATE.get("gpu_ort") and "onnxruntime" not in sys.modules:
        _sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "onnxruntime-gpu==1.31.0"])
        _STATE["gpu_ort"] = True
    _check_numpy_consistent()
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    token = _read_token()
    if token:
        os.environ["HF_TOKEN"] = token  # huggingface_hub reads it; never printed
        print("HF token: present")
    else:
        print("HF token: absent (fine for public datasets such as SoccerNet/SN-GSR-2025)")

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
                                             pose_device=pose_dev, with_calibration=with_calibration)
    if hf_layout not in ("zip", "folders"):
        raise ValueError(f"hf_layout must be 'zip' or 'folders', got {hf_layout!r}")
    _STATE.update({"head": head, "hf_dataset": hf_dataset, "hf_revision": hf_revision, "clip_pattern": clip_pattern,
                   "device": dev, "pose_device": pose_dev, "with_calibration": with_calibration,
                   "hf_layout": hf_layout})
    info = {"head": head, "device": dev, "pose_device": pose_dev, "with_calibration": with_calibration,
            "setup_seconds": round(time.time() - t0, 1), "backends": _STATE["pipeline"].backends()}
    print("setup", json.dumps(info))
    return info


def _split_zip(split: str):
    """The split's remote zip, opened once per kernel: ``(ZipFile, RangeFile, {clip: prefix})``."""
    from varlab_vision.remote_zip import clip_index, hf_dataset_file_url, open_remote_zip

    cache = _STATE.setdefault("zips", {})
    if split not in cache:
        url = hf_dataset_file_url(_STATE["hf_dataset"], f"{split}.zip", _STATE["hf_revision"])
        zf, rf = open_remote_zip(url, token=os.environ.get("HF_TOKEN"),
                                 log=lambda m: print(f"  [remote zip] {m}", flush=True))
        cache[split] = (zf, rf, clip_index(zf))
        print(f"opened {url}: {rf.size / 1e9:.2f} GB, {len(cache[split][2])} clips", flush=True)
    return cache[split]


def _fetch_clip(clip: str, split: str) -> Path:
    """Put one clip under ``DATA_DIR/<split>/<clip>/`` and return that folder.

    ``hf_layout == "zip"``: range-read it out of ``<split>.zip``. ``"folders"``: download
    the clip's folder with ``snapshot_download`` and ``clip_pattern``.
    """
    if _STATE.get("hf_layout", "zip") == "zip":
        from varlab_vision.remote_zip import extract_clip

        zf, rf, index = _split_zip(split)
        before = rf.bytes_fetched
        t0 = time.time()
        clip_dir = extract_clip(zf, clip, DATA_DIR / split, index=index)
        _STATE["last_fetch"] = {"seconds": round(time.time() - t0, 1),
                                "mb": round((rf.bytes_fetched - before) / 1e6, 1)}
        return clip_dir
    from huggingface_hub import snapshot_download

    pattern = _STATE["clip_pattern"].format(split=split, clip=clip)
    local = DATA_DIR / split
    t0 = time.time()
    snapshot_download(repo_id=_STATE["hf_dataset"], repo_type="dataset", revision=_STATE["hf_revision"],
                      allow_patterns=[pattern], local_dir=str(local))
    _STATE["last_fetch"] = {"seconds": round(time.time() - t0, 1), "mb": None}
    for cand in local.rglob(clip):
        if cand.is_dir() and (cand / "img1").is_dir():
            return cand
    raise FileNotFoundError(f"{clip}: no folder with img1/ under {local} after downloading {pattern!r}; "
                            "check --clip-pattern against the dataset layout")


class _Heartbeat:
    """Print a line every ``every`` seconds while a block runs. ``colab exec`` gives up with
    "Timeout waiting for output" when the kernel is silent for too long (2026-10-26, during a
    clip whose download was being retried), so long silent steps must keep talking."""

    def __init__(self, label: str, every: float = 20.0):
        import threading

        self.label, self.every = label, every
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = time.time()

    def _run(self):
        while not self._stop.wait(self.every):
            print(f"  {self.label}: still working ({time.time() - self._t0:.0f} s)", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        return False


def process_clip(clip: str, split: str = "valid", local_root: str | None = None, delete_frames: bool = True) -> dict:
    """Run the pipeline on one clip and write ``/content/varlab_out/<split>/<clip>/{Labels-GameState.json,report.json}``."""
    with _Heartbeat(clip):
        return _process_clip(clip, split, local_root, delete_frames)


def _process_clip(clip: str, split: str, local_root: str | None, delete_frames: bool) -> dict:
    from varlab_vision.gshota import gs_hota
    from varlab_vision.gsr_format import LABELS_FILE
    from varlab_vision.loaders import list_frames, objects_frame, read_gsr_clip

    pipe = _STATE["pipeline"]
    t0 = time.time()
    _STATE.pop("last_fetch", None)
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
    rep["download"] = _STATE.get("last_fetch")
    if info is not None:
        gt = objects_frame(read_gsr_clip(clip_dir / LABELS_FILE))
        pred = objects_frame(read_gsr_clip(out_dir / LABELS_FILE))
        if pred["x_pitch"].notna().any():
            r = gs_hota(gt, pred)
            rep["gs_hota"] = {"hota": r.hota, "deta": r.deta, "assa": r.assa, "loca": r.loca}
        else:
            rep["gs_hota"] = None
            why = ("calibration is off (setup(with_calibration=False))" if not _STATE.get("with_calibration", True)
                   else "no frame could be calibrated")
            rep["notes"].append(f"GS-HOTA skipped: no pitch coordinates ({why})")
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    if delete_frames and not local_root:
        shutil.rmtree(clip_dir / (info.im_dir if info else "img1"), ignore_errors=True)
    cal = rep.get("calibration") or {}
    cal_txt = (f", calibration {cal.get('calibrated', 0)}/{cal.get('tracked', 0)}/{cal.get('interpolated', 0)}/"
               f"{cal.get('uncalibrated', 0)} (calibrated/tracked/interpolated/uncalibrated)") if cal else ""
    print(f"{clip}: {rep['frames']} frames, {rep['detections']} detections, {rep['tracks']} tracks, "
          f"{rep['wall_seconds']} s{cal_txt}", flush=True)
    return rep
