#!/usr/bin/env python3
"""Probe the M4 vision environment on a Colab GPU and measure it on a synthetic clip.

    colab run --gpu L4 --timeout 600 -s varlab-vision-env colab/vision_env_probe.py --ref <commit-sha>

On the VM it clones this repository at ``--ref`` (pinned), installs
``requirements-vision.txt``, downloads the RF-DETR, SigLIP and RTMPose weights
from their official hosts, renders the synthetic clip (coloured rectangles on
green, 1920x1080, ``--frames`` frames) and reports installed versions, download
sizes, per-stage throughput and peak GPU memory. ``colab run`` provisions, runs
and releases the VM; nothing is left running.

The synthetic rectangles are not people, so RF-DETR is expected to detect
little or nothing on them; detection throughput does not depend on content, and
the tracking, embedding and pose stages are timed on the clip's ground-truth
boxes. The CPU smoke pipeline is also run and scored with GS-HOTA to close the
loader -> pipeline -> metric loop. No SoccerNet data is involved.

Local check (Mac, CPU): ``python colab/vision_env_probe.py --repo-dir . --skip-clone
--skip-install --device cpu --frames 3 --detector-size nano``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_REPO = "https://github.com/aizaguirre3/xg-lab.git"
SIGLIP_ID = "google/siglip-base-patch16-224"


def sh(cmd, capture=True):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, capture_output=capture)


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) if Path(path).exists() else 0


def mb(n: int) -> float:
    return round(n / 2**20, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--ref", default=os.environ.get("VARLAB_REF", "feat/varlab-vision"))
    ap.add_argument("--repo-dir", default="/content/xg-lab")
    ap.add_argument("--work-dir", default="/content/varlab_probe")
    ap.add_argument("--skip-clone", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--detector-size", default="medium", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--out", default=None, help="JSON summary path (default <work-dir>/vision_env_probe.json)")
    args = ap.parse_args(argv)

    t_all = time.time()
    R: dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
               "platform": platform.platform(), "cpus": os.cpu_count(), "steps": {}}
    print("python", R["python"], R["platform"], "cpus", R["cpus"])
    smi = shutil.which("nvidia-smi")
    if smi:
        q = sh([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        R["gpu"] = q.stdout.strip()
        print("gpu", R["gpu"])
    else:
        R["gpu"] = None
    try:
        with open("/proc/meminfo") as fh:
            R["ram"] = fh.readline().split()[1] + " kB"
    except OSError:
        R["ram"] = None

    repo_dir = Path(args.repo_dir)
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)

    # 1. Source at a pinned ref ------------------------------------------------
    if not args.skip_clone:
        t0 = time.time()
        if not repo_dir.is_dir():
            r = sh(["git", "clone", "--quiet", args.repo, str(repo_dir)])
            if r.returncode:
                print(r.stderr); return 1
        sh(["git", "-C", str(repo_dir), "fetch", "--quiet", "origin", args.ref])
        r = sh(["git", "-C", str(repo_dir), "checkout", "--quiet", args.ref])
        if r.returncode:
            print(r.stderr); return 1
        R["steps"]["clone_s"] = round(time.time() - t0, 1)
    head = sh(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]).stdout.strip()
    R["repo_head"] = head
    print("repo HEAD", head)

    # 2. Install the pinned requirements --------------------------------------
    if not args.skip_install:
        t0 = time.time()
        r = sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo_dir / "requirements-vision.txt")])
        R["steps"]["pip_install_s"] = round(time.time() - t0, 1)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:]); return 1
        if sys.platform.startswith("linux"):
            t0 = time.time()
            r = sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "onnxruntime-gpu==1.31.0"])
            R["steps"]["pip_onnxruntime_gpu_s"] = round(time.time() - t0, 1)
            if r.returncode:
                print("onnxruntime-gpu install failed (RTMPose will run on CPU):", r.stderr[-1000:])
    sys.path.insert(0, str(repo_dir))

    # 3. Versions --------------------------------------------------------------
    import numpy as np
    import cv2
    import torch

    versions = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                "numpy": np.__version__, "opencv": cv2.__version__}
    from importlib.metadata import PackageNotFoundError, version as dist_version

    for mod, dist in (("torchvision", "torchvision"), ("rfdetr", "rfdetr"), ("supervision", "supervision"),
                      ("trackers", "trackers"), ("transformers", "transformers"), ("huggingface_hub", "huggingface-hub"),
                      ("rtmlib", "rtmlib"), ("onnxruntime", "onnxruntime"), ("pandas", "pandas"), ("scipy", "scipy")):
        try:
            __import__(mod)
            versions[mod] = dist_version(dist)
        except PackageNotFoundError:
            versions[mod] = "installed (no dist metadata)"
        except Exception as e:  # noqa: BLE001
            versions[mod] = f"IMPORT FAILED: {type(e).__name__}"
    try:
        versions["onnxruntime_gpu"] = dist_version("onnxruntime-gpu")
    except PackageNotFoundError:
        versions["onnxruntime_gpu"] = None
    try:
        import onnxruntime as ort
        versions["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        versions["onnxruntime_providers"] = []
    R["versions"] = versions
    print("versions", json.dumps(versions))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pose_device = "cuda" if (device == "cuda" and "CUDAExecutionProvider" in versions["onnxruntime_providers"]) else "cpu"
    R["device"] = device; R["pose_device"] = pose_device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    from varlab_vision import pipeline as pl
    from varlab_vision.gshota import gs_hota
    from varlab_vision.loaders import objects_frame, read_gsr_clip
    from varlab_vision.synthetic import render_synthetic_clip

    # 4. Weights from their official hosts -----------------------------------
    W: dict = {}
    t0 = time.time()
    det = pl.RFDETRDetector(size=args.detector_size, threshold=0.5, device=device)
    from rfdetr.assets.model_weights import get_model_cache_dir
    cache = Path(get_model_cache_dir())
    pth = sorted(cache.glob(f"rf-detr-{args.detector_size}*.pth"))
    W["rfdetr"] = {"seconds": round(time.time() - t0, 1), "file": str(pth[0]) if pth else None,
                   "mb": mb(pth[0].stat().st_size) if pth else None, "host": "storage.googleapis.com/rfdetr"}
    print("rfdetr weights", W["rfdetr"], flush=True)

    t0 = time.time()
    from huggingface_hub import snapshot_download
    sig_dir = snapshot_download(SIGLIP_ID, revision=pl.SigLIPEmbedder.DEFAULT_REVISION,
                                allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
    emb = pl.SigLIPEmbedder(device=device)
    W["siglip"] = {"seconds": round(time.time() - t0, 1), "dir": sig_dir, "mb": mb(dir_size(Path(sig_dir).resolve())),
                   "revision": pl.SigLIPEmbedder.DEFAULT_REVISION, "host": "huggingface.co"}
    print("siglip weights", W["siglip"], flush=True)

    pose = None
    if not args.no_pose:
        t0 = time.time()
        try:
            pose = pl.RTMPoseEstimator(device=pose_device)
            rt_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rtmlib"
            W["rtmpose"] = {"seconds": round(time.time() - t0, 1), "mb": mb(dir_size(rt_cache)), "url": pose.url,
                            "device": pose_device, "host": "download.openmmlab.com"}
        except Exception as e:  # noqa: BLE001
            W["rtmpose"] = {"error": f"{type(e).__name__}: {e}"}
        print("rtmpose weights", W["rtmpose"], flush=True)
    R["weights"] = W

    # 5. Synthetic clip ---------------------------------------------------------
    syn_root = work / "syn"
    shutil.rmtree(syn_root, ignore_errors=True)
    clip = render_synthetic_clip(syn_root, n_frames=args.frames, width=args.width, height=args.height,
                                 n_per_team=5, with_referee=True, seed=0, box_wh=(60, 140))
    gt = read_gsr_clip(clip.labels_path)
    gt_df = objects_frame(gt)
    frames = [cv2.imread(str(clip.image_dir / im.file_name)) for im in gt.images]
    gt_boxes = {im.image_id: np.array([o.bbox_image.xyxy for o in gt.objects if o.image_id == im.image_id and o.role != "ball"])
                for im in gt.images}
    n_people = int(sum(len(b) for b in gt_boxes.values()))
    R["synthetic"] = {"frames": len(frames), "size": f"{args.width}x{args.height}", "people_boxes": n_people}

    # 6. Per-stage throughput on the real backends ------------------------------
    S: dict = {}

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for f in frames[:2]:  # warm-up
        det.detect(f)
    sync(); t0 = time.time()
    n_det = 0
    for f in frames:
        n_det += len(det.detect(f))
    sync(); dt = time.time() - t0
    S["detect"] = {"fps": round(len(frames) / dt, 2), "seconds": round(dt, 3), "detections": n_det,
                   "note": "RF-DETR on synthetic rectangles; count is expected to be ~0"}

    tracker = pl.ByteTrackTracker(frame_rate=25)
    t0 = time.time()
    for im, f in zip(gt.images, frames):
        b = gt_boxes[im.image_id]
        tracker.update(pl.FrameDetections(xyxy=b, confidence=np.ones(len(b)),
                                          class_name=np.array(["person"] * len(b), dtype=object)), f)
    dt = time.time() - t0
    S["track"] = {"fps": round(len(frames) / dt, 2), "seconds": round(dt, 3), "boxes_per_frame": round(n_people / len(frames), 1)}

    crops = [pl.crop(f, b) for im, f in zip(gt.images, frames) for b in gt_boxes[im.image_id]]
    emb.embed(crops[:8]); sync(); t0 = time.time()
    feats = emb.embed(crops)
    sync(); dt = time.time() - t0
    S["siglip_embed"] = {"crops_per_s": round(len(crops) / dt, 1), "frames_per_s_equiv": round(len(frames) / dt, 2),
                         "seconds": round(dt, 3), "crops": len(crops), "dim": int(feats.shape[1])}

    if pose is not None:
        pose.estimate(frames[0], gt_boxes[gt.images[0].image_id]); t0 = time.time()
        for im, f in zip(gt.images, frames):
            pose.estimate(f, gt_boxes[im.image_id])
        dt = time.time() - t0
        S["rtmpose"] = {"fps": round(len(frames) / dt, 2), "people_per_s": round(n_people / dt, 1), "seconds": round(dt, 3),
                        "device": pose_device}
    R["stages"] = S
    print("stages", json.dumps(S), flush=True)

    # 7. Full production pipeline on the clip (code path; expect few detections)
    prod = pl.VisionPipeline(detector=det, tracker=pl.ByteTrackTracker(frame_rate=25), embedder=emb, pose=pose,
                             projector=pl.HomographyProjector(clip.homography))
    rep = prod.run_clip(clip.clip_dir, work / "pred_prod" / "Labels-GameState.json", game_id="syn")
    R["production_run"] = rep.as_dict()
    print(rep.summary(), flush=True)

    # 8. Smoke pipeline + GS-HOTA closed loop (CPU backends)
    smoke = pl.smoke_pipeline(projector=pl.HomographyProjector(clip.homography))
    rep_s = smoke.run_clip(clip.clip_dir, work / "pred_smoke" / "Labels-GameState.json", game_id="syn")
    pred_df = objects_frame(read_gsr_clip(rep_s.out_path))
    h_all = gs_hota(gt_df, pred_df); h_pl = gs_hota(gt_df, pred_df, roles=("player",))
    R["smoke_run"] = rep_s.as_dict()
    R["smoke_gs_hota"] = {"all_roles": round(h_all.hota, 4), "players_only": round(h_pl.hota, 4),
                          "deta_players": round(h_pl.deta, 4), "assa_players": round(h_pl.assa, 4)}
    print(rep_s.summary()); print("smoke GS-HOTA", R["smoke_gs_hota"], flush=True)

    # 9. Memory ------------------------------------------------------------------
    if device == "cuda":
        R["gpu_mem"] = {"torch_max_allocated_mb": mb(torch.cuda.max_memory_allocated()),
                        "torch_max_reserved_mb": mb(torch.cuda.max_memory_reserved())}
        if smi:
            q = sh([smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader"])
            R["gpu_mem"]["nvidia_smi_used_total"] = q.stdout.strip()
    R["total_seconds"] = round(time.time() - t_all, 1)
    out = Path(args.out) if args.out else work / "vision_env_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1, default=str))
    print("PROBE_SUMMARY_JSON", json.dumps(R, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
