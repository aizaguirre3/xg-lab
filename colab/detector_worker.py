"""Colab-side functions for fine-tuning RF-DETR on the SoccerNet GSR boxes (player / goalkeeper / referee / ball / other).

Sent to a fresh kernel with ``colab exec -f`` by ``colab/train_detector.py``, which then calls

    setup(ref)                         # clone the repo at the pinned commit, report the rfdetr API
    prepare(every=5, holdout=[...])    # download train.zip, write a COCO dataset of the sampled frames
    train(minutes=..., epochs=...)     # train, resuming from the last checkpoint

and downloads the checkpoints after every ``train`` call. The pretrained COCO weights come from
Roboflow's bucket through the package, as in the pipeline. The dataset is public; no token.
"""
from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/content/det")
REPO_DIR = Path("/content/xg-lab")
CATEGORIES = ("player", "goalkeeper", "referee", "ball", "other")  # COCO ids 1..5 in this order
GSR_TO_COCO = {1: 1, 2: 2, 3: 3, 4: 4, 7: 5}  # GSR category id -> COCO id here


def setup(ref: str, repo: str = "https://github.com/aizaguirre3/xg-lab.git") -> None:
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "--quiet", repo, str(REPO_DIR)], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--quiet", "origin", ref], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--quiet", ref], check=True)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    WORK.mkdir(parents=True, exist_ok=True)
    import rfdetr
    import torch

    print("ref", subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
          "rfdetr", getattr(rfdetr, "__version__", "?"), "torch", torch.__version__,
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU", flush=True)
    print("train signature:", inspect.signature(rfdetr.RFDETRMedium.train), flush=True)
    try:
        from rfdetr.config import TrainConfig

        print("TrainConfig fields:", sorted(TrainConfig.model_fields), flush=True)
    except Exception as e:  # the API moved between versions; the print above is what matters
        print("TrainConfig not importable:", e, flush=True)
    m = rfdetr.RFDETRMedium()
    print("model config:", {k: getattr(m.model_config, k, None) for k in ("resolution", "patch_size", "num_windows",
                                                                          "num_classes", "pretrain_weights")}, flush=True)


def _download(url: str, dest: Path) -> None:
    import urllib.request

    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(".part")
    t0, done = time.time(), 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "varlab-det/1"}), timeout=120) as r, \
            open(tmp, "wb") as fh:
        while True:
            chunk = r.read(8 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if done % (1 << 30) < (8 << 20):
                print(f"  {done / 1e9:.1f} GB in {time.time() - t0:.0f} s", flush=True)
    tmp.rename(dest)


def _clip_records(zip_path: str, clip: str, prefix: str, every: int):
    """(image file name in the dataset, JPEG bytes' member name, [(coco category, x, y, w, h)]) per sampled frame."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        raw = json.loads(zf.read(prefix + "Labels-GameState.json"))
    name_of = {im["image_id"]: im["file_name"] for im in raw["images"]}
    boxes = {}
    for a in raw["annotations"]:
        if a.get("supercategory") != "object" or a.get("category_id") not in GSR_TO_COCO:
            continue
        b = a.get("bbox_image") or {}
        if b.get("w", 0) <= 0 or b.get("h", 0) <= 0:
            continue
        boxes.setdefault(name_of[a["image_id"]], []).append(
            (GSR_TO_COCO[a["category_id"]], float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])))
    out = []
    for k, fname in enumerate(sorted(name_of.values())):
        if k % every == 0:
            out.append((f"{clip}_{fname}", prefix + "img1/" + fname, boxes.get(fname, [])))
    return out


def prepare(every: int = 5, holdout=(), split: str = "train", dataset: str = "SoccerNet/SN-GSR-2025") -> dict:
    """COCO dataset under ``WORK/data/{train,valid,test}`` (``test`` is a copy of ``valid``'s list:
    the package expects the folder); frames are copied as they are, at 1920x1080."""
    import zipfile

    from varlab_vision.remote_zip import clip_index, hf_dataset_file_url

    zip_path = WORK / f"{split}.zip"
    t0 = time.time()
    _download(hf_dataset_file_url(dataset, f"{split}.zip"), zip_path)
    print(f"downloaded {zip_path.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        index = clip_index(zf)
        parts = {"train": [], "valid": []}
        for clip, prefix in index.items():
            parts["valid" if clip in set(holdout) else "train"] += _clip_records(str(zip_path), clip, prefix, every)
        counts = {}
        for part, recs in parts.items():
            d = WORK / "data" / part
            d.mkdir(parents=True, exist_ok=True)
            images, anns = [], []
            for img_id, (fname, member, bxs) in enumerate(recs, start=1):
                (d / fname).write_bytes(zf.read(member))
                images.append({"id": img_id, "file_name": fname, "width": 1920, "height": 1080})
                for cat, x, y, w, h in bxs:
                    anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cat, "bbox": [x, y, w, h],
                                 "area": w * h, "iscrowd": 0})
            coco = {"images": images, "annotations": anns,
                    "categories": [{"id": 0, "name": "objects", "supercategory": "none"}]
                    + [{"id": i + 1, "name": n, "supercategory": "objects"} for i, n in enumerate(CATEGORIES)]}
            (d / "_annotations.coco.json").write_text(json.dumps(coco))
            counts[part] = {"images": len(images), "boxes": len(anns)}
    test = WORK / "data" / "test"
    if test.exists():
        shutil.rmtree(test)
    test.symlink_to(WORK / "data" / "valid", target_is_directory=True)
    print(f"prepared {counts} in {time.time() - t0:.0f} s", flush=True)
    return counts


def train(epochs: int = 12, batch_size: int = 4, grad_accum_steps: int = 4, lr: float = 1e-4,
          resolution: int = 0, **extra) -> str:
    """Fine-tune RF-DETR Medium on ``WORK/data``; resumes from ``WORK/out/checkpoint.pth`` when it exists.
    ``resolution`` 0 keeps the model's default. Returns the output folder."""
    import rfdetr

    out = WORK / "out"
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(dataset_dir=str(WORK / "data"), epochs=epochs, batch_size=batch_size,
              grad_accum_steps=grad_accum_steps, lr=lr, output_dir=str(out), **extra)
    ckpt = out / "checkpoint.pth"
    if ckpt.exists():
        kw["resume"] = str(ckpt)
    model = rfdetr.RFDETRMedium(**({"resolution": resolution} if resolution else {}))
    t0 = time.time()
    model.train(**kw)
    print(f"train returned after {time.time() - t0:.0f} s; outputs: {sorted(p.name for p in out.iterdir())}", flush=True)
    return str(out)
