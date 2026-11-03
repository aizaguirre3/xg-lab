"""Colab-side functions for training the pitch-line segmenter (``varlab_vision/line_seg.py``).

Sent to a fresh kernel with ``colab exec -f`` by ``colab/train_line_seg.py``, which then calls

    setup(ref)                      # clone the repo at the pinned commit
    prepare(every=5, holdout=[...]) # download train.zip, write the training set (JPEG + label PNG)
    train(minutes=8, ...)           # train for a while, resuming from the last checkpoint

and downloads ``/content/lineseg/ckpt.pt`` and ``log.jsonl`` after every ``train`` call, so a
killed session loses at most one chunk. Nothing here reads a token: the dataset is public.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/content/lineseg")
REPO_DIR = Path("/content/xg-lab")


def setup(ref: str, repo: str = "https://github.com/aizaguirre3/xg-lab.git") -> None:
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "--quiet", repo, str(REPO_DIR)], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--quiet", "origin", ref], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--quiet", ref], check=True)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    WORK.mkdir(parents=True, exist_ok=True)
    import torch

    print("ref", subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
          "torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)


def _download(url: str, dest: Path) -> None:
    import urllib.request

    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(".part")
    t0, done = time.time(), 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "varlab-lineseg/1"}), timeout=120) as r, \
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


def _prepare_clip(args):
    zip_path, clip, prefix, every = args
    import zipfile

    import cv2
    import numpy as np

    from varlab_vision.line_seg import INPUT_SIZE, rasterise_lines

    out = WORK / "data" / clip
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        raw = json.loads(zf.read(prefix + "Labels-GameState.json"))
        name_of = {im["image_id"]: im["file_name"] for im in raw["images"]}
        lines = {name_of[a["image_id"]]: a["lines"] for a in raw["annotations"] if a.get("supercategory") == "pitch"}
        n = 0
        for k, fname in enumerate(sorted(lines)):
            if k % every:
                continue
            img = cv2.imdecode(np.frombuffer(zf.read(prefix + "img1/" + fname), np.uint8), cv2.IMREAD_COLOR)
            small = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
            lab = rasterise_lines(lines[fname], INPUT_SIZE[0], INPUT_SIZE[1], thickness=3)
            stem = fname.rsplit(".", 1)[0]
            cv2.imwrite(str(out / f"{stem}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / f"{stem}.png"), lab)
            n += 1
    return clip, n


def prepare(every: int = 5, holdout=(), split: str = "train", workers: int = 8,
            dataset: str = "SoccerNet/SN-GSR-2025") -> dict:
    """Download ``<split>.zip`` and write every ``every``-th labelled frame of each clip, resized to the
    network input, with its label map; clips in ``holdout`` go to the validation list."""
    from multiprocessing import get_context

    from varlab_vision.remote_zip import clip_index, hf_dataset_file_url

    import zipfile

    zip_path = WORK / f"{split}.zip"
    t0 = time.time()
    _download(hf_dataset_file_url(dataset, f"{split}.zip"), zip_path)
    print(f"downloaded {zip_path.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        index = clip_index(zf)
    jobs = [(str(zip_path), c, p, every) for c, p in index.items()]
    with get_context("fork").Pool(workers) as pool:
        counts = dict(pool.map(_prepare_clip, jobs))
    split_def = {"train": sorted(c for c in counts if c not in set(holdout)), "val": sorted(c for c in counts if c in set(holdout)),
                 "every": every, "frames": counts}
    (WORK / "split.json").write_text(json.dumps(split_def, indent=1))
    print(f"prepared {sum(counts.values())} frames of {len(counts)} clips in {time.time() - t0:.0f} s; "
          f"{len(split_def['train'])} train / {len(split_def['val'])} val clips", flush=True)
    return {"frames": sum(counts.values()), "clips": len(counts)}


def _flip_table():
    from varlab_vision.line_seg import CLASSES

    def mirror(name: str) -> str:
        swap = {"left": "right", "right": "left"}
        return " ".join(swap.get(w, w) for w in name.split(" "))

    table = [0] * (len(CLASSES) + 1)
    for k, c in enumerate(CLASSES):
        table[k + 1] = CLASSES.index(mirror(c)) + 1
    return table


def _worker_init(_k):
    import cv2

    cv2.setNumThreads(0)  # one decode thread per loader process: a few vCPUs are shared by all of them


def train(minutes: float = 8.0, epochs: int = 30, batch: int = 16, lr: float = 1e-3, workers: int = 8,
          bg_weight: float = 0.05, seed: int = 0, steps_per_epoch: int = 200) -> dict:
    """Train (resuming from ``ckpt.pt``) until ``minutes`` have passed or ``epochs`` are done. An "epoch"
    here is ``steps_per_epoch`` random batches, so the time limit is checked every minute or two and a
    checkpoint is written that often; ``progress.txt`` is rewritten every 20 steps (readable with
    ``colab download`` while the kernel is busy). Validation: per-class IoU on the held-out clips."""
    import random

    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, RandomSampler

    from varlab_vision.line_seg import MEAN, N_CLASSES, STD, build_model

    torch.manual_seed(seed)
    split = json.loads((WORK / "split.json").read_text())
    flip = np.array(_flip_table(), np.uint8)

    class Frames(Dataset):
        def __init__(self, clips, augment):
            self.items = sorted(p for c in clips for p in (WORK / "data" / c).glob("*.jpg"))
            self.augment = augment

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p = self.items[i]
            img = cv2.imread(str(p))[:, :, ::-1].astype(np.float32) / 255.0
            lab = cv2.imread(str(p.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
            if self.augment:
                if random.random() < 0.5:  # a mirrored pitch: left and right swap names
                    img, lab = img[:, ::-1], flip[lab[:, ::-1]]
                img = img * random.uniform(0.7, 1.3) + random.uniform(-0.1, 0.1)
                img = img * np.array([random.uniform(0.9, 1.1) for _ in range(3)], np.float32)
            img = ((np.clip(img, 0, 1) - MEAN) / STD).transpose(2, 0, 1)
            return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(lab)).long()

    dev = "cuda"
    model = build_model(pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ckpt = WORK / "ckpt.pt"
    state = {"epoch": 0, "best_miou": -1.0}
    if ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        state = saved["state"]
    workers = max(1, min(workers, (os.cpu_count() or 2)))
    train_set = Frames(split["train"], True)
    tr = DataLoader(train_set, batch_size=batch, num_workers=workers, drop_last=True, pin_memory=True,
                    worker_init_fn=_worker_init,
                    sampler=RandomSampler(train_set, replacement=True, num_samples=steps_per_epoch * batch))
    va = DataLoader(Frames(split["val"], False), batch_size=batch, shuffle=False, num_workers=workers,
                    worker_init_fn=_worker_init)
    print(f"{len(train_set)} training frames, {workers} loader workers, {os.cpu_count()} vCPUs", flush=True)
    weight = torch.ones(N_CLASSES, device=dev)
    weight[0] = bg_weight
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    sched_total = epochs * len(tr)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()
    log = []
    while state["epoch"] < epochs and (time.time() - t0) < minutes * 60:
        model.train()
        tot, nb = 0.0, 0
        for x, y in tr:
            step = state["epoch"] * len(tr) + nb
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + np.cos(np.pi * min(1.0, step / sched_total)))
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            nb += 1
            if nb % 20 == 0:
                (WORK / "progress.txt").write_text(f"epoch {state['epoch'] + 1} step {nb}/{len(tr)} "
                                                   f"loss {tot / nb:.4f} {time.time() - t0:.0f} s\n")
        model.eval()
        inter = torch.zeros(N_CLASSES, device=dev)
        union = torch.zeros(N_CLASSES, device=dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for x, y in va:
                pred = model(x.to(dev)).argmax(1)
                y = y.to(dev)
                for c in range(1, N_CLASSES):
                    pc, yc = pred == c, y == c
                    inter[c] += (pc & yc).sum()
                    union[c] += (pc | yc).sum()
        present = union[1:] > 0
        iou = (inter[1:] / union[1:].clamp(min=1)).cpu().numpy()
        miou = float(iou[present.cpu().numpy()].mean()) if bool(present.any()) else 0.0
        state["epoch"] += 1
        row = {"epoch": state["epoch"], "loss": round(tot / max(1, nb), 4), "val_miou": round(miou, 4),
               "iou": [round(float(v), 3) for v in iou], "seconds": round(time.time() - t0, 1)}
        log.append(row)
        with open(WORK / "log.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if miou > state["best_miou"]:
            state["best_miou"] = miou
            torch.save({"model": model.state_dict()}, WORK / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state}, ckpt)
        print(json.dumps({k: row[k] for k in ("epoch", "loss", "val_miou", "seconds")}), flush=True)
    return {"epoch": state["epoch"], "best_miou": state["best_miou"], "log": log}
