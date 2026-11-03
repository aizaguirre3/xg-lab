#!/usr/bin/env python3
"""Mac-side runner: fine-tune RF-DETR Medium on the GSR boxes on a Colab GPU, a few epochs per session.

    python colab/train_detector.py --ref <commit-sha> --out data/models/detector --gpu L4 \
        --holdout SNGS-060 SNGS-064 SNGS-072 SNGS-099 SNGS-103 --epochs 4

One session: ``colab new`` -> install the pinned ``rfdetr`` -> ``restart-kernel`` (installing and
importing in one kernel mixes numpy versions) -> ``colab exec -f detector_worker.py`` -> ``setup``
(prints the package's training API) -> ``prepare`` -> upload the last checkpoint if there is one
-> ``train(epochs=...)`` -> download the checkpoints -> ``colab stop`` in a ``finally``. The
package trains every epoch it is given in one call, so ``--epochs`` is the total to reach this
session: run again with a higher number to resume (``checkpoint.pth`` goes back up first).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_clips import ColabSession, printing_runner, subprocess_runner  # noqa: E402

WORKER = HERE / "detector_worker.py"
REMOTE = "/content/det/out"
RFDETR_PIN = "rfdetr==1.11.2"  # as in requirements-vision.txt
OUTPUTS = ("checkpoint.pth", "checkpoint_best_total.pth", "log.txt")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="commit to clone on the VM (must be pushed)")
    p.add_argument("--out", required=True, help="local folder for the checkpoints (gitignored)")
    p.add_argument("--holdout", nargs="*", default=[])
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--epochs", type=int, required=True, help="total epochs to have trained when this session ends")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resolution", type=int, default=0)
    p.add_argument("--train-timeout-min", type=float, default=40.0)
    p.add_argument("--session", default="varlab-detector")
    p.add_argument("--gpu", default="L4")
    p.add_argument("--setup-only", action="store_true", help="install, report the API, stop")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    session = ColabSession(args.session, args.gpu, printing_runner if args.dry_run else subprocess_runner)
    out = Path(args.out)
    session.new()
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(RFDETR_PIN + "\n")
            req = Path(fh.name)
        try:
            session.install(req)
        finally:
            req.unlink(missing_ok=True)
        session.restart_kernel()
        session.exec_file(WORKER, timeout=120)
        session.exec_code(f"setup({args.ref!r})", timeout=900)
        if args.setup_only:
            return 0
        session.exec_code(f"prepare(every={args.every}, holdout={list(args.holdout)!r})", timeout=1800)
        if (out / "checkpoint.pth").is_file():
            session.exec_code(f"import os; os.makedirs({REMOTE!r}, exist_ok=True)", timeout=60)
            session.upload(out / "checkpoint.pth", f"{REMOTE}/checkpoint.pth")
        session.exec_code(f"train(epochs={args.epochs}, batch_size={args.batch_size}, grad_accum_steps={args.grad_accum}, "
                          f"lr={args.lr}, resolution={args.resolution})", timeout=args.train_timeout_min * 60)
        for name in OUTPUTS:
            try:
                session.download(f"{REMOTE}/{name}", out / name)
            except RuntimeError as e:
                print(f"[train_detector] {name}: {e}")
        try:
            session.exec_code("check()", timeout=600)  # the class ids the model gives, for the pipeline's mapping
        except RuntimeError as e:
            print(f"[train_detector] check failed: {e}")
    finally:
        session.stop()
        print(session.sessions())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
