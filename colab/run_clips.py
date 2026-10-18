#!/usr/bin/env python3
"""Mac-side runner: process GSR clips on a Colab GPU, one at a time, pulling each result back at once.

    python colab/run_clips.py --clips SNGS-021 SNGS-022 --split valid --ref <commit-sha> \
        --out data/gsr_pred/valid --session varlab-clips --gpu L4 --budget-min 45

Pattern (the reason this is not one big ``colab run``): a server-side termination
of a CLI session has happened at 1 h and 3.4 h, so each run keeps a budget well
under one hour, and the result of EVERY clip is downloaded to the Mac right after
that clip finishes, so a kill loses at most the clip in flight. Re-run with
``--resume`` to continue from where the previous run stopped.

Flow: ``colab new`` -> (optional) upload the Hugging Face token as a file ->
``colab exec -f colab/vision_worker.py`` (defines ``setup`` and ``process_clip`` in
the kernel) -> ``setup(ref=...)`` (clone the repo at the pinned commit, install the
pinned requirements, download weights) -> per clip: ``process_clip(...)`` then
``colab download`` of its ``Labels-GameState.json`` and ``report.json`` ->
``colab stop`` in a ``finally`` -> ``colab sessions`` printed so the operator can
confirm nothing is left running.

Secrets: the token is read from ``HF_TOKEN`` in the Mac environment, written to a
0600 temp file, uploaded with ``colab upload`` and deleted. It never appears in a
command line or in the code sent through ``colab exec`` (the CLI logs those under
``~/.config/colab-cli/history``).

NOT RUN ON REAL DATA YET: SoccerNet access had not arrived when this was written.
The layout of the clips inside the Hugging Face dataset repository is unverified;
see ``vision_worker.py`` (``clip_pattern``). ``--dry-run`` prints the commands.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_COLAB = os.environ.get("COLAB_CLI", os.path.expanduser("~/.local/bin/colab"))
DEFAULT_REPO = "https://github.com/aizaguirre3/xg-lab.git"
REMOTE_OUT = "/content/varlab_out"
REMOTE_TOKEN = "/content/.hf_token"
WORKER = HERE / "vision_worker.py"


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[List[str], Optional[str]], CmdResult]


def subprocess_runner(cmd: List[str], stdin_text: Optional[str] = None) -> CmdResult:
    p = subprocess.run(cmd, input=stdin_text, text=True, capture_output=True)
    if p.stdout:
        sys.stdout.write(p.stdout if p.stdout.endswith("\n") else p.stdout + "\n")
    if p.stderr:
        sys.stderr.write(p.stderr if p.stderr.endswith("\n") else p.stderr + "\n")
    return CmdResult(p.returncode, p.stdout, p.stderr)


def printing_runner(cmd: List[str], stdin_text: Optional[str] = None) -> CmdResult:
    """Dry run: print what would be executed."""
    line = " ".join(shlex.quote(c) for c in cmd)
    if stdin_text is not None:
        line += "  <<< " + stdin_text.replace("\n", "\\n")
    print(line)
    return CmdResult(0, "", "")


class ColabSession:
    def __init__(self, name: str, gpu: str, runner: Runner, colab: str = DEFAULT_COLAB):
        self.name, self.gpu, self.runner, self.colab = name, gpu, runner, colab

    def _run(self, *args: str, stdin_text: Optional[str] = None, check: bool = True) -> CmdResult:
        cmd = [self.colab, *args]
        res = self.runner(cmd, stdin_text)
        if check and res.returncode != 0:
            raise RuntimeError(f"colab {args[0]} failed (rc={res.returncode}): {res.stderr.strip()[-500:]}")
        return res

    def new(self) -> None:
        self._run("new", "-s", self.name, "--gpu", self.gpu)

    def stop(self) -> CmdResult:
        return self._run("stop", "-s", self.name, check=False)

    def sessions(self) -> str:
        return self._run("sessions", check=False).stdout

    def exec_file(self, path: Path, timeout: float) -> CmdResult:
        return self._run("exec", "-s", self.name, "-f", str(path), "--timeout", str(timeout))

    def exec_code(self, code: str, timeout: float) -> CmdResult:
        return self._run("exec", "-s", self.name, "--timeout", str(timeout), stdin_text=code)

    def upload(self, local: Path, remote: str) -> None:
        self._run("upload", "-s", self.name, str(local), remote)

    def download(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._run("download", "-s", self.name, remote, str(local))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", nargs="+", required=True, help="clip names, e.g. SNGS-021")
    p.add_argument("--split", default="valid", choices=["train", "valid", "test", "challenge"])
    p.add_argument("--ref", required=True, help="git commit (or branch) of this repo to run on the VM; pin a SHA")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--out", required=True, help="local folder; one sub-folder per clip")
    p.add_argument("--session", default="varlab-clips")
    p.add_argument("--gpu", default="L4", choices=["T4", "L4"])
    p.add_argument("--budget-min", type=float, default=45.0, help="hard wall-clock budget for this run")
    p.add_argument("--clip-estimate-s", type=float, default=120.0, help="assumed duration of a clip before any is timed")
    p.add_argument("--stop-margin-s", type=float, default=120.0, help="time kept free for the stop/download tail")
    p.add_argument("--setup-timeout", type=float, default=1200.0)
    p.add_argument("--clip-timeout", type=float, default=900.0)
    p.add_argument("--detector-size", default="medium", choices=["nano", "small", "medium", "large"])
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--hf-dataset", default="SoccerNet/SN-GSR-2025", help="Hugging Face dataset repo id")
    p.add_argument("--hf-revision", default="main", help="pin to a dataset commit once known")
    p.add_argument("--clip-pattern", default="{split}/{clip}/*", help="allow_patterns template inside the dataset repo")
    p.add_argument("--resume", action="store_true", help="skip clips whose Labels-GameState.json is already local")
    p.add_argument("--colab", default=DEFAULT_COLAB)
    p.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    return p.parse_args(argv)


def pending_clips(clips: Sequence[str], out: Path, resume: bool) -> List[str]:
    if not resume:
        return list(clips)
    return [c for c in clips if not (out / c / "Labels-GameState.json").is_file()]


def _write_manifest(out: Path, entry: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_manifest.json"
    hist = json.loads(path.read_text()) if path.is_file() else []
    hist.append(entry)
    path.write_text(json.dumps(hist, indent=1))


def run(args: argparse.Namespace, runner: Optional[Runner] = None) -> int:
    dry = args.dry_run
    runner = runner or (printing_runner if dry else subprocess_runner)
    out = Path(args.out)
    todo = pending_clips(args.clips, out, args.resume)
    budget_s = args.budget_min * 60.0
    session = ColabSession(args.session, args.gpu, runner, args.colab)
    done: List[str] = []; failed: List[str] = []; durations: List[float] = []
    t_start = time.monotonic()
    exit_code = 0
    print(f"[run_clips] {len(todo)} clip(s) to process, budget {args.budget_min:.0f} min, session {args.session} ({args.gpu})")
    if not todo:
        print("[run_clips] nothing to do")
        return 0
    session.new()
    try:
        token = os.environ.get("HF_TOKEN")
        if token:
            fd, tmp = tempfile.mkstemp(prefix="hf_", suffix=".txt")
            try:
                os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
                with os.fdopen(fd, "w") as fh:
                    fh.write(token)
                session.upload(Path(tmp), REMOTE_TOKEN)
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        else:
            print("[run_clips] HF_TOKEN not set: the worker will try Colab Secrets, else gated downloads fail")
        session.exec_file(WORKER, timeout=args.setup_timeout)
        setup_call = (f"setup(ref={args.ref!r}, repo={args.repo!r}, detector_size={args.detector_size!r}, "
                      f"with_pose={not args.no_pose!r}, hf_dataset={args.hf_dataset!r}, "
                      f"hf_revision={args.hf_revision!r}, clip_pattern={args.clip_pattern!r})")
        session.exec_code(setup_call, timeout=args.setup_timeout)
        for clip in todo:
            elapsed = time.monotonic() - t_start
            est = max(durations) if durations else args.clip_estimate_s
            if elapsed + est + args.stop_margin_s > budget_s:
                print(f"[run_clips] budget: {elapsed/60:.1f} min used, stopping before {clip}")
                exit_code = 2
                break
            t0 = time.monotonic()
            try:
                session.exec_code(f"process_clip({clip!r}, split={args.split!r})", timeout=args.clip_timeout)
                session.download(f"{REMOTE_OUT}/{args.split}/{clip}/Labels-GameState.json", out / clip / "Labels-GameState.json")
                session.download(f"{REMOTE_OUT}/{args.split}/{clip}/report.json", out / clip / "report.json")
                if not dry and not (out / clip / "Labels-GameState.json").is_file():
                    raise RuntimeError(f"download of {clip} produced no file")
            except Exception as e:  # noqa: BLE001
                print(f"[run_clips] {clip} FAILED: {e}")
                failed.append(clip)
                exit_code = 1
                break
            durations.append(time.monotonic() - t0)
            done.append(clip)
            print(f"[run_clips] {clip} done in {durations[-1]:.0f} s ({len(done)}/{len(todo)})")
    finally:
        session.stop()
        listing = session.sessions()
        print("[run_clips] colab sessions after stop:\n" + (listing.strip() or "(empty)"))
    remaining = [c for c in todo if c not in done and c not in failed]
    if not dry:
        _write_manifest(out, {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "ref": args.ref, "split": args.split,
                              "done": done, "failed": failed, "remaining": remaining,
                              "seconds_per_clip": [round(d, 1) for d in durations], "exit_code": exit_code})
    print(f"[run_clips] done {len(done)}, failed {len(failed)}, remaining {len(remaining)}")
    if remaining:
        print("[run_clips] re-run with --resume to continue")
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
