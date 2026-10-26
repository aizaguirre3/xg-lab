"""The Colab clip runner's control flow, with a fake ``colab`` CLI (nothing is run remotely)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "colab" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses on Python 3.14 look the module up here
    spec.loader.exec_module(mod)
    return mod


class FakeColab:
    """Records every CLI call; ``fail_on`` makes one exec raise."""

    def __init__(self, out_dir: Path, fail_on: str | None = None):
        self.calls: list[tuple[list[str], str | None]] = []
        self.out_dir = out_dir
        self.fail_on = fail_on
        self.stopped = False

    def __call__(self, cmd, stdin_text=None):
        self.calls.append((list(cmd), stdin_text))
        rc = load_module("run_clips").CmdResult
        sub = cmd[1] if len(cmd) > 1 else ""
        if sub == "exec" and self.fail_on and stdin_text and self.fail_on in stdin_text:
            return rc(1, "", "boom")
        if sub == "download":
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[-1]).write_text(json.dumps({"info": {"name": "x"}}))
        if sub == "stop":
            self.stopped = True
        if sub == "sessions":
            return rc(0, "(no sessions)" if self.stopped else "varlab-clips RUNNING", "")
        return rc(0, "ok", "")


def test_run_processes_each_clip_then_downloads_and_stops(tmp_path: Path, monkeypatch):
    rc = load_module("run_clips")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = FakeColab(tmp_path)
    args = rc.parse_args(["--clips", "SNGS-021", "SNGS-022", "--split", "valid", "--ref", "abc123",
                          "--out", str(tmp_path / "out"), "--session", "varlab-clips", "--budget-min", "45"])
    code = rc.run(args, runner=fake)
    assert code == 0
    subs = [c[0][1] for c in fake.calls]
    assert subs[0] == "new" and subs[-2] == "stop" and subs[-1] == "sessions"
    # install the pins, restart the kernel, then define the worker and run setup
    assert subs.index("install") < subs.index("restart-kernel") < subs.index("exec")
    assert [c for c, _ in fake.calls if c[1] == "install"][0][-1].endswith("requirements-vision.txt")
    execs = [s for c, s in fake.calls if c[1] == "exec" and s]
    assert any("setup(" in s and "abc123" in s for s in execs)
    assert [s for s in execs if "process_clip(" in s] == [
        "process_clip('SNGS-021', split='valid')", "process_clip('SNGS-022', split='valid')"]
    # one download per clip, after that clip's exec, before the next clip starts
    order = [(c[1], (s or "") + " ".join(c)) for c, s in fake.calls]
    i21 = next(i for i, (k, s) in enumerate(order) if k == "exec" and "SNGS-021" in s)
    d21 = next(i for i, (k, s) in enumerate(order) if k == "download" and "SNGS-021" in s)
    i22 = next(i for i, (k, s) in enumerate(order) if k == "exec" and "SNGS-022" in s)
    assert i21 < d21 < i22
    assert (tmp_path / "out" / "SNGS-021" / "Labels-GameState.json").exists()
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest[-1]["done"] == ["SNGS-021", "SNGS-022"] and manifest[-1]["remaining"] == []


def test_session_is_stopped_even_when_a_clip_fails(tmp_path: Path, monkeypatch):
    rc = load_module("run_clips")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = FakeColab(tmp_path, fail_on="SNGS-022")
    args = rc.parse_args(["--clips", "SNGS-021", "SNGS-022", "SNGS-023", "--ref", "abc", "--out", str(tmp_path / "o")])
    code = rc.run(args, runner=fake)
    assert code != 0 and fake.stopped
    manifest = json.loads((tmp_path / "o" / "run_manifest.json").read_text())
    assert manifest[-1]["done"] == ["SNGS-021"] and manifest[-1]["failed"] == ["SNGS-022"]
    assert manifest[-1]["remaining"] == ["SNGS-023"]


def test_budget_stops_before_the_next_clip(tmp_path: Path, monkeypatch):
    rc = load_module("run_clips")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = FakeColab(tmp_path)
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 600.0  # every call advances 10 minutes
        return clock["t"]

    monkeypatch.setattr(rc.time, "monotonic", fake_time)
    args = rc.parse_args(["--clips", "A", "B", "C", "D", "--ref", "abc", "--out", str(tmp_path / "o"),
                          "--budget-min", "40", "--clip-estimate-s", "60", "--stop-margin-s", "60"])
    code = rc.run(args, runner=fake)
    assert code == 2 and fake.stopped
    manifest = json.loads((tmp_path / "o" / "run_manifest.json").read_text())
    assert manifest[-1]["remaining"]  # some clips left for the next run
    assert len(manifest[-1]["done"]) + len(manifest[-1]["remaining"]) == 4


def test_resume_skips_clips_already_downloaded(tmp_path: Path, monkeypatch):
    rc = load_module("run_clips")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    out = tmp_path / "o"
    (out / "A").mkdir(parents=True)
    (out / "A" / "Labels-GameState.json").write_text("{}")
    fake = FakeColab(tmp_path)
    args = rc.parse_args(["--clips", "A", "B", "--ref", "abc", "--out", str(out), "--resume"])
    assert rc.run(args, runner=fake) == 0
    execs = [s for c, s in fake.calls if c[1] == "exec" and s and "process_clip(" in s]
    assert execs == ["process_clip('B', split='valid')"]


def test_token_goes_through_a_file_upload_never_the_exec_stream(tmp_path: Path, monkeypatch):
    rc = load_module("run_clips")
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETVALUE")
    fake = FakeColab(tmp_path)
    args = rc.parse_args(["--clips", "A", "--ref", "abc", "--out", str(tmp_path / "o")])
    assert rc.run(args, runner=fake) == 0
    uploads = [c for c, _ in fake.calls if c[1] == "upload"]
    assert len(uploads) == 1 and uploads[0][-1] == rc.REMOTE_TOKEN
    for c, s in fake.calls:
        assert "hf_SECRETVALUE" not in " ".join(c) and "hf_SECRETVALUE" not in (s or "")
    assert not list(tmp_path.glob("**/*token*"))  # the temp file is gone


def test_dry_run_prints_commands_without_a_runner(capsys):
    rc = load_module("run_clips")
    code = rc.main(["--clips", "A", "--ref", "abc", "--out", "/tmp/ignored", "--dry-run"])
    captured = capsys.readouterr().out
    assert code == 0 and "colab new" in captured and "process_clip('A'" in captured and "colab stop" in captured
    assert "colab install" in captured and "colab restart-kernel" in captured


def test_all_lists_the_clips_from_the_dataset_zip_and_passes_the_layout(tmp_path: Path, monkeypatch):
    rc = load_module("run_clips")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    seen = []
    monkeypatch.setattr(rc, "list_dataset_clips", lambda ds, split, rev: seen.append((ds, split, rev)) or
                        ["SNGS-021", "SNGS-022"])
    fake = FakeColab(tmp_path)
    args = rc.parse_args(["--all", "--ref", "abc123", "--out", str(tmp_path / "out")])
    assert rc.run(args, runner=fake) == 0
    assert seen == [("SoccerNet/SN-GSR-2025", "valid", "main")]
    execs = [s for c, s in fake.calls if c[1] == "exec" and s]
    assert any("setup(" in s and "hf_layout='zip'" in s for s in execs)
    assert [s for s in execs if "process_clip(" in s] == [
        "process_clip('SNGS-021', split='valid')", "process_clip('SNGS-022', split='valid')"]


def test_all_and_clips_are_exclusive_and_all_needs_the_zip_layout(tmp_path: Path):
    rc = load_module("run_clips")
    with pytest.raises(SystemExit):
        rc.parse_args(["--all", "--clips", "A", "--ref", "x", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        rc.parse_args(["--ref", "x", "--out", str(tmp_path)])  # one of the two is required
    args = rc.parse_args(["--all", "--hf-layout", "folders", "--ref", "x", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        rc.run(args, runner=FakeColab(tmp_path))


def test_a_transient_download_failure_is_retried_not_fatal(tmp_path: Path, monkeypatch):
    rc = load_module("run_clips")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)

    class Flaky(FakeColab):
        def __init__(self, out_dir):
            super().__init__(out_dir)
            self.failed = 0

        def __call__(self, cmd, stdin_text=None):
            if len(cmd) > 1 and cmd[1] == "download" and self.failed < 2:
                self.failed += 1
                self.calls.append((list(cmd), stdin_text))
                return load_module("run_clips").CmdResult(1, "", "")
            return super().__call__(cmd, stdin_text)

    fake = Flaky(tmp_path)
    args = rc.parse_args(["--clips", "SNGS-021", "--ref", "abc", "--out", str(tmp_path / "out")])
    assert rc.run(args, runner=fake) == 0
    assert fake.failed == 2 and (tmp_path / "out" / "SNGS-021" / "Labels-GameState.json").exists()
