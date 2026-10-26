# VAR-Lab M4: vision environment on a Colab L4

Measured on 2026-10-17 (VM clock 2026-10-18 03:50 UTC) with

```bash
colab run --gpu L4 --timeout 600 -s varlab-vision-env colab/vision_env_probe.py --ref 8527356540bab8af74324cb1709549b428f0c523
```

`colab run` provisioned the VM, ran the probe and released it; `colab sessions`
listed no session afterwards. The probe clones this repository at the pinned
commit, installs `requirements-vision.txt`, downloads the weights from their
official hosts and measures each stage on a **synthetic clip** (30 frames of
coloured rectangles on green at 1920x1080, 11 "people" per frame). No SoccerNet
data was used; the rectangles only exercise the code path, so RF-DETR's zero
detections on them are expected and say nothing about accuracy.

## Machine and installed versions

| Item | Value |
|---|---|
| GPU | NVIDIA L4, 23034 MiB, driver 580.82.07 |
| CPU / RAM / disk | 12 vCPU, 55.5 GB RAM (disk 193 GB free under `/content` per the 2026-10-17 probe in `tools/colab_gpu_probe.py`) |
| Python | 3.13.15, Linux 6.6.122 (glibc 2.39) |
| torch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 (VM build, CUDA 13.0) |
| numpy | 2.1.3 (the VM's; kept on purpose, see `requirements-vision.txt`) |
| rfdetr / supervision / trackers | 1.11.2 / 0.31.0 / 2.6.1 |
| transformers / huggingface-hub | 5.20.1 / 2.2.0 |
| rtmlib / onnxruntime-gpu | 0.0.16 / 1.31.0 (providers: TensorRT, CUDA, CPU) |
| opencv / pandas / scipy | 5.0.0 / 3.0.6 / 1.18.1 |

## Setup time and download sizes

| Step | Time | Size | Host |
|---|---|---|---|
| `git clone` + checkout at the pinned SHA | 2.2 s | | github.com |
| `pip install -r requirements-vision.txt` | 40.0 s | | pypi.org |
| `pip install --no-deps --force-reinstall onnxruntime-gpu==1.31.0` | 7.2 s | | pypi.org |
| RF-DETR Medium COCO checkpoint (`rf-detr-medium.pth`, md5-checked by rfdetr) | 26.1 s incl. model build | 386.2 MB | storage.googleapis.com/rfdetr |
| SigLIP `google/siglip-base-patch16-224` @ `7fd15f06` (config + safetensors) | 7.3 s incl. model load | 775.0 MB | huggingface.co |
| RTMPose-m body7 ONNX (rtmlib zip) | 1.8 s | 51.8 MB | download.openmmlab.com |
| Stage 2 (weights + synthetic clip + all measurements) | 63.6 s | | |
| Whole `colab run` including VM provisioning and release | 2 min 7 s wall clock on the Mac (push at 23:48:59, exit at 23:51:06 local) | | |

Weights total about 1.21 GB per fresh VM; a `colab run` therefore spends
roughly a minute on packages and weights before any clip is processed.

## Per-stage throughput (L4, fp32, 1920x1080, no `optimize_for_inference`)

| Stage | Backend | Throughput | Notes |
|---|---|---|---|
| Detection | RF-DETR Medium (576 px input) | **46.1 fps** | 30 frames after 2 warm-up frames; 0 detections on the rectangles, as expected |
| Tracking | ByteTrack (`trackers`) | 822.6 fps | 11 ground-truth boxes per frame |
| Team embeddings | SigLIP base, batch 64 | 325.9 crops/s (29.6 frames/s at 11 crops per frame) | 330 crops, 768-d; the pipeline samples at most 20 crops per track, so this is negligible per clip |
| Pose | RTMPose-m via rtmlib, CUDA provider | 19.6 fps = 215.6 people/s | one ONNX run per box; the slowest stage with 11 people, about 2x slower with 22 |
| Full production pipeline, end to end | RF-DETR + ByteTrack + SigLIP + RTMPose | 35.9 fps | nothing downstream fired because there were no detections |
| Smoke pipeline (CPU backends on the VM) | colour blobs + ByteTrack + mean colour | 41.4 fps | 330 detections, 11 tracks |

Peak GPU memory: torch max allocated 975.8 MB, max reserved 1206.0 MB;
`nvidia-smi` 1590 MiB used of 23034 MiB (includes ONNX Runtime's CUDA context).
The L4 has an order of magnitude of headroom for batching or a larger detector.

## Closed loop on the synthetic clip

The smoke pipeline's output was read back with the loader and scored with the
GS-HOTA adapter against the clip's known ground truth: GS-HOTA 0.8837 over all
roles, 0.9231 for players only (DetA 0.8815, AssA 0.9667). The gap to 1.0 is the
expected one: the referee rectangle is predicted as a player (one FP and one FN
per frame) and ByteTrack confirms identities from the second frame. The same
code path on the Mac CPU gives 0.866 / 0.926 with the greedy tracker.

## What this means for the 49 validation clips

A GSR clip is 750 frames at 1920x1080. At the rates above a clip costs about
16 s of detection, 40 to 80 s of pose (11 to 22 people), a few seconds of
tracking, embedding and JPEG decoding, plus the clip download: roughly 1.5 to 2
minutes per clip, so 49 clips are 1.5 to 2 hours of GPU time. `colab/run_clips.py`
keeps each run under a 45-minute budget and downloads every clip's output as soon
as it is written, so two to three runs with `--resume` cover the split and a
server-side termination loses at most one clip. Pose is the first stage to
batch when that matters.

Not measured here (no data): real detection quality, team-cluster purity,
pitch calibration on real broadcast frames (so far only synthetic clips, see
the calibration section below), jersey recognition (not implemented; GS-HOTA
requires it), and the layout of the clips inside the Hugging Face dataset
repository.

## Camera calibration (added 2026-10-19)

Measured on 2026-10-19 on the Mac CPU (Apple M5), not on the Colab L4, and
only on synthetic clips. The synthetic clip renders the pitch markings under a
known pinhole broadcast camera, so the calibration can be scored against ground
truth. The stage (`varlab_vision/pitch_model.py`, `varlab_vision/calibration.py`) is
classical and uses only numpy, SciPy and OpenCV. A frame counts as calibrated
only when at least 4 lines match, with at least two from each family;
otherwise it is reported uncalibrated with null pitch coordinates.

| Check | Result (synthetic clips) |
|---|---|
| Global search accuracy | 24 scenes (6 camera positions x wide and medium views x 1280x720 and 1920x1080), all calibrated |
| Global search, worst keypoint error | 0.040 m (wide view at 1280x720); at most 0.021 m in the other three groups |
| Tracking, 0.8 m/frame pan | within 0.060 m at the keypoints |
| Tracking, steady pans of 1.5 and 3.0 m/frame | followed with no lag (the earlier linear blend lagged 1.8 m behind a 1.5 m/frame pan) |
| Tracking, 6 m/frame pan | falls back to the global search and stays within 0.5 m |
| Shot-cut rule | flagged when the grey-thumbnail difference is at least 25 (0-255 scale) or the hue-saturation histogram distance is at least 0.2 |
| Within a shot | grey difference 2.0 to 5.8; histogram distance 0.02 to 0.05 |
| At a cut | histogram distance 0.33 to 0.89 |
| Line extraction | 44 to 62 ms per frame |
| Global search | 0.6 to 0.84 s (first frame and after a cut) |
| Tracked frame | 58 to 88 ms at both 1280x720 and 1920x1080; about one minute of CPU per 750-frame 1080p clip |
| Pipeline, wide camera at 1280x720, 8 frames, players only | GS-HOTA 1.0 on four scenes |
| Calibration foot-position error | at most 0.013 m (ground-truth boxes through the written homographies) |
| End-to-end foot error | up to 0.65 m; the CPU smoke detector's boxes sit 0 or 1 px below the drawn rectangles, and one pixel at the far touchline of a 720p wide shot spans about half a metre |
| Tests | `tests/vision` 87 passed; `tests/python` 40 passed, 4 skipped |

### On the Colab L4 (2026-10-20)

The probe ran twice on a fresh L4 VM (12 vCPU) with the 1080p synthetic clip,
where 60x140 px players stand over the markings. On the merged code
(`8e0ab92`) the calibration failed: 16 of 30 frames uncalibrated, wrong locks of
tens of metres on others, 1.2 s per frame. The cause was a pairing bug in the
global search, invisible on the clean test scenes, fixed in `a4a231c`. On the
fix, measured on the VM:

| Check | Result |
|---|---|
| Frames | 1 calibrated, 29 tracked, 0 uncalibrated |
| Foot-position error against the true camera | mean 0.021 m, max 0.092 m |
| Fit error | mean 0.20 px, max 0.53 px |
| Calibrated smoke run, GS-HOTA | 0.8837 all roles, 0.9231 players (identical to the run given the true camera), LocA 0.9954 |
| Global search (first frame) | 1.44 s |
| Tracked frame | 0.264 s mean (3.3 fps), about 3x slower than the Mac |

On the VM the calibrator, not the GPU, is the slowest stage: about 3.3 minutes
of CPU per 750-frame clip against about 16 s of detection. With pose, a real
clip would take about 4.5 to 5 minutes instead of the 1.5 to 2 minutes
estimated above, so the 49 validation clips need roughly four hours of L4
time. The frame stride below (2026-10-22) brings a steady clip's calibration
down to about 42 s.

Stress test on the Mac after the fix: 24 occluded 1080p clips (60x140 and
80x180 px players, 5 and 7 per team, wide and medium views, 240 frames), all
240 frames calibrated or tracked, worst keypoint error 0.20 m; the merged code
had 40 uncalibrated frames and 32 wrong locks over 1 m on the same clips.

### Frame stride (added 2026-10-22)

During steady tracking the calibrator now fits about one frame in six and
interpolates the frames between (`FrameCalibrator.push` / `flush`, status
`interpolated`; `update` still fits every frame and reproduces the earlier
results exactly). The pipeline feeds the calibrator this way, so the stride is on
by default. The rules are in `calibration.py` above `DEFAULT_FRAME_STRIDE`:
- **Interpolation, not extrapolation.** The skipped frames are interpolated
  quadratically through the last three fits. A constant-velocity prediction
  drifts by a pixel within three frames on a smooth pan.
- **Cuts.** The shot-cut test still runs on every frame. A cut closes the open
  stretch before the new shot starts.
- **Closing fit.** A stretch is interpolated only when its closing fit is
  tracked and lands near its prediction. Otherwise it is rewound and fitted
  frame by frame.
- **Per-frame check.** Every interpolated frame must line up with its own
  markings, checked by `marking_alignment`, a sub-pixel profile check that
  reads a few thousand pixels. A frame that fails is fitted.
- **Adaptive length.** The stretch length grows after clean stretches and drops
  back to fitting every frame after any failure.

Speed, back to back against main on the same frames (150-frame 1920x1080 pan,
0.5 m/frame, mean per frame over frames 2-150):

| Machine | every frame (main) | stride (default 6) | speed-up | 750-frame clip |
|---|---|---|---|---|
| Mac, Apple M5 (3 interleaved rounds) | 82.1 ms | 17.1 ms | 4.80x | 62 s -> 13 s |
| Colab L4 VM, 12 vCPU (2 rounds, `calib_pan_ab` in the probe) | 0.269 s | 0.056 s | 4.81x | 202 s -> 42 s |

A fitted frame costs what it did (0.27 s on the VM, 82 ms on the Mac); a held
frame costs 2.8 ms on the VM. The gain is the share of frames that need no fit
(120 of 150 on this pan).

The probe ran on a fresh L4 on 2026-10-22, pinned to `1896146`
(`colab run --gpu L4 --timeout 900 -s varlab-calib-stride colab/vision_env_probe.py --ref 18961460d267408119a0c25a1798ee1b1d4c73c2`;
4 min 27 s wall clock including provisioning; `colab sessions` showed no
session afterwards). On the probe's own 30-frame clip (static wide camera,
60x140 px players):

| Check | every frame | stride |
|---|---|---|
| Frames | 1 calibrated, 29 tracked | 1 calibrated, 9 tracked, 20 interpolated, 0 replayed |
| Mean per frame after the first | 0.266 s | 0.086 s (includes the ramp-up of a short clip) |
| Fitted / held frame | 0.266 s / - | 0.271 s / 0.0028 s |
| Steady-state cost per 750-frame clip | 200 s | 36 s |
| Foot-position error, mean / max | 0.021 / 0.092 m | 0.022 / 0.069 m |
| Calibrated smoke GS-HOTA | 0.8837 all roles, 0.9231 players, LocA 0.9954 (unchanged) | |

Accuracy on the Mac (synthetic clips, main against the stride):

| Check | main | stride |
|---|---|---|
| Clean 24-scene sweep (6 camera positions x wide and medium x 720p and 1080p, 8 frames each), worst keypoint error on the global-search frame / over all frames | 0.042 / 0.067 m | 0.042 / 0.067 m (48 of 192 frames interpolated, worst 0.048 m on them) |
| Occlusion stress test (24 occluded 1080p clips, 240 frames) | 240 good, 0 uncalibrated, 0 wrong locks over 1 m, worst 0.198 m | 240 good, 0 uncalibrated, 0 wrong locks, worst 0.129 m (72 frames interpolated, worst 0.041 m on them) |
| 60-frame 1080p pan at 0.8 m/frame, worst keypoint error | 0.090 m | 0.025 m |

Camera motion other than a steady pan (1080p frames rendered in memory with 14
player boxes, 48 frames each, every frame fitted against the stride):

| Motion | every frame: worst / ms per frame | stride: worst / ms per frame |
|---|---|---|
| steady pan, 0.8 m/frame | 0.075 m / 99 | 0.025 m / 35 |
| fast pan, 2 m/frame | 0.106 m / 87 | 0.112 m / 66 |
| accelerating pan (to 4.7 m/frame) | 0.062 m / 89 | 0.051 m / 87 |
| start-stop | 0.133 m / 92 | 0.120 m / 81 |
| whip, 20 m in two frames, no cut | 0.004 m / 102 | 0.004 m / 74 |
| zoom, 58 to 38 degrees | 0.052 m / 87 | 0.051 m / 63 |
| camera shake (medium) | 0.040 m / 98-102 | 0.040 m / 98-112 (never holds a frame) |
| oscillating pan, +-2 m/frame | 0.093 m / 94 | 0.089 m / 93 |

No scenario had a wrong lock or an uncalibrated frame. On accelerating,
oscillating or shaky motion the stride falls back to fitting nearly every frame,
at about main's cost. The gain on real footage therefore depends on how much of a
clip the camera holds still or pans steadily. That share is unknown until
SoccerNet clips are available, so the 42 s figure is the steady-camera case and
3.3 minutes is the bound.

Tried and rejected, measured on the same harness:
- Line extraction on a half-resolution frame: 2.1x faster, but the clean sweep
  went to 0.23 m and the stress test to 0.63 m.
- The same with a full-resolution refit of every line: 1.4x faster, 0.07 m and
  0.34 m.
- Two refit passes per line instead of four: 1.35x faster, clean sweep 0.27 m.
- Hough seeds from a half-resolution mask: 1.08x faster, and one frame went to
  0.55 m.
- A stride with constant-velocity extrapolation instead of interpolation:
  interpolated frames up to 0.56 m off.

Overlapping the calibration with the GPU stages was not built. It does not lower
the CPU cost, and after the stride a steady clip's calibration (about 42 s) is
shorter than its detection and pose.

One more observation on the 150-frame pan: fitting every frame, frame 79 has a
0.727 m keypoint error. It is a goal-area corner near the image edge, on main as
on this branch. The stride happens to interpolate that frame.

Not measured: accuracy on real broadcast frames (lens distortion, crowds,
shadows, worn or partly hidden lines, close-ups showing fewer than four
markings), because no SoccerNet clips are available yet. Known limits: one camera, a planar pitch, no lens distortion, straight
markings only (the centre circle and arcs are not used), and at least four
markings in view. The synthetic numbers are an upper bound on real accuracy.

## Two things the first runs taught

1. The Colab kernel already has numpy imported. Pinning `numpy==2.5.4` upgraded
   it on disk and every later compiled import failed with mixed versions. numpy
   now floats in `requirements-vision.txt` (2.1.3 on the VM, 2.5.4 in fresh
   environments) and the probe measures in a fresh subprocess; `run_clips.py`
   installs with `colab install` and then `colab restart-kernel`.
2. A subprocess that inherits the kernel's file descriptor 1 writes to the VM
   console, not to the captured output; the probe pipes the child's output
   through `print()` and echoes the JSON summary at the end.
