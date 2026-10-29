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

## First real footage: SoccerNet GSR validation, 58 clips (2026-10-26)

**Data.** `SoccerNet/SN-GSR-2025` on Hugging Face is public and not gated (no NDA
or token needed; the dataset card states licence gpl-3.0; nothing from it is
committed or redistributed here). It ships one zip per split: `valid.zip` is
11.2 GB with 58 clips of 750 frames at 1920x1080 (the 2024 edition had 49).
`varlab_vision/remote_zip.py` reads the zip's directory once (8.5 MB) and range-reads
one clip at a time (about 200 MB), so no session downloads the whole split.

**Run.** `python colab/run_clips.py --all --split valid --ref <sha> --out data/gsr_pred/valid
--gpu L4 --no-pose --no-calibration --resume`. Pose was off because GSR has no
pose labels to score it against; calibration was off on Colab because the Mac
measurement below showed it does not work on this footage yet, and at about
0.7 s per frame on the VM it would have cost roughly ten hours of L4. All 58
clips: 43 min of processing over five L4 sessions; per clip (median) 44 s wall,
of which download 15 s, JPEG read 8 s, RF-DETR 17 s, ByteTrack 1.3 s, SigLIP
teams 1 s. Three runner faults surfaced and are fixed: a transient
`colab download` failure (now retried), a `colab exec` that waited 35 min on an
idle VM (now a hard local time limit), and a client "Timeout waiting for
output" during a silent step (now a 20 s heartbeat).

**Detection, tracking, teams** (`tools/eval_gsr_tracking.py`; people = player,
goalkeeper, referee, other; IoU >= 0.5; tracker output only):

| Metric | All 58 clips | Per clip, median (min to max) |
|---|---|---|
| Box precision | 0.946 | 0.956 (0.797 to 0.995) |
| Box recall | 0.814 | 0.850 (0.639 to 0.944) |
| Recall: players / goalkeepers / referees | 0.838 / 0.691 / 0.627 | |
| MOTA | 0.763 | 0.785 (0.485 to 0.932) |
| IDF1 | 0.628 | 0.625 (0.405 to 0.908) |
| Tracks predicted vs labelled | 2,957 vs 1,350 (2,867 ID switches) | |
| Team assignment (player and goalkeeper boxes) | 0.735 | 0.733 (0.529 to 0.968) |
| Ball found within 15 px / ball detections correct | 0.346 / 0.886 | ball recall 0.327 (0.057 to 0.827) |

The detector is sound (95% precise); recall falls on goalkeepers and referees.
The tracker splits each labelled person into about 2.2 tracks, so identity is the
weakest part, and the colour-free SigLIP + k-means team split is right about
three times in four. The ball is found in about a third of the frames that show it.

**Calibration** (`tools/eval_gsr_calibration.py`, every 50th frame of every clip,
870 frames, global search per frame on the Mac CPU, compared with the labels'
own pitch positions, which come from SoccerNet's annotated camera):

| Measure | Result |
|---|---|
| Frames calibrated | 265 of 870 (30%) |
| Calibrated frames with labelled people: median error under 1 m / 1 to 5 m / over 5 m | 25 / 19 / 216 of 260 |
| Labelled people placed within 1 m, over all sampled frames | 3.1% |
| Clips whose calibrated frames are right (median under 1 m) | 3 of 58 (SNGS-039, -050, -056) |
| Seconds per frame (global search) | 0.71 |

The calibration does not work on broadcast footage yet, and worse, most frames
it reports as calibrated are wrong (over 5 m), which breaks the "never guess"
rule the synthetic tests enforce. The frames show why: a typical midfield view
holds the halfway line, one touchline and the centre circle; the method uses
straight lines only, cuts the circle into straight chords that it then matches
as markings, reads white shirts as marking pixels, and misses faint distant
lines with its fixed brightness threshold. On a full tracked clip (SNGS-021,
750 frames) the median error was 46 m. Until this is fixed the calibrator should
stay off for real clips (`--no-calibration`). (Reworked on 2026-10-29: see "Calibration on broadcast frames" below.)

**Next, in order of what the numbers say:** (1) calibration: mask out detected
people, find marking pixels by local contrast (white top-hat) inside the grass
area, reject curved segments, fit the centre circle and penalty arcs as ellipses
and use them in the homography, and add a consistency check that refuses a
solution instead of reporting a wrong one; (2) tracking: re-identification or a
longer ByteTrack buffer to cut the 2.2x fragmentation; (3) a role classifier for
goalkeepers and referees and a better team split; (4) jersey recognition, which
GS-HOTA needs. Per-clip numbers: `analysis/gsr_valid_2026-10-26.json`.

## Calibration on broadcast frames (2026-10-29)

The 2026-10-26 run showed the calibration guessing on real frames: 30% of the
sampled validation frames calibrated, 216 of the 260 with labelled people over
5 m wrong, 3.1% of the labelled people within 1 m. Four changes, each measured
on a fixed tuning set from `train.zip` before the next, then the validation set at
the end.

**Tuning set.** Every 50th frame (1, 51, ..., 701) of 15 train clips spread over
the split (SNGS-060, -064, -068, -072, -076, -099, -103, -107, -111, -115, -154,
-158, -162, -166, -170): 225 frames, cached locally (gitignored). The validation
clips were first measured after step 4; the changes after that were chosen on the
tuning set alone, and both validation runs are reported. Person mask: the frame's *labelled* person
boxes stand in for the detector's (the pipeline passes its detections; the eval
tool uses the labels' image boxes, never their pitch positions).

**What changed** (`varlab_vision/calibration.py`):

1. *Marking pixels by local contrast.* `marking_mask`: a white top-hat (21 px
   square) on the minimum colour channel, blurred (sigma 1), thresholded at 2.5x
   its local median (31 px window, capped at 25 so thick strokes cannot raise it)
   plus 11, inside `grass_region` (the largest green component, holes filled) and
   outside the person boxes. Against the markings that the labels' homography
   projects into 69 tuning frames: the old mask found 35% of them with 6.5% of its
   pixels on a marking; the new one finds 66% with 48% (median 22k pixels a frame
   against 63k). The board text above the far touchline and white shirts were the
   bulk of the old mask.
2. *Curved candidates are not lines.* `stroke_bend` fits a parabola to a
   candidate's cross-section centroids; one that bows by over 0.7 px with a radius
   of curvature under 1.3 frame widths is dropped. On 1,213 tuning lines labelled
   by the labels' homography this removes 62% of the arc chords and 5.6% of the
   straight lines (1.9% of their length: short ones).
3. *The centre circle and the penalty arcs.* `detect_ellipses` erases the long
   straight lines, grows an ellipse from each remaining connected piece over the
   pixels within about the stroke width, and refits it geometrically (Sampson
   distance); lines lying on an ellipse are dropped as chords. The search adds
   hypotheses from the centre circle, the halfway line and one touchline: the
   halfway line meets the ellipse at (0, +-9.15), its pole with respect to the
   ellipse is the x vanishing point (the touchline must pass through it), the
   touchline crossing gives (0, +-34), the cross-ratio along the halfway line the
   centre, and the line from the centre to the vanishing point meets the ellipse at
   (+-9.15, 0); six correspondences, one DLT. Ellipses also score hypotheses and
   enter the refinement (Sampson distance to the projected model circle), at 0.3
   weight when the lines alone fix the homography. Every line of at least a
   quarter of the frame width is erased before the ellipse search, whatever its
   bow (lens distortion bows a long halfway line past the 0.5 px rule for shorter
   ones, and left it joined to the circle).
4. *Refuse instead of guess.* Each solution (global or tracked) is refined once
   more as a pinhole camera (square pixels, principal point at the centre: seven
   parameters) and checked against the frame (`verify_solution`); `VerifyConfig`
   refuses it when the homography has no real camera behind it (no positive
   focal length, or a camera outside x +-70 m, y 15 to 150 m, 3 to 50 m up: the
   GSR main camera sits on the +y side), when over 30% of the detected line length
   is unexplained, when leaving out any one matched marking moves the visible pitch
   by over 5 m, when the free fit is far from a camera (inconsistency over 0.2), or
   when the fit's standard error at the people's feet (parameter covariance carried
   to the pitch) exceeds 0.11 m (0.5 m at the 90th percentile of the visible pitch
   without person boxes). The same camera box also prunes the global search, where
   hypotheses no camera could produce often outscored the right one. A refused
   global solution is followed by up to three distinct runner-up hypotheses; if all
   fail the frame is `uncalibrated`, with the reasons in `note` and the evidence in
   `verify`.

Thresholds, on the tuning set only:

| Step | Frames claimed | Claimed frames: under 1 m / 1 to 5 m / over 5 m | Precision (under 1 m) | People within 1 m |
|---|---|---|---|---|
| Before (`e3ec8a1`) | 86 of 225 | 16 / 2 / 67 | 0.19 | 5.7% |
| 1. contrast mask, people boxed out | 160 of 225 | 75 / 19 / 65 | 0.47 | 30.9% |
| 2. + bend test | 165 of 225 | 78 / 23 / 63 | 0.48 | 32.5% |
| 3. + centre circle and arcs | 177 of 225 | 103 / 28 / 46 | 0.58 | 42.5% |
|    robustness for the synthetic tests (see below) | 164 of 225 | 103 / 29 / 32 | 0.63 | 43.2% |
|    + refinement as a camera | 168 of 225 | 107 / 32 / 29 | 0.64 | 44.7% |
| 4. + verification (refuse) | 79 of 225 | 75 / 4 / 0 | 0.95 | 30.0% |
|    + runner-up hypotheses | 81 of 225 | 77 / 4 / 0 | 0.95 | 30.6% |
|    + fit weights out of the boxes | 81 of 225 | 75 / 6 / 0 | 0.93 | 30.4% |
|    + camera prior in the search, long lines erased before the ellipse search, limit 0.11 m (final) | 89 of 225 | 83 / 6 / 0 | 0.93 | 33.7% |

The robustness row is what the synthetic tests needed: the noise level capped at
25 (thick near-camera strokes filled the median window), a 2 px peel halo (the
contrast mask keeps the anti-aliased fringe on one side of a stroke, and a 1 px
halo left it as a parallel "line" 4 px off), arcs at 0.3 weight when lines
suffice, and fit weights kept out of the person boxes (they cost the tuning set two
frames). Without verification the calibrator places 44.7% of the people within 1 m
but claims 168 frames, 61 of them wrongly; with it, the final code claims 89 and is
wrong on 6, all between 1 and 5 m. The test that refuses most of the far-off
solutions is the camera: 29 of the 32 frames over 5 m wrong have no positive focal
length.

**Validation** (`tools/eval_gsr_calibration.py --labels data/gsr_valid --every 50
--workers 8`, 870 frames of all 58 clips, each calibrated from scratch, labelled
person boxes as the people mask):

| Measure | 2026-10-26, before (`e3ec8a1`) | 2026-10-29, after (`25f73a6`) |
|---|---|---|
| Frames claimed | 265 of 870 (30%) | 351 of 870 (40%) |
| Claimed frames with labelled people: median under 1 m / 1 to 5 m / over 5 m | 25 / 19 / 216 of 260 | 338 / 11 / 0 of 349 |
| **Precision of the claim** (share of those under 1 m) | 0.096 | **0.969** |
| **Labelled people within 1 m**, over all 14,244 on the sampled frames | 3.1% | **38.1%** |
| Labelled people within 2 m | 3.7% | 41.1% |
| Foot error on the claimed frames: median / share within 1 m | 56.5 m / 10.5% | 0.40 m / 91.6% |
| Clips with at least one claimed frame | 58 of 58 (mostly wrong) | 54 of 58 |
| Uncalibrated frames: refused / no consistent hypothesis / too few markings | (no refusal step) | 297 / 220 / 2 |

Per-clip precision is 1.0 in 45 of the 54 clips with a claimed frame and at
least 0.8 in 52. The circle or an arc took part in 263 of the 351 claimed frames.
The most common refusal is a standard error at the people over 0.11 m (262 of the
297), then unexplained lines (118) and camera inconsistency (76); a frame can have
several. An intermediate run one commit before the person-box weight fix and the
search camera prior (`c140171`) gave 313 claimed, precision 0.955, 33.7% within 1 m;
the final code was then tuned on the train set only.

**The labels are not the paint.** The labels' pitch positions fit a homography
to about 0.1 to 0.2 m, so that homography stands for the annotation; scored with the
same marking-support measure as ours (share of the model's marking points in view
that land on marking pixels), it fits the painted lines worse than our solution.
On the 337 validation frames within 1 m: median support 0.84 for the labels against
0.99 for ours (the labels clearly worse, by over 0.1, in 185). On the 11 frames
between 1 and 5 m: 0.73 against 0.98 (labels clearly worse in 9). The tuning set
showed the same, and overlays of such frames show our lines on the paint and the
labels' a few pixels off (up to about 10 px on a far touchline). So most of the "1
to 5 m" column is annotation error, and precision against the labels understates
precision against the pitch; the labels are still the yardstick here.

**Tracked clip.** All 750 frames of SNGS-021 in order through a `FrameCalibrator`
(tracking, filter, frame stride; `--tracked --every 1 --clips SNGS-021`): before, the
median error over the clip was 46 m. Now 314 frames are claimed (20 global, 278
tracked, 16 interpolated), 264 within 1 m, 50 between 1 and 5 m, none over 5 m
(precision 0.84); 33.7% of the labelled people within 1 m; median foot error on the
claimed frames 0.42 m. The 50 are all tracked frames in three stretches (391-408,
477-500, 602-618), and there the annotation is what moves: the camera the labels
imply sits a median 6.8 m from its own clip-median position on those 50 frames
(1.8 m on the 264 we agree on), while ours stays at about (0.4, 49.4, 9.5) m, as a
fixed broadcast camera should. On frame 609, for one, our model lines lie on the
painted penalty area and near touchline and the labels' are visibly off; the
labels' homography reaches marking support 0.73 there against our 0.91. A
from-scratch fit of frame 603 agrees with the tracked one (1.33 m from the labels).

**Timing**, back to back on the Mac (one process, the same 45 tuning frames,
each twice): old code 0.68 and 0.71 s per frame (median of the
two passes), new code 0.80 and 0.92 s: the global search with verification is about
1.25 times slower. The SNGS-021 tracked run took 506 s for 750 frames (0.68 s a frame)
on one core, because every uncalibrated frame falls back to the global search;
on the Colab VM's CPU, about three times slower, calibration would be the slowest
stage of a real clip (roughly 25 minutes). The validation run's 2.6 s per frame is
eight workers sharing ten cores.

**Tests.** `tests/vision` 137 passed (the 120 synthetic tests unchanged, plus
`test_calibration_broadcast.py`, 15 tests on rendered frames, one or more per
mechanism; a pipeline test that the person detections reach the calibrator; an
eval-tool test of the precision summary); `tests/python` 40 passed, 4 skipped.

**Not done / limits.** Coverage is the price of precision: 40% of the sampled
validation frames are claimed. A sample of the uncalibrated tuning frames shows
ordinary main-camera views, not close-ups: midfield views where the circle is
faint or only partly found, and views whose markings are few, short or far, so
the fit's standard error at the people stays over the limit. Better ellipse
detection and a lens-distortion term are the next levers. Penalty arcs are scored
and fitted but do not seed hypotheses (an arc of about 106 degrees does not fix its
conic well enough for the pole construction). The camera prior (main camera on
the +y side, 3 to 50 m up) is GSR's; another dataset may need
`VerifyConfig.camera_box` relaxed. The person mask was measured with the labelled
boxes; the detector misses about one person in five (recall 0.81), and a missed
white shirt can still read as a marking.

## Two things the first runs taught

1. The Colab kernel already has numpy imported. Pinning `numpy==2.5.4` upgraded
   it on disk and every later compiled import failed with mixed versions. numpy
   now floats in `requirements-vision.txt` (2.1.3 on the VM, 2.5.4 in fresh
   environments) and the probe measures in a fresh subprocess; `run_clips.py`
   installs with `colab install` and then `colab restart-kernel`.
2. A subprocess that inherits the kernel's file descriptor 1 writes to the VM
   console, not to the captured output; the probe pipes the child's output
   through `print()` and echoes the JSON summary at the end.
