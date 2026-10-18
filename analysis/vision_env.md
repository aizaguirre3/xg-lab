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

Not measured here (no data): real detection quality, team-cluster purity, the
pitch calibration stage (not implemented; `bbox_pitch` is null without a
`PitchProjector`), jersey recognition (not implemented; GS-HOTA requires it),
and the layout of the clips inside the Hugging Face dataset repository.

## Two things the first runs taught

1. The Colab kernel already has numpy imported. Pinning `numpy==2.5.4` upgraded
   it on disk and every later compiled import failed with mixed versions. numpy
   now floats in `requirements-vision.txt` (2.1.3 on the VM, 2.5.4 in fresh
   environments) and the probe measures in a fresh subprocess; `run_clips.py`
   installs with `colab install` and then `colab restart-kernel`.
2. A subprocess that inherits the kernel's file descriptor 1 writes to the VM
   console, not to the captured output; the probe pipes the child's output
   through `print()` and echoes the JSON summary at the end.
