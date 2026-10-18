# Licences of the M4 vision dependencies

Checked on 2026-10-17 against the PyPI metadata of the pinned versions in
`requirements-vision.txt` (and the model cards / repository LICENSE files for
weights). Nothing GPL or AGPL is imported; the GPL-3 sn-gamestate / TrackLab code
and the GPL-2 PnLCalib code were not read or copied (loaders and GS-HOTA are
written from the public format description, see `gsr_format.py`, `gshota.py`).

| Package / asset | Version | Licence | Role | Source |
|---|---|---|---|---|
| rfdetr | 1.11.2 | Apache-2.0 | RF-DETR detector | https://github.com/roboflow/rf-detr |
| RF-DETR Medium COCO weights (`rf-detr-medium.pth`) | md5 `7223f764a87b863f02eb8d52bf0ce2ee` (shipped in rfdetr 1.11.2) | Apache-2.0 (Nano/Small/Medium/Large only; Atto/Femto/Pico/XL/2XL are PML 1.0 and are **not** used) | detector weights | https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth (Roboflow bucket, fetched by the package) |
| supervision | 0.31.0 | MIT | `Detections` container | https://github.com/roboflow/supervision |
| trackers | 2.6.1 | Apache-2.0 | ByteTrack (clean-room re-implementation; supervision 0.31 no longer ships ByteTrack) | https://github.com/roboflow/trackers |
| transformers | 5.20.1 | Apache-2.0 | SigLIP vision model | https://github.com/huggingface/transformers |
| SigLIP `google/siglip-base-patch16-224` | revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | Apache-2.0 (model card tag) | team-assignment embeddings | https://huggingface.co/google/siglip-base-patch16-224 |
| huggingface-hub | 2.2.0 | Apache-2.0 | pinned downloads | https://github.com/huggingface/huggingface_hub |
| safetensors | 0.8.0 | Apache-2.0 | weight format | https://github.com/huggingface/safetensors |
| rtmlib | 0.0.16 | Apache-2.0 (LICENSE: "Copyright 2018-2020 Open-MMLab") | RTMPose runner (no mmcv/mmpose) | https://github.com/Tau-J/rtmlib |
| RTMPose-m body7 ONNX (`rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip`) | 2023-05-04 release | Apache-2.0 (MMPose project licence) | 2D keypoints | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/ (OpenMMLab host; rtmlib falls back to its Hugging Face mirror) |
| onnxruntime / onnxruntime-gpu | 1.31.0 | MIT | ONNX inference | https://github.com/microsoft/onnxruntime |
| opencv-python, opencv-contrib-python (opencv-python-headless in CI) | 5.0.0.93 | Apache-2.0 | image I/O, colour, connected components | https://github.com/opencv/opencv-python |
| numpy | 2.5.4 | BSD-3-Clause (plus 0BSD/MIT/Zlib/CC0 vendored parts) | arrays | https://numpy.org |
| pandas | 3.0.6 | BSD-3-Clause | tables | https://pandas.pydata.org |
| scipy | 1.18.1 | BSD-3-Clause | Hungarian matching (GS-HOTA, IoU tracker) | https://scipy.org |
| pillow | 13.0.0 | MIT-CMU | image conversion for transformers | https://python-pillow.org |
| torch | VM build (>=2.2) | BSD-3-Clause | model runtime | https://pytorch.org |
| torchvision | VM build (>=0.17) | BSD-3-Clause | rfdetr dependency | https://pytorch.org |

Transitive packages pulled by the above (av, matplotlib, pydantic, tqdm, requests,
tokenizers, regex, sympy, networkx, jinja2, filelock, fsspec, hf-xet, …) are all
MIT, BSD, Apache-2.0 or PSF; `pip install` of the pinned file on 2026-10-17 pulled
no copyleft package. Re-check with `pip-licenses` after any pin change.

Excluded on purpose: Ultralytics YOLO (AGPL-3.0), sn-gamestate and TrackLab
(GPL-3.0), PnLCalib (GPL-2.0), RF-DETR Atto/Femto/Pico/XL/2XL (PML 1.0).
