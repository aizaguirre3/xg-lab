"""VAR-Lab M4: the vision stack (detection, tracking, team assignment, pose).

Kept separate from ``varlab`` so that package stays dependency-light. This
package needs OpenCV, numpy, pandas and scipy for the loaders and the GS-HOTA
adapter; the model backends (RF-DETR, supervision/ByteTrack, SigLIP via
transformers, RTMPose via rtmlib) are imported lazily, only when a backend is
constructed, so the loaders and the CPU smoke backends work without them.
Pinned versions and licences: ``requirements-vision.txt`` and
``varlab_vision/LICENSES.md``.

Nothing here assumes SoccerNet files exist: every reader takes an explicit path
and every test uses hand-written fixtures or a synthetic clip.
"""

__all__ = ["gsr_format", "loaders", "gshota", "synthetic", "pipeline"]
__version__ = "0.1.0"
