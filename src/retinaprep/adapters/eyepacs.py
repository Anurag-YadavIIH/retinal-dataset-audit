"""EyePACS / Kaggle Diabetic Retinopathy adapter — STUB, not needed tonight.

Filenames follow `<patient_id>_<left|right>.jpeg`, so patient_id and eye come
free from the filename. Grades are 0-4; binary label = 1 for grade >= 2
(referable DR), which is the standard clinical threshold. Implement this later
to demonstrate the adapter layer generalises.
"""

from __future__ import annotations

from retinaprep.adapters.base import register


@register("eyepacs")
def build_manifest(cfg: dict):
    raise NotImplementedError("Deferred. See roadmap in README.md.")
