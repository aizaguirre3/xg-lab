"""varlab: calibration and evaluation helpers ported from xg-lab's R code.

See :mod:`varlab.calibration` for the function reference and the list of
documented differences from the R implementation.
"""

from .calibration import (
    abstention_curve,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    group_folds,
    isotonic_by_fold,
    log_loss,
    reliability_data,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "abstention_curve",
    "brier_decomposition",
    "brier_score",
    "expected_calibration_error",
    "group_folds",
    "isotonic_by_fold",
    "log_loss",
    "reliability_data",
]
