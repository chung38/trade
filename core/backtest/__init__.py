"""回測 / 樣本外驗證。"""

from .validate import (
    OutOfSampleReport,
    holdout_validate,
    walk_forward_validate,   # deprecated alias
)

__all__ = ["OutOfSampleReport", "holdout_validate", "walk_forward_validate"]
