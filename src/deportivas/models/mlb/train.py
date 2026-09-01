"""MLB moneyline: thin wiring of the shared moneyline pipeline
(``models/moneyline_training.py``) over ``mlb_v1`` features. MLB has no
``spread`` market at all (``config/markets.yaml``'s ``sports`` list for it
omits baseball); ``total`` is out of scope for the same reason as the other
three sports — it needs a real line from a live bookmaker snapshot, not one
invented here.
"""

from __future__ import annotations

from deportivas.domain.enums import Sport
from deportivas.models.moneyline_training import compute_and_write_moneyline_model

FEATURE_SET = "mlb_v1"
MODEL_NAME_PREFIX = "mlb_moneyline"


def compute_and_write_mlb_moneyline_model(
    competition_id: str,
    *,
    calibration_method: str | None = None,
    min_training_samples: int | None = None,
    reliability_bins: int | None = None,
) -> list[int]:
    return compute_and_write_moneyline_model(
        competition_id,
        sport=Sport.BASEBALL.value,
        feature_set=FEATURE_SET,
        model_name_prefix=MODEL_NAME_PREFIX,
        calibration_method=calibration_method,
        min_training_samples=min_training_samples,
        reliability_bins=reliability_bins,
    )
