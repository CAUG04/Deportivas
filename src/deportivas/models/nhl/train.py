"""NHL moneyline: thin wiring of the shared moneyline pipeline
(``models/moneyline_training.py``) over ``nhl_v1`` features. Spread and
total are out of scope: ``config/markets.yaml`` declares no fixed line grid
for them (a real one comes from a live bookmaker snapshot, not invented
here), so they wait for the phase that joins model output with odds.
"""

from __future__ import annotations

from deportivas.domain.enums import Sport
from deportivas.models.moneyline_training import compute_and_write_moneyline_model

FEATURE_SET = "nhl_v1"
MODEL_NAME_PREFIX = "nhl_moneyline"


def compute_and_write_nhl_moneyline_model(
    competition_id: str,
    *,
    calibration_method: str | None = None,
    min_training_samples: int | None = None,
    reliability_bins: int | None = None,
) -> list[int]:
    return compute_and_write_moneyline_model(
        competition_id,
        sport=Sport.ICE_HOCKEY.value,
        feature_set=FEATURE_SET,
        model_name_prefix=MODEL_NAME_PREFIX,
        calibration_method=calibration_method,
        min_training_samples=min_training_samples,
        reliability_bins=reliability_bins,
    )
