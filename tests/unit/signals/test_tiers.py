"""tiers.py: top-down alta/media/baja/descartar classification against
config/thresholds.yaml's tiers section."""

from __future__ import annotations

from deportivas.config.catalog import TierConfig, TiersConfig, load_thresholds
from deportivas.domain.enums import Tier
from deportivas.signals.tiers import TierInputs, classify_tier

_TIERS = TiersConfig(
    alta=TierConfig(
        min_edge=0.04,
        min_sample_matches=300,
        requires_calibration=True,
        requires_sharp_price=True,
        requires_favourable_line_move=True,
    ),
    media=TierConfig(
        min_edge=0.02,
        min_sample_matches=300,
        requires_calibration=True,
        requires_sharp_price=True,
        requires_favourable_line_move=False,
    ),
    baja=TierConfig(
        min_edge=0.0,
        min_sample_matches=0,
        requires_calibration=False,
        requires_sharp_price=False,
        requires_favourable_line_move=False,
    ),
)


def _inputs(**overrides: object) -> TierInputs:
    base: dict[str, object] = {
        "edge": 0.05,
        "sample_matches": 400,
        "is_calibrated": True,
        "has_sharp_price": True,
        "has_favourable_line_move": True,
    }
    base.update(overrides)
    return TierInputs(**base)  # type: ignore[arg-type]


def test_meets_every_condition_of_alta() -> None:
    result = classify_tier(_inputs(), _TIERS)
    assert result.tier == Tier.ALTA
    assert all(result.reasons.values())


def test_falls_to_media_without_favourable_line_move() -> None:
    result = classify_tier(_inputs(has_favourable_line_move=False), _TIERS)
    assert result.tier == Tier.MEDIA


def test_falls_to_baja_without_sharp_price() -> None:
    result = classify_tier(_inputs(edge=0.03, has_sharp_price=False), _TIERS)
    assert result.tier == Tier.BAJA


def test_falls_to_baja_when_not_calibrated() -> None:
    result = classify_tier(_inputs(edge=0.03, is_calibrated=False), _TIERS)
    assert result.tier == Tier.BAJA


def test_negative_edge_is_descartar() -> None:
    result = classify_tier(_inputs(edge=-0.01), _TIERS)
    assert result.tier == Tier.DESCARTAR


def test_below_baja_sample_size_is_descartar() -> None:
    tiers = TiersConfig(
        alta=_TIERS.alta,
        media=_TIERS.media,
        baja=TierConfig(
            min_edge=0.0,
            min_sample_matches=300,
            requires_calibration=False,
            requires_sharp_price=False,
            requires_favourable_line_move=False,
        ),
    )
    result = classify_tier(_inputs(sample_matches=10), tiers)
    assert result.tier == Tier.DESCARTAR
    assert result.reasons["min_sample_matches"] is False


def test_reasons_report_every_condition_for_the_assigned_tier() -> None:
    result = classify_tier(_inputs(has_favourable_line_move=False), _TIERS)
    assert result.tier == Tier.MEDIA
    assert set(result.reasons) == {
        "min_edge",
        "min_sample_matches",
        "requires_calibration",
        "requires_sharp_price",
        "requires_favourable_line_move",
    }
    assert result.reasons["requires_favourable_line_move"] is True  # media no lo exige


def test_works_against_the_real_thresholds_yaml_config() -> None:
    tiers = load_thresholds().tiers
    result = classify_tier(_inputs(edge=0.0, sample_matches=0, is_calibrated=False), tiers)
    assert result.tier in {Tier.BAJA, Tier.DESCARTAR}
