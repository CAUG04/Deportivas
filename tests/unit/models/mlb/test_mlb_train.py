"""mlb/train.py: wires the shared moneyline pipeline to mlb_v1 with the
right sport/feature_set/model_name_prefix. The walk-forward mechanics
themselves are tested against moneyline_training.py directly."""

from __future__ import annotations

import pytest

from deportivas.models.mlb.train import compute_and_write_mlb_moneyline_model


def test_wires_sport_feature_set_and_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str, **kwargs: object) -> list[int]:
        calls["competition_id"] = competition_id
        calls.update(kwargs)
        return [10, 20]

    monkeypatch.setattr("deportivas.models.mlb.train.compute_and_write_moneyline_model", _fake)

    result = compute_and_write_mlb_moneyline_model("usa-mlb", calibration_method="platt")

    assert result == [10, 20]
    assert calls["competition_id"] == "usa-mlb"
    assert calls["sport"] == "baseball"
    assert calls["feature_set"] == "mlb_v1"
    assert calls["model_name_prefix"] == "mlb_moneyline"
    assert calls["calibration_method"] == "platt"
