"""nba/train.py: wires the shared moneyline pipeline to nba_v1 with the
right sport/feature_set/model_name_prefix. The walk-forward mechanics
themselves are tested against moneyline_training.py directly."""

from __future__ import annotations

import pytest

from deportivas.models.nba.train import compute_and_write_nba_moneyline_model


def test_wires_sport_feature_set_and_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake(competition_id: str, **kwargs: object) -> list[int]:
        calls["competition_id"] = competition_id
        calls.update(kwargs)
        return [10, 20]

    monkeypatch.setattr("deportivas.models.nba.train.compute_and_write_moneyline_model", _fake)

    result = compute_and_write_nba_moneyline_model("usa-nba", calibration_method="platt")

    assert result == [10, 20]
    assert calls["competition_id"] == "usa-nba"
    assert calls["sport"] == "basketball"
    assert calls["feature_set"] == "nba_v1"
    assert calls["model_name_prefix"] == "nba_moneyline"
    assert calls["calibration_method"] == "platt"
