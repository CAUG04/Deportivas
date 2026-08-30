"""moneyline_training.py: walk-forward training and out-of-sample
prediction for the shared moneyline classifier, used by NFL/NBA/NHL/MLB."""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from deportivas.config.settings import get_settings
from deportivas.contracts.tables import FEATURES, FIXTURES, MODEL_REGISTRY, PREDICTIONS
from deportivas.models.moneyline_training import (
    TrainedMoneylineWindow,
    compute_and_write_moneyline_model,
    train_and_predict_moneyline_window,
)
from deportivas.models.walkforward import WalkForwardWindow, walk_forward_windows
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository

_TEAMS = [f"team{i}" for i in range(10)]  # 10*9 = 90 partidos por temporada


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _season_rows(
    season: str, teams: list[str], *, start: datetime, seed: int
) -> list[dict[str, object]]:
    """A full round-robin, every match finished, with a feature vector whose
    single field (``edge``) genuinely drives the outcome — so the fitted
    classifier has something real to learn, and the test data isn't just
    coin flips."""
    rng = random.Random(seed)
    rows = []
    kickoff = start
    for idx, (home, away) in enumerate(itertools.permutations(teams, 2)):
        edge = rng.uniform(-3.0, 3.0)
        home_wins = rng.random() < 1.0 / (1.0 + pow(2.718281828, -edge))
        rows.append(
            {
                "id": f"{season}-{idx}",
                "competition_id": "usa-nfl",
                "season": season,
                "home_team_id": home,
                "away_team_id": away,
                "kickoff_utc": kickoff,
                "status": "finished",
                "home_score": 20 if home_wins else 10,
                "away_score": 10 if home_wins else 20,
                "as_of_timestamp": kickoff - timedelta(hours=1),
                "vector": {"edge": edge},
            }
        )
        kickoff += timedelta(hours=6)
    return rows


def _features_df(*seasons_rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([row for rows in seasons_rows for row in rows])


def test_window_returns_none_below_min_training_samples() -> None:
    season1 = _season_rows("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_rows("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    features_df = _features_df(season1, season2)
    window = walk_forward_windows(features_df)[0]

    result = train_and_predict_moneyline_window(
        features_df,
        window,
        competition_id="usa-nfl",
        sport="american_football",
        feature_set="nfl_v1",
        model_name="nfl_moneyline_usa-nfl",
        calibration_method="isotonic",
        min_training_samples=1000,
        reliability_bins=10,
    )

    assert result is None


def _trained_window(
    **overrides: object,
) -> tuple[TrainedMoneylineWindow, pd.DataFrame, WalkForwardWindow]:
    season1 = _season_rows("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_rows("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    features_df = _features_df(season1, season2)
    window = walk_forward_windows(features_df)[0]
    kwargs: dict[str, object] = {
        "competition_id": "usa-nfl",
        "sport": "american_football",
        "feature_set": "nfl_v1",
        "model_name": "nfl_moneyline_usa-nfl",
        "calibration_method": "isotonic",
        "min_training_samples": 40,
        "reliability_bins": 10,
    }
    kwargs.update(overrides)
    result = train_and_predict_moneyline_window(features_df, window, **kwargs)  # type: ignore[arg-type]
    assert result is not None
    return result, features_df, window


def test_predicts_home_and_away_for_every_validate_fixture() -> None:
    result, features_df, window = _trained_window()
    validate_count = int((features_df["season"] == window.validate_season).sum())
    assert len(result.prediction_rows) == validate_count * 2
    assert set(result.prediction_rows["selection"]) == {"home", "away"}


def test_home_and_away_raw_probs_sum_to_one() -> None:
    result, _, _ = _trained_window()
    sums = result.prediction_rows.groupby("fixture_id")["prob_raw"].sum()
    assert (sums.round(6) == 1.0).all()


def test_as_of_timestamp_is_constant_and_equals_training_cutoff() -> None:
    result, features_df, window = _trained_window()
    train_df = features_df[features_df["season"].isin(window.train_seasons)]
    expected = train_df["as_of_timestamp"].max()
    assert result.prediction_rows["as_of_timestamp"].nunique() == 1
    assert result.prediction_rows["as_of_timestamp"].iloc[0] == expected


def test_as_of_strictly_before_every_validate_kickoff() -> None:
    result, features_df, window = _trained_window()
    validate_fixtures = features_df[features_df["season"] == window.validate_season][
        ["id", "kickoff_utc"]
    ].rename(columns={"id": "fixture_id"})
    merged = result.prediction_rows.merge(validate_fixtures, on="fixture_id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_prob_calibrated_is_a_valid_probability() -> None:
    result, _, _ = _trained_window()
    calibrated = result.prediction_rows["prob_calibrated"].dropna()
    assert len(calibrated) > 0
    assert calibrated.between(0.0, 1.0).all()


def test_model_row_metrics_are_valid_json() -> None:
    result, _, _ = _trained_window()
    metrics = json.loads(result.model_row["metrics"])  # type: ignore[arg-type]
    assert "moneyline" in metrics
    assert "brier" in metrics["moneyline"]
    assert "log_loss" in metrics["moneyline"]
    assert "reliability" in metrics["moneyline"]


def test_model_row_names_and_sport() -> None:
    result, _, window = _trained_window()
    assert result.model_row["model_name"] == "nfl_moneyline_usa-nfl"
    assert result.model_row["model_version"] == window.validate_season
    assert result.model_row["sport"] == "american_football"
    assert result.model_row["feature_set"] == "nfl_v1"


def test_tied_matches_are_excluded_from_training() -> None:
    season1 = _season_rows("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season1[0] = {**season1[0], "home_score": 14, "away_score": 14}
    season2 = _season_rows("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    features_df = _features_df(season1, season2)
    window = walk_forward_windows(features_df)[0]

    result = train_and_predict_moneyline_window(
        features_df,
        window,
        competition_id="usa-nfl",
        sport="american_football",
        feature_set="nfl_v1",
        model_name="nfl_moneyline_usa-nfl",
        calibration_method="isotonic",
        min_training_samples=40,
        reliability_bins=10,
    )

    assert result is not None
    metrics = json.loads(result.model_row["metrics"])  # type: ignore[arg-type]
    assert metrics["n_train_matches"] == len(season1) - 1  # el empate no cuenta


def test_scheduled_training_match_is_excluded_from_training() -> None:
    season1 = _season_rows("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season1[0] = {**season1[0], "status": "scheduled", "home_score": None, "away_score": None}
    season2 = _season_rows("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    features_df = _features_df(season1, season2)
    window = walk_forward_windows(features_df)[0]

    result = train_and_predict_moneyline_window(
        features_df,
        window,
        competition_id="usa-nfl",
        sport="american_football",
        feature_set="nfl_v1",
        model_name="nfl_moneyline_usa-nfl",
        calibration_method="isotonic",
        min_training_samples=40,
        reliability_bins=10,
    )

    assert result is not None
    metrics = json.loads(result.model_row["metrics"])  # type: ignore[arg-type]
    assert metrics["n_train_matches"] == len(season1) - 1


def test_window_returns_none_when_training_outcomes_have_no_variance() -> None:
    season1 = _season_rows("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season1 = [{**row, "home_score": 20, "away_score": 10} for row in season1]  # local gana siempre
    season2 = _season_rows("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    features_df = _features_df(season1, season2)
    window = walk_forward_windows(features_df)[0]

    result = train_and_predict_moneyline_window(
        features_df,
        window,
        competition_id="usa-nfl",
        sport="american_football",
        feature_set="nfl_v1",
        model_name="nfl_moneyline_usa-nfl",
        calibration_method="isotonic",
        min_training_samples=40,
        reliability_bins=10,
    )

    assert result is None


def test_metrics_omit_moneyline_key_when_validate_season_has_no_decided_matches() -> None:
    season1 = _season_rows("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_rows("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    season2 = [
        {**row, "status": "scheduled", "home_score": None, "away_score": None} for row in season2
    ]
    features_df = _features_df(season1, season2)
    window = walk_forward_windows(features_df)[0]

    result = train_and_predict_moneyline_window(
        features_df,
        window,
        competition_id="usa-nfl",
        sport="american_football",
        feature_set="nfl_v1",
        model_name="nfl_moneyline_usa-nfl",
        calibration_method="isotonic",
        min_training_samples=40,
        reliability_bins=10,
    )

    assert result is not None
    # las predicciones se escriben igual, pero sin resultado conocido no hay metricas
    assert len(result.prediction_rows) == len(season2) * 2
    metrics = json.loads(result.model_row["metrics"])  # type: ignore[arg-type]
    assert "moneyline" not in metrics


def _fixture_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "competition_id": "usa-nfl",
        "home_team_id": "team0",
        "away_team_id": "team1",
        "status": "finished",
        "stage": None,
        "matchday": None,
        "home_score_ht": None,
        "away_score_ht": None,
        "kickoff_is_estimated": False,
        "source": "test",
        "ingested_at": datetime.now(UTC),
    }
    base.update(overrides)
    return base


def _feature_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "feature_set": "nfl_v1",
        "computed_at": datetime.now(UTC),
        "source": "nfl_v1",
        "ingested_at": datetime.now(UTC),
    }
    base.update(overrides)
    return base


def test_compute_and_write_moneyline_model_persists_across_windows() -> None:
    season1 = _season_rows("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_rows("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    season3 = _season_rows("s3", _TEAMS, start=datetime(2022, 8, 1, tzinfo=UTC), seed=3)
    rows = [*season1, *season2, *season3]

    fixtures = [
        _fixture_row(
            id=row["id"],
            season=row["season"],
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            kickoff_utc=row["kickoff_utc"],
            home_score=row["home_score"],
            away_score=row["away_score"],
        )
        for row in rows
    ]
    features = [
        _feature_row(
            fixture_id=row["id"],
            competition_id="usa-nfl",
            season=row["season"],
            as_of_timestamp=row["as_of_timestamp"],
            vector=json.dumps(row["vector"]),
        )
        for row in rows
    ]
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(pd.DataFrame(fixtures))
    ParquetTableRepository(FEATURES, get_settings().parquet_dir).write(pd.DataFrame(features))

    written = compute_and_write_moneyline_model(
        "usa-nfl",
        sport="american_football",
        feature_set="nfl_v1",
        model_name_prefix="nfl_moneyline",
        calibration_method="platt",
        min_training_samples=40,
        reliability_bins=10,
    )

    assert len(written) == 2
    models = ParquetTableRepository(MODEL_REGISTRY, get_settings().parquet_dir).read()
    assert len(models) == 2
    assert set(models["model_version"]) == {"s2", "s3"}
    assert set(models["model_name"]) == {"nfl_moneyline_usa-nfl"}

    predictions = ParquetTableRepository(PREDICTIONS, get_settings().parquet_dir).read()
    assert len(predictions) == sum(written)
    assert set(predictions["market"]) == {"moneyline"}


def test_compute_and_write_moneyline_model_defaults_from_thresholds_yaml() -> None:
    season1 = _season_rows("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_rows("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    rows = [*season1, *season2]
    fixtures = [
        _fixture_row(
            id=row["id"],
            season=row["season"],
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            kickoff_utc=row["kickoff_utc"],
            home_score=row["home_score"],
            away_score=row["away_score"],
        )
        for row in rows
    ]
    features = [
        _feature_row(
            fixture_id=row["id"],
            competition_id="usa-nfl",
            season=row["season"],
            as_of_timestamp=row["as_of_timestamp"],
            vector=json.dumps(row["vector"]),
        )
        for row in rows
    ]
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(pd.DataFrame(fixtures))
    ParquetTableRepository(FEATURES, get_settings().parquet_dir).write(pd.DataFrame(features))

    written = compute_and_write_moneyline_model(
        "usa-nfl",
        sport="american_football",
        feature_set="nfl_v1",
        model_name_prefix="nfl_moneyline",
        min_training_samples=40,
        reliability_bins=10,
    )

    assert len(written) == 1
    models = ParquetTableRepository(MODEL_REGISTRY, get_settings().parquet_dir).read()
    assert models.iloc[0]["calibration_method"] == "isotonic"


def test_windows_below_threshold_are_skipped_others_persisted() -> None:
    season1 = _season_rows("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_rows("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    season3 = _season_rows("s3", _TEAMS, start=datetime(2022, 8, 1, tzinfo=UTC), seed=3)
    rows = [*season1, *season2, *season3]
    fixtures = [
        _fixture_row(
            id=row["id"],
            season=row["season"],
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            kickoff_utc=row["kickoff_utc"],
            home_score=row["home_score"],
            away_score=row["away_score"],
        )
        for row in rows
    ]
    features = [
        _feature_row(
            fixture_id=row["id"],
            competition_id="usa-nfl",
            season=row["season"],
            as_of_timestamp=row["as_of_timestamp"],
            vector=json.dumps(row["vector"]),
        )
        for row in rows
    ]
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(pd.DataFrame(fixtures))
    ParquetTableRepository(FEATURES, get_settings().parquet_dir).write(pd.DataFrame(features))

    # s1 solo (90 partidos) < 100: se salta. s1+s2 (180) >= 100: se entrena.
    written = compute_and_write_moneyline_model(
        "usa-nfl",
        sport="american_football",
        feature_set="nfl_v1",
        model_name_prefix="nfl_moneyline",
        calibration_method="isotonic",
        min_training_samples=100,
        reliability_bins=10,
    )

    assert len(written) == 1
    models = ParquetTableRepository(MODEL_REGISTRY, get_settings().parquet_dir).read()
    assert list(models["model_version"]) == ["s3"]
