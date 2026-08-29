"""train.py: walk-forward training and out-of-sample prediction for
football's Poisson match model."""

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
from deportivas.contracts.tables import FIXTURES, MODEL_REGISTRY, PREDICTIONS
from deportivas.models.football.train import (
    TrainedWindow,
    compute_and_write_football_models,
    train_and_predict_window,
)
from deportivas.models.walkforward import WalkForwardWindow, walk_forward_windows
from deportivas.storage.duckdb_repo.repository import ParquetTableRepository


@pytest.fixture(autouse=True)
def _settings_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEPORTIVAS_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _season_fixtures(
    season: str, teams: list[str], *, start: datetime, seed: int
) -> list[dict[str, object]]:
    """A full round-robin among ``teams``, every match finished, scores drawn
    from a fixed seed so results (and therefore calibration behaviour) are
    deterministic across test runs."""
    rng = random.Random(seed)
    rows = []
    kickoff = start
    for idx, (home, away) in enumerate(itertools.permutations(teams, 2)):
        rows.append(
            {
                "id": f"{season}-{idx}",
                "competition_id": "eng-premier-league",
                "season": season,
                "home_team_id": home,
                "away_team_id": away,
                "kickoff_utc": kickoff,
                "status": "finished",
                "home_score": rng.randint(0, 4),
                "away_score": rng.randint(0, 4),
            }
        )
        kickoff += timedelta(hours=6)
    return rows


def _fixtures_df(*seasons_rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([row for rows in seasons_rows for row in rows])


_TEAMS = [f"team{i}" for i in range(10)]  # 10*9 = 90 partidos por temporada


def test_window_returns_none_below_min_training_samples() -> None:
    season1 = _season_fixtures("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_fixtures("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    fixtures = _fixtures_df(season1, season2)
    window = walk_forward_windows(fixtures)[0]

    result = train_and_predict_window(
        fixtures,
        window,
        competition_id="eng-premier-league",
        calibration_method="isotonic",
        min_training_samples=1000,  # mas partidos de los que hay
        reliability_bins=10,
    )

    assert result is None


def _trained_window(
    **overrides: object,
) -> tuple[TrainedWindow, pd.DataFrame, WalkForwardWindow]:
    season1 = _season_fixtures("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_fixtures("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    fixtures = _fixtures_df(season1, season2)
    window = walk_forward_windows(fixtures)[0]
    kwargs: dict[str, object] = {
        "competition_id": "eng-premier-league",
        "calibration_method": "isotonic",
        "min_training_samples": 40,
        "reliability_bins": 10,
    }
    kwargs.update(overrides)
    result = train_and_predict_window(fixtures, window, **kwargs)  # type: ignore[arg-type]
    assert result is not None
    return result, fixtures, window


def test_predicts_every_market_and_selection_per_validate_fixture() -> None:
    result, fixtures, window = _trained_window()
    validate_count = int((fixtures["season"] == window.validate_season).sum())
    # 1x2 (3) + btts (2) + over_under (2 selecciones * 3 lineas = 6) = 11 filas/partido
    assert len(result.prediction_rows) == validate_count * 11


def test_as_of_timestamp_is_constant_and_equals_training_cutoff() -> None:
    result, fixtures, window = _trained_window()
    train_fixtures = fixtures[fixtures["season"].isin(window.train_seasons)]
    expected = train_fixtures["kickoff_utc"].max()
    assert result.prediction_rows["as_of_timestamp"].nunique() == 1
    assert result.prediction_rows["as_of_timestamp"].iloc[0] == expected


def test_as_of_strictly_before_every_validate_kickoff() -> None:
    result, fixtures, window = _trained_window()
    validate_fixtures = fixtures[fixtures["season"] == window.validate_season][
        ["id", "kickoff_utc"]
    ].rename(columns={"id": "fixture_id"})
    merged = result.prediction_rows.merge(validate_fixtures, on="fixture_id")
    assert (merged["as_of_timestamp"] < merged["kickoff_utc"]).all()


def test_prob_raw_is_a_valid_probability_everywhere() -> None:
    result, _, _ = _trained_window()
    assert result.prediction_rows["prob_raw"].between(0.0, 1.0).all()


def test_prob_calibrated_is_a_valid_probability_where_present() -> None:
    result, _, _ = _trained_window()
    calibrated = result.prediction_rows["prob_calibrated"].dropna()
    assert len(calibrated) > 0
    assert calibrated.between(0.0, 1.0).all()


def test_one_x_two_probabilities_sum_to_one_per_fixture() -> None:
    result, _, _ = _trained_window()
    one_x_two = result.prediction_rows[result.prediction_rows["market"] == "1x2"]
    sums = one_x_two.groupby("fixture_id")["prob_raw"].sum()
    assert (sums.round(6) == 1.0).all()


def test_model_row_names_include_competition_id() -> None:
    result, _, window = _trained_window()
    assert result.model_row["model_name"] == "football_poisson_eng-premier-league"
    assert result.model_row["model_version"] == window.validate_season
    assert result.model_row["sport"] == "football"
    assert result.model_row["calibration_method"] == "isotonic"


def test_model_row_metrics_are_valid_json_with_every_market() -> None:
    result, _, _ = _trained_window()
    metrics = json.loads(result.model_row["metrics"])  # type: ignore[arg-type]
    assert set(metrics) >= {"1x2", "over_under", "btts", "n_train_matches", "n_validate_matches"}
    for market_id in ("1x2", "over_under", "btts"):
        assert "brier" in metrics[market_id]
        assert "log_loss" in metrics[market_id]
        assert "reliability" in metrics[market_id]


def test_model_row_hyperparameters_are_valid_json() -> None:
    result, _, _ = _trained_window()
    hyperparameters = json.loads(result.model_row["hyperparameters"])  # type: ignore[arg-type]
    assert hyperparameters["max_goals"] == 10


def test_glm_fit_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deportivas.models.football.train.fit_dixon_coles_glm", lambda matches: None
    )
    season1 = _season_fixtures("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_fixtures("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    fixtures = _fixtures_df(season1, season2)
    window = walk_forward_windows(fixtures)[0]

    result = train_and_predict_window(
        fixtures,
        window,
        competition_id="eng-premier-league",
        calibration_method="isotonic",
        min_training_samples=40,
        reliability_bins=10,
    )

    assert result is None


def _fixture_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "competition_id": "eng-premier-league",
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


def test_compute_and_write_football_models_persists_across_windows() -> None:
    season1 = _season_fixtures("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_fixtures("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    season3 = _season_fixtures("s3", _TEAMS, start=datetime(2022, 8, 1, tzinfo=UTC), seed=3)
    rows = [_fixture_row(**row) for row in (*season1, *season2, *season3)]
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(pd.DataFrame(rows))

    written = compute_and_write_football_models(
        "eng-premier-league",
        calibration_method="platt",
        min_training_samples=40,
        reliability_bins=10,
    )

    # 3 temporadas -> 2 ventanas walk-forward (s1->s2, s1+s2->s3)
    assert len(written) == 2
    assert all(count > 0 for count in written)

    models = ParquetTableRepository(MODEL_REGISTRY, get_settings().parquet_dir).read()
    assert len(models) == 2
    assert set(models["model_version"]) == {"s2", "s3"}

    predictions = ParquetTableRepository(PREDICTIONS, get_settings().parquet_dir).read()
    assert len(predictions) == sum(written)
    assert set(predictions["competition_id"]) == {"eng-premier-league"}


def test_compute_and_write_football_models_defaults_from_thresholds_yaml() -> None:
    """Without an explicit calibration_method, the real config value is used
    (isotonic, per config/thresholds.yaml) — proven by not crashing and by
    the persisted model_row recording that method."""
    season1 = _season_fixtures("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_fixtures("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    rows = [_fixture_row(**row) for row in (*season1, *season2)]
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(pd.DataFrame(rows))

    written = compute_and_write_football_models(
        "eng-premier-league", min_training_samples=40, reliability_bins=10
    )

    assert len(written) == 1
    models = ParquetTableRepository(MODEL_REGISTRY, get_settings().parquet_dir).read()
    assert models.iloc[0]["calibration_method"] == "isotonic"


def test_unfinished_validate_fixture_still_gets_predictions_but_no_outcome() -> None:
    season1 = _season_fixtures("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_fixtures("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    season2[0] = {**season2[0], "status": "scheduled", "home_score": None, "away_score": None}
    fixtures = _fixtures_df(season1, season2)
    window = walk_forward_windows(fixtures)[0]

    result = train_and_predict_window(
        fixtures,
        window,
        competition_id="eng-premier-league",
        calibration_method="isotonic",
        min_training_samples=40,
        reliability_bins=10,
    )

    assert result is not None
    validate_count = len(season2)
    # las predicciones se escriben para todos los partidos, jugados o no
    assert len(result.prediction_rows) == validate_count * 11

    metrics = json.loads(result.model_row["metrics"])  # type: ignore[arg-type]
    one_x_two_total_count = sum(b["count"] for b in metrics["1x2"]["reliability"])
    # el partido sin resultado no aporta ninguna de sus 3 filas a las metricas
    assert one_x_two_total_count == (validate_count - 1) * 3


def test_constant_scoreline_leaves_every_selection_uncalibrated() -> None:
    """A training window with zero outcome variance anywhere (every match
    ends 3-3) must skip calibrating every (market, selection, line) rather
    than fitting a degenerate target."""
    teams = _TEAMS
    rows = []
    kickoff = datetime(2020, 8, 1, tzinfo=UTC)
    for idx, (home, away) in enumerate(itertools.permutations(teams, 2)):
        rows.append(
            {
                "id": f"flat-{idx}",
                "competition_id": "eng-premier-league",
                "season": "flat1",
                "home_team_id": home,
                "away_team_id": away,
                "kickoff_utc": kickoff,
                "status": "finished",
                "home_score": 3,
                "away_score": 3,
            }
        )
        kickoff += timedelta(hours=6)
    season2 = _season_fixtures("flat2", teams, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    fixtures = _fixtures_df(rows, season2)
    window = walk_forward_windows(fixtures)[0]

    result = train_and_predict_window(
        fixtures,
        window,
        competition_id="eng-premier-league",
        calibration_method="isotonic",
        min_training_samples=40,
        reliability_bins=10,
    )

    assert result is not None
    assert result.prediction_rows["prob_calibrated"].isna().all()


def test_windows_below_threshold_are_skipped_others_persisted() -> None:
    season1 = _season_fixtures("s1", _TEAMS, start=datetime(2020, 8, 1, tzinfo=UTC), seed=1)
    season2 = _season_fixtures("s2", _TEAMS, start=datetime(2021, 8, 1, tzinfo=UTC), seed=2)
    season3 = _season_fixtures("s3", _TEAMS, start=datetime(2022, 8, 1, tzinfo=UTC), seed=3)
    rows = [_fixture_row(**row) for row in (*season1, *season2, *season3)]
    ParquetTableRepository(FIXTURES, get_settings().parquet_dir).write(pd.DataFrame(rows))

    # s1 solo (90 partidos) < 100: se salta. s1+s2 (180) >= 100: se entrena.
    written = compute_and_write_football_models(
        "eng-premier-league",
        calibration_method="isotonic",
        min_training_samples=100,
        reliability_bins=10,
    )

    assert len(written) == 1
    models = ParquetTableRepository(MODEL_REGISTRY, get_settings().parquet_dir).read()
    assert list(models["model_version"]) == ["s3"]
