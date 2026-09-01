"""Walk-forward training and out-of-sample prediction for football's Poisson
match model.

One ``model_registry`` row per (train_seasons, validate_season) window —
rule #3: "Entrena en temporadas 1..N, valida en N+1, avanza. Métricas por
ventana" — and one ``predictions`` row per (fixture, market, selection,
line) for every fixture in that window's ``validate_season``, home team and
away team predicted with a model that has seen nothing from that season.

This model consumes ``fixtures`` directly, not the ``football_v1`` features
table: it fits its own attack/defense/home-advantage coefficients from match
scores (``features/football/dixon_coles_glm.py``) rather than reading a
feature vector, so ``feature_set`` is recorded as ``FEATURE_SET`` below to
name that self-contained representation honestly, not to claim a dependency
on ``football_v1`` that doesn't exist.

Calibration is fit once per window, separately for every (market, selection,
line) combination, using only that window's training-season in-sample raw
probabilities against their known outcomes (rule #4) — never anything from
the validate_season. A combination with no outcome variation in the
training window (an extreme over/under line where almost every match landed
the same way) is left uncalibrated for that window (``prob_calibrated`` is
``None``) rather than fit on a degenerate target.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import cast

import pandas as pd

from deportivas.config.catalog import MarketSpec, load_markets, load_thresholds
from deportivas.contracts.tables import MODEL_REGISTRY, PREDICTIONS
from deportivas.domain.enums import Market
from deportivas.domain.ids import deterministic_id
from deportivas.domain.leakage import assert_no_leakage
from deportivas.features.asof import load_fixtures
from deportivas.features.football.dixon_coles_glm import DixonColesGlmFit, fit_dixon_coles_glm
from deportivas.models.calibration import Calibrator, fit_calibrator
from deportivas.models.football.poisson import market_probabilities, score_matrix
from deportivas.models.metrics import brier_score, log_loss, reliability_curve
from deportivas.models.walkforward import WalkForwardWindow, walk_forward_windows
from deportivas.storage.factory import get_table_repository

MODEL_NAME_PREFIX = "football_poisson"
FEATURE_SET = "dixon_coles_glm_v1"
_FOOTBALL_MARKETS: tuple[str, ...] = (
    Market.ONE_X_TWO.value,
    Market.OVER_UNDER.value,
    Market.BTTS.value,
)

_CalibrationKey = tuple[str, str, float | None]  # (market, selection, line)


@dataclass(frozen=True, slots=True)
class RawPrediction:
    market: str
    selection: str
    line: float | None
    prob_raw: float
    outcome: int | None  # None cuando el partido aun no tiene resultado


def _model_name(competition_id: str) -> str:
    return f"{MODEL_NAME_PREFIX}_{competition_id}"


def _market_specs() -> list[MarketSpec]:
    catalog = load_markets()
    return [catalog.get(market_id) for market_id in _FOOTBALL_MARKETS]


def _finished_matches(fixtures: pd.DataFrame) -> list[dict[str, object]]:
    finished = fixtures[
        (fixtures["status"] == "finished")
        & fixtures["home_score"].notna()
        & fixtures["away_score"].notna()
    ]
    return [
        {
            "home_team_id": record["home_team_id"],
            "away_team_id": record["away_team_id"],
            "home_score": record["home_score"],
            "away_score": record["away_score"],
            "kickoff_utc": record["kickoff_utc"],
        }
        for record in finished.to_dict("records")
    ]


def _actual_outcome(
    market_id: str, selection: str, line: float | None, *, home_score: float, away_score: float
) -> int:
    if market_id == Market.ONE_X_TWO.value:
        if home_score > away_score:
            result = "home"
        elif home_score < away_score:
            result = "away"
        else:
            result = "draw"
        return int(selection == result)
    if market_id == Market.BTTS.value:
        both_scored = home_score >= 1 and away_score >= 1
        return int((selection == "yes") == both_scored)
    if market_id == Market.OVER_UNDER.value:
        if line is None:  # pragma: no cover - defensivo, over_under siempre trae linea
            raise ValueError("over_under requiere una linea")
        is_over = (home_score + away_score) > line
        return int((selection == "over") == is_over)
    raise ValueError(f"mercado no soportado: {market_id!r}")  # pragma: no cover - defensivo


def _raw_predictions_for_match(
    model: DixonColesGlmFit,
    *,
    home_team_id: str,
    away_team_id: str,
    home_score: float | None,
    away_score: float | None,
) -> list[RawPrediction]:
    matrix = score_matrix(model, home_team_id, away_team_id)
    rows = []
    for market in _market_specs():
        for prediction in market_probabilities(matrix, market):
            outcome = None
            if home_score is not None and away_score is not None:
                outcome = _actual_outcome(
                    market.id,
                    prediction.selection,
                    prediction.line,
                    home_score=home_score,
                    away_score=away_score,
                )
            rows.append(
                RawPrediction(
                    market.id, prediction.selection, prediction.line, prediction.prob, outcome
                )
            )
    return rows


def _group_by_key(rows: list[RawPrediction]) -> dict[_CalibrationKey, list[RawPrediction]]:
    groups: dict[_CalibrationKey, list[RawPrediction]] = defaultdict(list)
    for row in rows:
        groups[(row.market, row.selection, row.line)].append(row)
    return groups


def _fit_calibrators(
    train_predictions: list[RawPrediction], *, method: str
) -> dict[_CalibrationKey, Calibrator]:
    calibrators: dict[_CalibrationKey, Calibrator] = {}
    for key, group in _group_by_key(train_predictions).items():
        outcomes = [row.outcome for row in group]
        if len(set(outcomes)) < 2:
            continue  # sin variacion en el resultado no hay nada que calibrar
        probs = [row.prob_raw for row in group]
        calibrators[key] = fit_calibrator(method, probs, outcomes)  # type: ignore[arg-type]
    return calibrators


def _metrics_by_market(rows: list[RawPrediction], *, reliability_bins: int) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for market_id, group in _group_by_market(rows).items():
        probs = [row.prob_raw for row in group]
        outcomes = [row.outcome for row in group]
        metrics[market_id] = {
            "brier": brier_score(probs, outcomes),  # type: ignore[arg-type]
            "log_loss": log_loss(probs, outcomes),  # type: ignore[arg-type]
            "reliability": [
                asdict(b)
                for b in reliability_curve(probs, outcomes, n_bins=reliability_bins)  # type: ignore[arg-type]
            ],
        }
    return metrics


def _group_by_market(rows: list[RawPrediction]) -> dict[str, list[RawPrediction]]:
    groups: dict[str, list[RawPrediction]] = defaultdict(list)
    for row in rows:
        if row.outcome is not None:
            groups[row.market].append(row)
    return groups


@dataclass(frozen=True, slots=True)
class TrainedWindow:
    model_row: dict[str, object]
    prediction_rows: pd.DataFrame


def train_and_predict_window(
    fixtures: pd.DataFrame,
    window: WalkForwardWindow,
    *,
    competition_id: str,
    calibration_method: str,
    min_training_samples: int,
    reliability_bins: int,
) -> TrainedWindow | None:
    """Returns ``None`` when this window's training data is too thin to
    fit anything meaningful — fewer matches than ``min_training_samples``
    (``config/thresholds.yaml``'s ``calibration.min_training_samples`` by
    default, injected rather than read here so this function stays testable
    against small synthetic windows), or a GLM fit that failed to converge.
    """
    train_fixtures = fixtures[fixtures["season"].isin(window.train_seasons)]
    validate_fixtures = fixtures[fixtures["season"] == window.validate_season]

    train_matches = _finished_matches(train_fixtures)
    if len(train_matches) < min_training_samples:
        return None

    model = fit_dixon_coles_glm(train_matches)
    if model is None:
        return None

    train_predictions = [
        prediction
        for match in train_matches
        for prediction in _raw_predictions_for_match(
            model,
            home_team_id=str(match["home_team_id"]),
            away_team_id=str(match["away_team_id"]),
            home_score=float(match["home_score"]),  # type: ignore[arg-type]
            away_score=float(match["away_score"]),  # type: ignore[arg-type]
        )
    ]
    calibrators = _fit_calibrators(train_predictions, method=calibration_method)

    train_kickoffs: list[datetime] = [
        cast(datetime, match["kickoff_utc"]) for match in train_matches
    ]
    train_window_start = min(train_kickoffs)
    train_window_end = max(train_kickoffs)
    now = datetime.now(UTC)

    validate_rows = []
    validate_predictions_for_metrics: list[RawPrediction] = []
    for record in validate_fixtures.to_dict("records"):
        home_score = float(record["home_score"]) if pd.notna(record.get("home_score")) else None
        away_score = float(record["away_score"]) if pd.notna(record.get("away_score")) else None
        predictions = _raw_predictions_for_match(
            model,
            home_team_id=str(record["home_team_id"]),
            away_team_id=str(record["away_team_id"]),
            home_score=home_score,
            away_score=away_score,
        )
        validate_predictions_for_metrics.extend(predictions)
        for prediction in predictions:
            key = (prediction.market, prediction.selection, prediction.line)
            calibrator = calibrators.get(key)
            prob_calibrated = calibrator.calibrate([prediction.prob_raw])[0] if calibrator else None
            validate_rows.append(
                {
                    "id": deterministic_id(
                        str(record["id"]),
                        _model_name(competition_id),
                        window.validate_season,
                        prediction.market,
                        prediction.selection,
                        str(prediction.line),
                    ),
                    "fixture_id": record["id"],
                    "competition_id": competition_id,
                    "season": window.validate_season,
                    "model_name": _model_name(competition_id),
                    "model_version": window.validate_season,
                    "market": prediction.market,
                    "selection": prediction.selection,
                    "line": prediction.line,
                    "prob_raw": prediction.prob_raw,
                    "prob_calibrated": prob_calibrated,
                    "as_of_timestamp": train_window_end,
                    "predicted_at": now,
                    "source": _model_name(competition_id),
                    "ingested_at": now,
                }
            )

    metrics: dict[str, object] = {
        "n_train_matches": len(train_matches),
        "n_validate_matches": len(validate_fixtures),
        **_metrics_by_market(validate_predictions_for_metrics, reliability_bins=reliability_bins),
    }

    model_row = {
        "id": deterministic_id(_model_name(competition_id), window.validate_season),
        "model_name": _model_name(competition_id),
        "model_version": window.validate_season,
        "sport": "football",
        "feature_set": FEATURE_SET,
        "trained_at": now,
        "train_window_start": train_window_start,
        "train_window_end": train_window_end,
        "hyperparameters": json.dumps({"max_goals": 10}, sort_keys=True),
        "metrics": json.dumps(metrics, sort_keys=True, default=str),
        "calibration_method": calibration_method,
        "git_sha": None,
        "source": _model_name(competition_id),
        "ingested_at": now,
    }

    return TrainedWindow(model_row=model_row, prediction_rows=pd.DataFrame(validate_rows))


def compute_and_write_football_models(
    competition_id: str,
    *,
    calibration_method: str | None = None,
    min_training_samples: int | None = None,
    reliability_bins: int | None = None,
) -> list[int]:
    """Runs every walk-forward window for ``competition_id``, training a
    fresh model and writing out-of-sample predictions for each. Returns the
    number of prediction rows written per window (windows skipped for lack
    of data contribute nothing to the list, not a zero). Every ``None``
    parameter defaults from ``config/thresholds.yaml``'s ``calibration``
    section — never hardcoded here, per that file's own rule."""
    thresholds = load_thresholds().calibration
    method = calibration_method or thresholds.method
    min_samples = (
        min_training_samples
        if min_training_samples is not None
        else thresholds.min_training_samples
    )
    bins = reliability_bins if reliability_bins is not None else thresholds.reliability_bins
    fixtures = load_fixtures(competition_id)
    fixtures_for_leakage = fixtures[["id", "kickoff_utc"]].rename(columns={"id": "fixture_id"})

    model_repo = get_table_repository(MODEL_REGISTRY)
    predictions_repo = get_table_repository(PREDICTIONS, temporal_column="as_of_timestamp")

    written_counts = []
    for window in walk_forward_windows(fixtures):
        result = train_and_predict_window(
            fixtures,
            window,
            competition_id=competition_id,
            calibration_method=method,
            min_training_samples=min_samples,
            reliability_bins=bins,
        )
        if result is None:
            continue
        assert_no_leakage(result.prediction_rows, fixtures_for_leakage)
        model_repo.write(pd.DataFrame([result.model_row]))
        written_counts.append(predictions_repo.write(result.prediction_rows))
    return written_counts
