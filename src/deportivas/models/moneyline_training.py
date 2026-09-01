"""Walk-forward training and out-of-sample prediction for a sport's
moneyline classifier — shared by NFL, NBA, NHL and MLB (``models/nfl/``,
``models/nba/``, ``models/nhl/``, ``models/mlb/`` are thin wrappers that
only supply ``sport``, ``feature_set`` and a model-name prefix).

Same discipline as football's Poisson model (``models/football/train.py``):
one ``model_registry`` row per (train_seasons, validate_season) window
(rule #3), a single model frozen at the end of training and used unchanged
across the whole validation season, and calibration fit only on that
window's training-season in-sample probabilities (rule #4).

Unlike football, this reads already-computed feature vectors (``nfl_v1``,
``nba_v1``, ...) rather than fitting its own coefficients from raw scores —
whatever signals a sport's feature pipeline produced are what the
classifier sees, nothing more.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import pandas as pd

from deportivas.config.catalog import load_thresholds
from deportivas.contracts.tables import MODEL_REGISTRY, PREDICTIONS
from deportivas.domain.enums import Market, Selection
from deportivas.domain.ids import deterministic_id
from deportivas.domain.leakage import assert_no_leakage
from deportivas.features.asof import load_fixtures
from deportivas.models.calibration import Calibrator, fit_calibrator
from deportivas.models.features_loader import load_feature_vectors
from deportivas.models.metrics import brier_score, log_loss, reliability_curve
from deportivas.models.moneyline import fit_moneyline_model
from deportivas.models.walkforward import WalkForwardWindow, walk_forward_windows
from deportivas.storage.factory import get_table_repository


def _home_win(home_score: float, away_score: float) -> int | None:
    """``None`` for a tie: uninformative for a home/away classifier, and
    rare enough in these sports (NFL only, under current rules) that
    excluding it from training is simpler than modelling a third outcome
    the ``moneyline`` market doesn't even have a selection for."""
    if home_score > away_score:
        return 1
    if home_score < away_score:
        return 0
    return None


@dataclass(frozen=True, slots=True)
class TrainedMoneylineWindow:
    model_row: dict[str, object]
    prediction_rows: pd.DataFrame


def train_and_predict_moneyline_window(
    features_df: pd.DataFrame,
    window: WalkForwardWindow,
    *,
    competition_id: str,
    sport: str,
    feature_set: str,
    model_name: str,
    calibration_method: str,
    min_training_samples: int,
    reliability_bins: int,
) -> TrainedMoneylineWindow | None:
    """``features_df`` is ``features_loader.load_feature_vectors``'s own
    output. Returns ``None`` when this window's training data is too thin —
    fewer finished, decisive (non-tied) matches than ``min_training_samples``,
    or every one of them had the same outcome (nothing for a classifier to
    learn)."""
    train_df = features_df[features_df["season"].isin(window.train_seasons)]
    validate_df = features_df[features_df["season"] == window.validate_season]

    train_examples = []
    for record in train_df.to_dict("records"):
        if (
            record["status"] != "finished"
            or pd.isna(record["home_score"])
            or pd.isna(record["away_score"])
        ):
            continue
        outcome = _home_win(float(record["home_score"]), float(record["away_score"]))
        if outcome is not None:
            train_examples.append((record["vector"], outcome, record["as_of_timestamp"]))

    if len(train_examples) < min_training_samples:
        return None
    vectors = [example[0] for example in train_examples]
    outcomes = [example[1] for example in train_examples]
    if len(set(outcomes)) < 2:
        return None

    model = fit_moneyline_model(vectors, outcomes)
    raw_train_probs = [model.predict_proba_home(vector) for vector in vectors]
    calibrator: Calibrator | None = fit_calibrator(calibration_method, raw_train_probs, outcomes)

    train_kickoffs = [example[2] for example in train_examples]
    train_window_start = min(train_kickoffs)
    train_window_end = max(train_kickoffs)
    now = datetime.now(UTC)

    validate_rows = []
    metrics_probs: list[float] = []
    metrics_outcomes: list[int] = []
    for record in validate_df.to_dict("records"):
        vector = record["vector"]
        prob_home_raw = model.predict_proba_home(vector)
        prob_away_raw = 1.0 - prob_home_raw
        calibrated_home = calibrator.calibrate([prob_home_raw])[0] if calibrator else None
        calibrated_away = 1.0 - calibrated_home if calibrated_home is not None else None

        outcome = None
        if pd.notna(record.get("home_score")) and pd.notna(record.get("away_score")):
            outcome = _home_win(float(record["home_score"]), float(record["away_score"]))
        if outcome is not None:
            metrics_probs.extend([prob_home_raw, prob_away_raw])
            metrics_outcomes.extend([outcome, 1 - outcome])

        for selection, prob_raw, prob_calibrated in (
            (Selection.HOME.value, prob_home_raw, calibrated_home),
            (Selection.AWAY.value, prob_away_raw, calibrated_away),
        ):
            validate_rows.append(
                {
                    "id": deterministic_id(
                        str(record["id"]),
                        model_name,
                        window.validate_season,
                        Market.MONEYLINE.value,
                        selection,
                        "None",
                    ),
                    "fixture_id": record["id"],
                    "competition_id": competition_id,
                    "season": window.validate_season,
                    "model_name": model_name,
                    "model_version": window.validate_season,
                    "market": Market.MONEYLINE.value,
                    "selection": selection,
                    "line": None,
                    "prob_raw": prob_raw,
                    "prob_calibrated": prob_calibrated,
                    "as_of_timestamp": train_window_end,
                    "predicted_at": now,
                    "source": model_name,
                    "ingested_at": now,
                }
            )

    metrics: dict[str, object] = {
        "n_train_matches": len(train_examples),
        "n_validate_matches": len(validate_df),
    }
    if metrics_outcomes:
        metrics["moneyline"] = {
            "brier": brier_score(metrics_probs, metrics_outcomes),
            "log_loss": log_loss(metrics_probs, metrics_outcomes),
            "reliability": [
                asdict(b)
                for b in reliability_curve(metrics_probs, metrics_outcomes, n_bins=reliability_bins)
            ],
        }

    model_row = {
        "id": deterministic_id(model_name, window.validate_season),
        "model_name": model_name,
        "model_version": window.validate_season,
        "sport": sport,
        "feature_set": feature_set,
        "trained_at": now,
        "train_window_start": train_window_start,
        "train_window_end": train_window_end,
        "hyperparameters": json.dumps({"classifier": "logistic_regression"}, sort_keys=True),
        "metrics": json.dumps(metrics, sort_keys=True, default=str),
        "calibration_method": calibration_method,
        "git_sha": None,
        "source": model_name,
        "ingested_at": now,
    }

    return TrainedMoneylineWindow(model_row=model_row, prediction_rows=pd.DataFrame(validate_rows))


def compute_and_write_moneyline_model(
    competition_id: str,
    *,
    sport: str,
    feature_set: str,
    model_name_prefix: str,
    calibration_method: str | None = None,
    min_training_samples: int | None = None,
    reliability_bins: int | None = None,
) -> list[int]:
    """Runs every walk-forward window for ``competition_id``, training a
    fresh moneyline classifier and writing out-of-sample predictions for
    each. Returns the number of prediction rows written per window (windows
    skipped for lack of data contribute nothing to the list, not a zero).
    Every ``None`` parameter defaults from ``config/thresholds.yaml``'s
    ``calibration`` section."""
    thresholds = load_thresholds().calibration
    method = calibration_method or thresholds.method
    min_samples = (
        min_training_samples
        if min_training_samples is not None
        else thresholds.min_training_samples
    )
    bins = reliability_bins if reliability_bins is not None else thresholds.reliability_bins
    model_name = f"{model_name_prefix}_{competition_id}"

    features_df = load_feature_vectors(competition_id, feature_set)
    fixtures_for_leakage = load_fixtures(competition_id)[["id", "kickoff_utc"]].rename(
        columns={"id": "fixture_id"}
    )

    model_repo = get_table_repository(MODEL_REGISTRY)
    predictions_repo = get_table_repository(PREDICTIONS, temporal_column="as_of_timestamp")

    written_counts = []
    for window in walk_forward_windows(features_df):
        result = train_and_predict_moneyline_window(
            features_df,
            window,
            competition_id=competition_id,
            sport=sport,
            feature_set=feature_set,
            model_name=model_name,
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
