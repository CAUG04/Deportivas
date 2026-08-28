"""Point-in-time leakage guard.

Regla innegociable #1 del proyecto: ninguna feature puede usar informacion
posterior al inicio del partido que predice. Esta es la unica funcion en todo
el proyecto autorizada a decidir si un vector de features es valido para
entrenar o predecir; el motor de features (Fase 2) y el motor de backtest
(Fase 4) llaman a esta funcion, no reimplementan la comparacion.

Se comprueba a proposito con un margen estricto (``>=`` es leakage): una
feature calculada en el mismo instante del kickoff ya pudo ver el once
inicial o el movimiento de cuotas de ultima hora, que no estaban disponibles
cuando "decidimos" apostar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class LeakageError(ValueError):
    """Raised when one or more features are computed at or after kickoff."""


@dataclass(frozen=True, slots=True)
class LeakageViolation:
    fixture_id: str
    as_of_timestamp: object
    kickoff_utc: object

    def __str__(self) -> str:
        return (
            f"fixture {self.fixture_id}: as_of_timestamp={self.as_of_timestamp} "
            f">= kickoff_utc={self.kickoff_utc}"
        )


def find_leakage(
    features: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    fixture_id_col: str = "fixture_id",
    as_of_col: str = "as_of_timestamp",
    kickoff_col: str = "kickoff_utc",
) -> list[LeakageViolation]:
    """Returns every row in ``features`` computed at or after its fixture's kickoff.

    Both timestamp columns must be timezone-aware UTC — comparing naive and
    aware timestamps raises rather than silently assuming a timezone, because a
    silently-wrong timezone is exactly the kind of leakage this function
    exists to catch.
    """
    import pandas as pd

    for label, df, col in (("features", features, as_of_col), ("fixtures", fixtures, kickoff_col)):
        if col not in df.columns:
            raise KeyError(f"{label}: falta la columna {col!r}")
        series = df[col]
        if not pd.api.types.is_datetime64_any_dtype(series):
            raise TypeError(f"{label}.{col}: debe ser datetime, no {series.dtype}")
        if series.dt.tz is None:
            raise ValueError(f"{label}.{col}: debe ser timezone-aware (UTC), no naive")

    merged = features[[fixture_id_col, as_of_col]].merge(
        fixtures[[fixture_id_col, kickoff_col]],
        on=fixture_id_col,
        how="inner",
        validate="many_to_one",
    )
    leaked = merged[merged[as_of_col] >= merged[kickoff_col]]
    return [
        LeakageViolation(row[fixture_id_col], row[as_of_col], row[kickoff_col])
        for _, row in leaked.iterrows()
    ]


def assert_no_leakage(
    features: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    fixture_id_col: str = "fixture_id",
    as_of_col: str = "as_of_timestamp",
    kickoff_col: str = "kickoff_utc",
) -> None:
    """Raises :class:`LeakageError` naming every offending fixture, or returns silently."""
    violations = find_leakage(
        features,
        fixtures,
        fixture_id_col=fixture_id_col,
        as_of_col=as_of_col,
        kickoff_col=kickoff_col,
    )
    if violations:
        detail = "\n".join(f"  - {v}" for v in violations)
        raise LeakageError(
            f"{len(violations)} feature(s) con as_of_timestamp >= kickoff_utc:\n{detail}"
        )
