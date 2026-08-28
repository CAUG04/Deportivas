"""Row-level Pandera validation shared by every repository implementation.

Fase 10 rule: "Rechaza y registra la fila mala en vez de propagar nulos
silenciosamente al modelo." A whole-DataFrame validation failure would throw
away 999 good rows because of 1 bad one; this drops only the offending rows,
logs exactly what was wrong with each, and lets the rest through. A failure
that cannot be attributed to specific rows (a missing column, a totally wrong
dtype) is a structural bug in the caller and still raises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandera.errors as pa_errors
import structlog

from deportivas.contracts.pandera_adapter import build_schema
from deportivas.contracts.types import TableSpec

if TYPE_CHECKING:
    import pandas as pd

logger = structlog.get_logger(__name__)


class ValidationError(ValueError):
    """Raised when validation fails in a way that cannot be attributed to
    specific rows (missing/extra columns, wholly wrong dtypes)."""


def validate_rows(spec: TableSpec, rows: pd.DataFrame) -> pd.DataFrame:
    """Returns the subset of ``rows`` that conform to ``spec``'s schema.

    Rejected rows are logged (not raised) via structlog with their reason, one
    log line per distinct row index, so a bad-data problem shows up in job
    logs instead of silently reaching the model as a null.
    """
    schema = build_schema(spec)
    try:
        return schema.validate(rows, lazy=True)
    except pa_errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        if "index" not in failure_cases.columns:  # pragma: no cover - variante rara de pandera
            # No se puede atribuir el fallo a filas concretas (columna
            # ausente, esquema completamente distinto): es un bug de quien
            # llama, no un dato sucio puntual. En la practica observada,
            # pandera siempre incluye 'index' (con NaN cuando no aplica);
            # esta rama es una salvaguarda defensiva, no una que dispare el
            # comportamiento normal -ver la rama de bad_indices vacio abajo,
            # que si cubre ese escenario end-to-end.
            raise ValidationError(
                f"{spec.name}: validacion fallo sin filas atribuibles: {exc}"
            ) from exc

        bad_indices = set(failure_cases["index"].dropna().tolist())
        if not bad_indices:
            raise ValidationError(
                f"{spec.name}: validacion fallo sin filas atribuibles: {exc}"
            ) from exc

        for idx in sorted(bad_indices):
            reasons = failure_cases.loc[failure_cases["index"] == idx]
            logger.warning(
                "fila_rechazada_en_validacion",
                table=spec.name,
                row_index=idx,
                reasons=reasons[["column", "check", "failure_case"]].to_dict("records"),
            )

        clean = rows.drop(index=list(bad_indices), errors="ignore")
        if clean.empty:
            return clean
        # Revalida el resto: si sigue fallando, el problema es estructural
        # (no atribuible a las filas ya descartadas) y debe propagarse.
        return schema.validate(clean, lazy=True)
