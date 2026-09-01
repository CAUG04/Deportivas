"""Read-only FastAPI layer over :mod:`deportivas.api.views`.

This exists for local development and as an alternative to the static
export (:mod:`deportivas.export.json_export`) — but the static export, not
this, is what ``frontend/`` actually reads in production (see
``Settings.export_dir``'s own docstring): a live API means a process that
has to stay running, which the $0 static-site deployment this project
targets has no room for. Run this with ``uvicorn deportivas.api.app:app``
when you want live queries against local data instead of regenerating the
JSON export on every change.

No authentication: this is a single-operator research tool, not a public
multi-tenant service. Do not expose it beyond a trusted network without
adding one.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from deportivas.api import views
from deportivas.config.settings import get_settings

app = FastAPI(title="Deportivas API", version="0.1.0")

_cors_origins = get_settings().cors_allow_origins
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_cors_origins),
        allow_methods=["GET"],
        allow_headers=["*"],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/competitions", response_model=list[views.CompetitionOut])
def list_competitions() -> list[views.CompetitionOut]:
    return views.list_competitions()


@app.get("/competitions/{competition_id}/signals", response_model=list[views.SignalOut])
def list_signals(
    competition_id: str,
    tier: str | None = None,
    market: str | None = None,
    only_actionable: bool = True,
    limit: int = 200,
) -> list[views.SignalOut]:
    try:
        return views.list_signals(
            competition_id, tier=tier, market=market, only_actionable=only_actionable, limit=limit
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/competitions/{competition_id}/backtest", response_model=views.BacktestReportOut)
def backtest_report(competition_id: str) -> views.BacktestReportOut:
    try:
        return views.backtest_report(competition_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
