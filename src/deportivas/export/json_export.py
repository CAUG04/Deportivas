"""Writes the same views :mod:`deportivas.api.views` exposes live to static
JSON files under ``Settings.export_dir`` (``frontend/public/data/`` by
default) — what the $0 static frontend (Fase 7) actually reads, so it never
needs a server running to show current signals or the backtest report.

Unlike the live API's ``only_actionable=True`` default, the export writes
*every* tier, ``descartar`` included: the export is a point-in-time
snapshot a static site can't re-query with different filters later, so
leaving a tier out here would make it permanently invisible to the
frontend rather than just uninteresting by default.
"""

from __future__ import annotations

import json
from pathlib import Path

from deportivas.api import views
from deportivas.config.settings import get_settings

_EXPORT_SIGNALS_LIMIT = 1000


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def export_competition(competition_id: str, *, out_dir: Path | None = None) -> dict[str, Path]:
    """Writes ``{competition_id}/signals.json`` (every tier) and
    ``{competition_id}/backtest.json`` under ``out_dir``
    (``Settings.export_dir`` when omitted). Raises ``KeyError`` for an
    unknown ``competition_id`` — same as ``api/views.py``, and for the same
    reason: a missing file the frontend can't distinguish from "no signals
    yet" is worse than a loud failure here, at export time."""
    base = out_dir if out_dir is not None else get_settings().export_dir
    competition_dir = base / competition_id

    signals = views.list_signals(competition_id, only_actionable=False, limit=_EXPORT_SIGNALS_LIMIT)
    report = views.backtest_report(competition_id)

    signals_path = competition_dir / "signals.json"
    backtest_path = competition_dir / "backtest.json"
    _write_json(signals_path, [s.model_dump(mode="json") for s in signals])
    _write_json(backtest_path, report.model_dump(mode="json"))
    return {"signals": signals_path, "backtest": backtest_path}


def export_all(
    competition_ids: list[str] | None = None, *, out_dir: Path | None = None
) -> dict[str, Path]:
    """Writes the global ``competitions.json`` plus ``export_competition``'s
    two files for each of ``competition_ids`` (every enabled competition
    when omitted). Returns every path written, keyed
    ``"competitions"``/``"{competition_id}:signals"``/``"{competition_id}:backtest"``."""
    base = out_dir if out_dir is not None else get_settings().export_dir
    competitions = views.list_competitions()

    written: dict[str, Path] = {}
    competitions_path = base / "competitions.json"
    _write_json(competitions_path, [c.model_dump(mode="json") for c in competitions])
    written["competitions"] = competitions_path

    ids = competition_ids if competition_ids is not None else [c.id for c in competitions]
    for competition_id in ids:
        for label, path in export_competition(competition_id, out_dir=base).items():
            written[f"{competition_id}:{label}"] = path
    return written
