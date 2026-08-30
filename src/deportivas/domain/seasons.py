"""Which season a competition is currently in, and its recent history —
what the daily incremental ingestion workflow (Fase 8) needs to fill in
``--seasons`` without a human deciding it by hand each run. Backfill and
incremental ingestion are the same CLI command with a different season
list (see ``cli.py``'s own module docstring); this is what computes that
list for the "incremental, right now" case.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from deportivas.config.catalog import Competition
from deportivas.domain.enums import Sport


def season_labels(
    competition: Competition, *, count: int = 2, today: date | None = None
) -> list[str]:
    """The ``count`` most recent season labels for ``competition``, newest
    first. The season currently under way is the one whose
    ``season_start_month`` has already passed this calendar year; before
    that month, the season under way is still last year's.

    Football labels a season by its hyphenated academic-year code — the
    fbref/football-data.co.uk convention, e.g. ``"2526"`` for 2025-26.
    Every other sport in this project labels it by its single starting
    calendar year instead — the nfl_data_py/pybaseball/sportsdataverse
    convention, e.g. ``"2025"``.
    """
    reference = today if today is not None else datetime.now(UTC).date()
    current_start_year = (
        reference.year if reference.month >= competition.season_start_month else reference.year - 1
    )
    start_years = [current_start_year - i for i in range(count)]
    if competition.sport is Sport.FOOTBALL:
        return [f"{year % 100:02d}{(year + 1) % 100:02d}" for year in start_years]
    return [str(year) for year in start_years]
