"""xG conceded, adjusted for the strength of the attacks faced.

A raw rolling "xG against" average (``xg_rolling.py``) treats a team that
conceded a lot against a run of the league's best attacks the same as one
that conceded the same amount against its weakest sides. This adjusts for
that: for each match in a team's trailing window, it subtracts the
opponent's pre-match attack rating (from ``strength.py``, already
point-in-time correct for that fixture) from the xG conceded in it, then
averages. A team whose adjusted number improves relative to its raw
xG-against is doing better than the scoreline suggests once opponent
quality is taken into account.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20)


def compute_opponent_adjusted_defense(
    fixtures: pd.DataFrame,
    team_match_stats: pd.DataFrame,
    strength_by_fixture: pd.DataFrame,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """``strength_by_fixture`` is ``strength.compute_strength()``'s own output
    (fixture_id, vector with strength_attack_home/away) — computed once by the
    pipeline and passed in here, rather than refit, to keep the two features
    consistent and avoid fitting the same GLM twice.
    """
    xg_by_fixture_team = {
        (row["fixture_id"], row["team_id"]): row["xg"]
        for row in team_match_stats.to_dict("records")
        if pd.notna(row["xg"])
    }
    strength_by_fixture_id = {
        row["fixture_id"]: row["vector"] for row in strength_by_fixture.to_dict("records")
    }

    # history[team] = [(xg_conceded, opponent_attack_rating), ...] cronologico
    history: dict[str, list[tuple[float, float]]] = defaultdict(list)
    rows: list[dict[str, object]] = []

    for record in fixtures.to_dict("records"):
        fixture_id = record["id"]
        home, away = record["home_team_id"], record["away_team_id"]

        vector: dict[str, object] = {}
        for side, team in (("home", home), ("away", away)):
            recent_first = list(reversed(history[team]))
            for window in windows:
                window_slice = recent_first[:window]
                if window_slice:
                    avg_conceded = sum(v[0] for v in window_slice) / len(window_slice)
                    avg_opponent_attack = sum(v[1] for v in window_slice) / len(window_slice)
                    vector[f"defense_adjusted_{window}_{side}"] = avg_conceded - avg_opponent_attack
                else:
                    vector[f"defense_adjusted_{window}_{side}"] = None

        rows.append({"fixture_id": fixture_id, "vector": vector})

        home_xg = xg_by_fixture_team.get((fixture_id, home))
        away_xg = xg_by_fixture_team.get((fixture_id, away))
        if home_xg is not None and away_xg is not None:
            match_strength = strength_by_fixture_id.get(fixture_id, {})
            away_attack = match_strength.get("strength_attack_away", 0.0)
            home_attack = match_strength.get("strength_attack_home", 0.0)
            history[home].append((away_xg, away_attack))
            history[away].append((home_xg, home_attack))

    return pd.DataFrame(rows)
