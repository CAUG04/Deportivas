#!/usr/bin/env bash
set -uo pipefail

# Pipeline diario (Fase 8/9): un solo script para todas las competiciones
# habilitadas de config/competitions.yaml, invocado por
# .github/workflows/daily.yml -- nunca a mano por temporada. Ver el
# docstring de cli.py: backfill e incremental son el mismo comando de
# ingesta con --seasons distinto; este script decide QUE competicion, QUE
# fuente y EN QUE ORDEN, la logica de negocio vive en el paquete Python.
#
# Dos fases:
#   A. Ingesta de calendario/stats + entrenamiento de modelos, solo para las
#      competiciones cuya cadencia toca hoy: "daily" todos los dias,
#      "weekly" solo el dia ISO que fija WEEKLY_DAY (domingo=7 por
#      defecto) -- la palanca que el propio comentario de
#      config/competitions.yaml describe contra el limite de tiempo de
#      GitHub Actions.
#   B. Cierre de linea + senales + liquidacion, todos los dias, para TODA
#      competicion habilitada -- no depende de si la fase A corrio hoy: un
#      partido puede arrancar y terminar entre dos refrescos de calendario
#      de una competicion "weekly".
#
# No usa "set -e": cada paso externo se aisla explicitamente con "|| aviso"
# para que una fuente rota (red caida, scraping bloqueado, credencial
# vencida) no tumbe el resto de competiciones ni de fases.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=scripts/lib_common.sh
source scripts/lib_common.sh

WEEKLY_DAY="${WEEKLY_DAY:-7}" # ISO 8601: 1=lunes .. 7=domingo
today_dow="$(date -u +%u)"

# Cuantas temporadas pedir por competicion. 2 es lo incremental de cada dia:
# la actual y la anterior, que es todo lo que puede cambiar. Subirlo convierte
# esta misma corrida en el backfill que competitions.yaml lleva declarando en
# "seasons_back" -- ver el README, "Backfill: por que 2 temporadas no bastan".
#
# No es cosmetico: el walk-forward entrena en las temporadas 1..N y valida en
# N+1, asi que con 2 temporadas solo hay UNA ventana y entrena con una sola
# temporada. Una liga europea son 306-380 partidos por temporada y
# thresholds.yaml pide 500 para calibrar, asi que con 2 temporadas ningun
# modelo de futbol puede entrenar nunca -- estructuralmente, no "todavia no".
# NBA/NHL no lo notaron porque una temporada suya son ~1300 partidos.
SEASONS_COUNT="${SEASONS_COUNT:-2}"

run() {
    echo "+ $*"
    "$@" || echo "  aviso: fallo -- $*"
}

load_competitions

echo "###### Fase A: ingesta de calendario/stats + entrenamiento ######"
while IFS= read -r competition; do
    id="$(jq -r '.id' <<<"$competition")"
    sport="$(jq -r '.sport' <<<"$competition")"
    refresh="$(jq -r '.refresh' <<<"$competition")"
    sources="$(jq -c '.sources' <<<"$competition")"

    if [ "$refresh" = "weekly" ] && [ "$today_dow" != "$WEEKLY_DAY" ]; then
        continue
    fi

    echo "--- ${id} (${sport}, ${refresh}) ---"
    seasons="$(uv run deportivas current-seasons --competition-id "$id" --count "$SEASONS_COUNT")"

    case "$sport" in
    football)
        # sources.fbref/understat/match_history/espn son solo alias legibles
        # (lo que ingest/soccerdata_config.py inyecta en el league_dict.json
        # de soccerdata); se usan aqui unicamente para decidir SI esa fuente
        # aplica a esta competicion. Todo lector de soccerdata espera la
        # CLAVE de LEAGUE_DICT (soccerdata_key, p.ej. "ENG-Premier League")
        # en --*-league -- pasarle el alias en vez de la clave es el bug que
        # sources-health.yml encontro en produccion la primera vez que corrio.
        soccerdata_key="$(jq -r '.soccerdata_key // empty' <<<"$sources")"
        fbref_league="$(jq -r '.fbref // empty' <<<"$sources")"
        understat_league="$(jq -r '.understat // empty' <<<"$sources")"
        match_history_league="$(jq -r '.match_history // empty' <<<"$sources")"
        espn_league="$(jq -r '.espn // empty' <<<"$sources")"

        if [ -z "$soccerdata_key" ]; then
            echo "  aviso: ${id} sin sources.soccerdata_key en competitions.yaml -- se salta futbol"
        else
            # DEPORTIVAS_FBREF_ENABLED=false en daily.yml: FBref nunca pasa
            # desde un runner de GitHub Actions (CAPTCHA, ver README "FBref
            # bloquea con CAPTCHA..."), asi que intentarlo ahi es tiempo de
            # CI gastado en un resultado ya conocido. Sigue en true por
            # defecto para que este mismo script, corrido a mano desde una
            # IP normal, si intente FBref.
            if [ -n "$fbref_league" ] && [ "${DEPORTIVAS_FBREF_ENABLED:-true}" = "true" ]; then
                run uv run deportivas ingest fbref-schedule \
                    --competition-id "$id" --fbref-league "$soccerdata_key" --seasons "$seasons"
                run uv run deportivas ingest fbref-stats \
                    --competition-id "$id" --fbref-league "$soccerdata_key" --seasons "$seasons"
            fi
            if [ -n "$understat_league" ]; then
                run uv run deportivas ingest understat-stats \
                    --competition-id "$id" --understat-league "$soccerdata_key" --seasons "$seasons"
            fi
            if [ -n "$match_history_league" ]; then
                run uv run deportivas ingest footballdata-games \
                    --competition-id "$id" --match-history-league "$soccerdata_key" --seasons "$seasons"
                run uv run deportivas ingest footballdata-odds \
                    --competition-id "$id" --match-history-league "$soccerdata_key" --seasons "$seasons"
            elif [ -n "$espn_league" ]; then
                # Sin football-data.co.uk (UEFA, Liga BetPlay): ESPN es el
                # respaldo de calendario -- ver README, "unica fuente para
                # Liga BetPlay Dimayor".
                run uv run deportivas ingest espn-schedule \
                    --competition-id "$id" --espn-league "$soccerdata_key" --seasons "$seasons"
            fi
        fi
        run uv run deportivas features compute-football --competition-id "$id"
        run uv run deportivas models train-football --competition-id "$id"
        ;;
    american_football)
        if [ "$(jq -r '.nfl_data_py // false' <<<"$sources")" = "true" ]; then
            run uv run deportivas ingest nfl-schedule --competition-id "$id" --seasons "$seasons"
            run uv run deportivas ingest nfl-team-game-stats --competition-id "$id" --seasons "$seasons"
        fi
        run uv run deportivas features compute-nfl --competition-id "$id"
        run uv run deportivas models train-nfl --competition-id "$id"
        ;;
    basketball)
        if [ "$(jq -r '.sportsdataverse // empty' <<<"$sources")" = "nba" ]; then
            run uv run deportivas ingest nba-schedule --competition-id "$id" --seasons "$seasons"
        fi
        run uv run deportivas features compute-nba --competition-id "$id"
        run uv run deportivas models train-nba --competition-id "$id"
        ;;
    ice_hockey)
        if [ "$(jq -r '.sportsdataverse // empty' <<<"$sources")" = "nhl" ]; then
            run uv run deportivas ingest nhl-schedule --competition-id "$id" --seasons "$seasons"
        fi
        run uv run deportivas features compute-nhl --competition-id "$id"
        run uv run deportivas models train-nhl --competition-id "$id"
        ;;
    baseball)
        if [ "$(jq -r '.pybaseball // false' <<<"$sources")" = "true" ]; then
            teams="$(jq -r '.team_abbreviations // [] | join(",")' <<<"$sources")"
            if [ -n "$teams" ]; then
                mlb_season="$(uv run deportivas current-seasons --competition-id "$id" --count 1)"
                run uv run deportivas ingest mlb-schedule \
                    --competition-id "$id" --season "$mlb_season" --teams "$teams"
            else
                echo "  aviso: ${id} sin sources.team_abbreviations en competitions.yaml -- se salta ingesta"
            fi
        fi
        run uv run deportivas features compute-mlb --competition-id "$id"
        run uv run deportivas models train-mlb --competition-id "$id"
        ;;
    *)
        echo "  aviso: deporte desconocido '${sport}' -- se salta ingesta"
        ;;
    esac
done < <(competitions)

echo "###### Fase B: cierre de linea + senales + liquidacion (toda competicion habilitada) ######"
while IFS= read -r id; do
    run_settlement_chain "$id"
done < <(competition_ids)

echo "pipeline diario terminado"
