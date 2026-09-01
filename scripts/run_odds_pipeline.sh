#!/usr/bin/env bash
set -uo pipefail

# Captura de cuotas (Fase 9): pensado para correr con una cadencia mas
# frecuente que run_daily_pipeline.sh (varias veces al dia, ver
# .github/workflows/odds.yml), porque cuanto mas seguido se capture, mejor
# aproxima el fallback de closing_price() (backtest/settlement.py) al
# precio real de cierre, y antes se detecta una senal accionable nueva.
#
# Requiere DEPORTIVAS_THE_ODDS_API_KEY (The Odds API, plan gratuito -- ver
# .env.example). Si no esta configurada, este script no tiene nada que
# hacer y termina en 0 sin fallar el workflow que lo invoca.
#
# ODDS_REGIONS="eu" (Pinnacle vive ahi -- la referencia de este proyecto,
# ver config/thresholds.yaml) en vez del default de tres regiones del CLI:
# el plan gratuito de The Odds API es 500 creditos/mes y cada region pedida
# multiplica el costo. Una corrida completa de esta lista de competiciones
# con 3 mercados (h2h/spreads/totals para la mayoria, 2 para baseball) y 1
# region cuesta ~44 creditos; con las 3 regiones por defecto del CLI serian
# ~132 -- suficiente para solo ~3 corridas al mes. .github/workflows/odds.yml
# fija la cadencia real (dias por semana) contra este mismo presupuesto.

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=scripts/lib_common.sh
source scripts/lib_common.sh

if [ -z "${DEPORTIVAS_THE_ODDS_API_KEY:-}" ]; then
    echo "DEPORTIVAS_THE_ODDS_API_KEY no esta configurada -- nada que capturar"
    exit 0
fi

run() {
    echo "+ $*"
    "$@" || echo "  aviso: fallo -- $*"
}

# raw:nuestro, especifico por deporte -- la misma clave de mercado de la API
# significa selecciones distintas segun el deporte (ver el docstring de
# ingest/sources/theoddsapi.py). Baseball no lleva "spreads": el mercado
# "spread" no existe para baseball en config/markets.yaml.
market_map_for() {
    case "$1" in
    football) echo "h2h:1x2,spreads:asian_handicap,totals:over_under" ;;
    american_football | basketball | ice_hockey) echo "h2h:moneyline,spreads:spread,totals:total" ;;
    baseball) echo "h2h:moneyline,totals:total" ;;
    *) echo "" ;;
    esac
}

load_competitions

echo "###### captura de cuotas (The Odds API) ######"
while IFS= read -r competition; do
    id="$(jq -r '.id' <<<"$competition")"
    sport="$(jq -r '.sport' <<<"$competition")"
    sport_key="$(jq -r '.odds.the_odds_api // empty' <<<"$competition")"
    market_map="$(market_map_for "$sport")"

    if [ -z "$sport_key" ]; then
        # the_odds_api: null en competitions.yaml -- The Odds API confirmado
        # que no cubre esta competicion bajo ninguna clave (ver README,
        # bloque de Colombia). Calendario/resultados siguen intactos, solo
        # se salta la captura de cuotas.
        echo "  aviso: ${id} sin odds.the_odds_api en competitions.yaml -- se salta captura de cuotas"
        continue
    fi
    if [ -z "$market_map" ]; then
        echo "  aviso: sin market_map para el deporte '${sport}' -- se salta ${id}"
        continue
    fi

    season="$(uv run deportivas current-seasons --competition-id "$id" --count 1)"
    run uv run deportivas ingest odds-snapshot \
        --competition-id "$id" --sport-key "$sport_key" --season "$season" \
        --market-map "$market_map" --regions "${ODDS_REGIONS:-eu}"
done < <(competitions)

echo "###### cierre de linea + senales + liquidacion (toda competicion habilitada) ######"
while IFS= read -r id; do
    run_settlement_chain "$id"
done < <(competition_ids)

echo "captura de cuotas terminada"
