#!/usr/bin/env bash
# Funciones compartidas por run_daily_pipeline.sh y run_odds_pipeline.sh --
# este fichero se hace "source", nunca se ejecuta solo.

COMPETITIONS_JSON_FILE="$(mktemp)"
trap 'rm -f "$COMPETITIONS_JSON_FILE"' EXIT

# Vuelca las competiciones habilitadas a un fichero temporal una sola vez --
# "deportivas list-competitions" arranca un interprete Python; llamarlo una
# vez por script en vez de una vez por competicion es la diferencia entre
# segundos y minutos sobre 15 competiciones (ver config/competitions.yaml).
# Falla ruidosamente si esto no puede cargar: sin esto ninguno de los dos
# scripts que lo usan tiene nada que hacer.
load_competitions() {
    if ! uv run deportivas list-competitions >"$COMPETITIONS_JSON_FILE"; then
        echo "ERROR: 'deportivas list-competitions' fallo -- no se puede continuar" >&2
        exit 1
    fi
}

# Un id de competicion por linea.
competition_ids() {
    jq -r '.[].id' "$COMPETITIONS_JSON_FILE"
}

# Un objeto JSON de competicion por linea (compacto, para leer con jq -c en
# un bucle "while read").
competitions() {
    jq -c '.[]' "$COMPETITIONS_JSON_FILE"
}

# mark-closing + signals generate + backtest settle + backtest report para
# UNA competicion. Corre todos los dias para TODA competicion habilitada, sin
# importar si su calendario/stats se refresco hoy en run_daily_pipeline.sh:
# un partido puede arrancar y terminar entre dos refrescos de una
# competicion "weekly" (ver config/competitions.yaml), y su cierre de linea
# / liquidacion no puede esperar a la proxima ingesta. run_odds_pipeline.sh
# la corre tambien porque una cuota nueva puede convertir una senal
# "descartar" en accionable.
#
# Cada paso se aisla con "||": un fallo en una competicion no debe tumbar a
# las demas ni al resto de la cadena (settle puede tener sentido igual si
# mark-closing fallo, gracias al fallback documentado en settlement.py).
run_settlement_chain() {
    local competition_id="$1"
    echo "-- liquidacion: ${competition_id} --"
    uv run deportivas ingest mark-closing --competition-id "$competition_id" \
        || echo "  aviso: mark-closing fallo para ${competition_id}"
    uv run deportivas signals generate --competition-id "$competition_id" \
        || echo "  aviso: signals generate fallo para ${competition_id}"
    uv run deportivas backtest settle --competition-id "$competition_id" \
        || echo "  aviso: backtest settle fallo para ${competition_id}"
    uv run deportivas backtest report --competition-id "$competition_id" || true
}
