#!/usr/bin/env bash
set -euo pipefail

# Empaqueta data/raw + data/parquet y los publica como el asset del Release
# fijo "data-lake" -- la contraparte de restore_data_lake.sh. data/cache/
# queda fuera a proposito: es cache de scraper, regenerable en cualquier
# momento, nunca la fuente de verdad (ver Settings.cache_dir).
#
# El tag "data-lake" no es una version de software -- es un Release unico y
# fijo cuyo unico asset se reemplaza en cada corrida ("--clobber"). Requiere
# el CLI "gh" autenticado (GH_TOKEN en el entorno) y "tar".

RELEASE_TAG="data-lake"
ASSET_NAME="deportivas-data-lake.tar.gz"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p data/raw data/parquet
tar -czf "$ASSET_NAME" data/raw data/parquet

if gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
    gh release upload "$RELEASE_TAG" "$ASSET_NAME" --clobber
else
    gh release create "$RELEASE_TAG" "$ASSET_NAME" \
        --title "Data lake (Parquet + raw)" \
        --notes "Actualizado automaticamente por scripts/publish_data_lake.sh en cada corrida de daily.yml/odds.yml. No es una version de software: es el estado persistente de data/raw + data/parquet entre corridas de GitHub Actions (ver Fase 9 en el README)."
fi
rm -f "$ASSET_NAME"
echo "data lake publicado en el release '${RELEASE_TAG}'"
