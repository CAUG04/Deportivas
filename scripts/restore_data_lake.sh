#!/usr/bin/env bash
set -uo pipefail

# Descarga y desempaqueta el asset de datos publicado por
# publish_data_lake.sh en el Release fijo "data-lake" -- el mecanismo real
# por el que el backend DuckDB/Parquet (efimero en cada runner de GitHub
# Actions) persiste entre corridas. Ver "Arquitectura de despliegue
# gratuito" en el README (Fase 9).
#
# Tolera que el Release todavia no exista (primera corrida jamas): "gh
# release download" falla, se avisa y el pipeline arranca con data/ vacio --
# el mismo estado que un "make install" fresco, nunca un error fatal.
#
# Requiere el CLI "gh" autenticado (GH_TOKEN en el entorno) y "tar".

RELEASE_TAG="data-lake"
ASSET_NAME="deportivas-data-lake.tar.gz"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p data/raw data/parquet

if gh release download "$RELEASE_TAG" --pattern "$ASSET_NAME" --output "$ASSET_NAME" --clobber 2>/dev/null; then
    tar -xzf "$ASSET_NAME"
    rm -f "$ASSET_NAME"
    echo "data lake restaurado desde el release '${RELEASE_TAG}'"
else
    echo "sin release '${RELEASE_TAG}' todavia -- arrancando con data/ vacio"
fi
