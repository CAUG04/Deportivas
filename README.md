# Deportivas

Plataforma propia de análisis y pronóstico deportivo con backtesting riguroso.
No es un producto de "tips": estima probabilidades calibradas, las compara
contra el mercado, y solo señala una apuesta cuando hay ventaja medible.
También dice explícitamente a qué **no** apostar.

Un solo usuario inicial. Corrección estadística por encima de features
vistosas. Coste de infraestructura: **$0**, sin tarjeta de crédito, sin
servidor encendido — ver [Fase 9](#arquitectura-de-despliegue-gratuito).

## Reglas innegociables

Estas reglas son la razón de ser del proyecto y están reforzadas con tests,
no solo con documentación:

1. **Corrección temporal (point-in-time).** Ninguna feature puede usar
   información posterior al inicio del partido que predice. Toda tabla de
   features lleva `as_of_timestamp`. Ver
   `src/deportivas/domain/leakage.py` y `tests/unit/test_leakage.py`.
2. **Cuotas con timestamp de captura.** Ninguna cuota se guarda sin
   `captured_at`. El backtest usa el precio disponible en el momento de la
   decisión, nunca la línea de cierre.
3. **Validación walk-forward.** Nada de `train_test_split` aleatorio.
   Entrena en temporadas 1..N, valida en N+1, avanza. Métricas por ventana.
4. **Calibración obligatoria.** Todo modelo pasa por calibración (isotonic o
   Platt) ajustada solo con datos de entrenamiento. Un modelo sin calibración
   reportada no se despliega.
5. **Línea justa desde Pinnacle.** El edge se mide contra la probabilidad
   implícita sin margen (devig multiplicativo, power o Shin — configurable
   en `config/thresholds.yaml`), nunca contra la cuota nominal.
6. **CLV es la métrica principal.** El CLV medio importa más que el ROI.
7. **Sin claves en el código.** Todo por variables de entorno — ver
   [`.env.example`](.env.example).

## Estado del proyecto

**Fase 0 — Scaffolding.** Estructura del repo, esquema de datos declarado una
única vez (`src/deportivas/contracts/`), configuración validada
(`src/deportivas/config/`), migración inicial de Alembic, capa de
repositorio abstracta (interfaces, sin implementación todavía), y el
guardián de leakage temporal. Sin ingesta de datos real todavía — eso es la
Fase 1.

## Cobertura objetivo

- **Fútbol europeo:** Premier League, La Liga, Serie A, Bundesliga,
  Ligue 1, Eredivisie, Primeira Liga
- **Competiciones UEFA:** Champions League, Europa League, Conference League
- **Fútbol colombiano:** Liga BetPlay Dimayor (Primera A)
- **Deportes americanos:** NFL, NBA, MLB, NHL

Las 15 competiciones están declaradas en
[`config/competitions.yaml`](config/competitions.yaml). Añadir una liga
nueva es añadir un bloque en ese YAML, no escribir código.

## Sobre las fuentes de datos: qué existe y qué no

- **Fútbol europeo (5 grandes ligas):** FBref, Understat, Club Elo, ESPN y
  football-data.co.uk vía `soccerdata`, con identificadores de liga
  verificados contra `soccerdata.LEAGUE_DICT`.
- **Eredivisie, Primeira Liga, UEFA, Liga BetPlay:** los identificadores de
  fuente en `competitions.yaml` están declarados pero **no verificados
  contra las fuentes en vivo** en esta fase (esta sesión de desarrollo no
  tiene acceso de red a FBref, ESPN ni The Odds API). El workflow
  `sources-health.yml` (Fase 10) los valida en cada ejecución y falla
  nombrando exactamente qué fuente y qué campo no coincide.
- **Cuotas históricas de la Liga BetPlay Dimayor: no existen en ninguna
  fuente abierta.** No se inventa una fuente. El backtest de esta liga
  arrancará únicamente con las cuotas que el sistema capture desde el día en
  que el job de captura (`odds.yml`, Fase 9) entre en producción.
- **Cuotas en vivo y de cierre (resto de competiciones):** The Odds API
  (plan gratuito, incluye Pinnacle). Requiere
  `DEPORTIVAS_THE_ODDS_API_KEY` — ver `.env.example`.
- Nunca se usa el endpoint de "predictions" de una API comercial como
  verdad: son cajas negras sin calibración publicada.

## Arquitectura de datos

El esquema se declara **una sola vez**, en
[`src/deportivas/contracts/tables.py`](src/deportivas/contracts/tables.py),
como `TableSpec` agnósticos de motor. Tres adaptadores lo derivan sin
duplicar la definición:

| Adaptador | Para qué |
|---|---|
| `contracts/sqlalchemy_adapter.py` | Metadata SQLAlchemy 2.0 → migraciones Alembic → Postgres |
| `contracts/duckdb_adapter.py` | DDL de DuckDB y esquema Arrow → Parquet particionado |
| `contracts/pandera_adapter.py` | Validación de todo DataFrame antes de escribirse |

`tests/unit/test_schema_parity.py` prueba que los tres adaptadores nunca
diverjan: si una columna se añade en `tables.py` y un adaptador no la
recoge, el test falla ahí, no en producción.

La capa de repositorio (`src/deportivas/storage/protocols.py`) define
interfaces (`Protocol` de Python) para lectura/escritura de cada tabla.
Todo el resto del proyecto (ingesta, features, modelos, backtest, API)
depende solo de esas interfaces. Eso es lo que hace que la migración de
DuckDB/Parquet a Postgres (documentada más abajo) sea cambiar qué
implementación está conectada, no reescribir lógica.

### Backend activo: DuckDB sobre Parquet (producción, coste $0)

`DEPORTIVAS_STORAGE_BACKEND=duckdb` (por defecto). Sin servidor: DuckDB es
solo un motor de consulta sobre ficheros Parquet particionados por
competición y temporada. Los Parquet se publican como *assets* de GitHub
Releases (Fase 9), no como archivos del repositorio.

### Backend alternativo: PostgreSQL (desarrollo local / migración futura)

`DEPORTIVAS_STORAGE_BACKEND=postgres`. `docker-compose.yml` levanta un
Postgres 16 local. Las migraciones viven en `alembic/`, generadas por
`alembic revision --autogenerate` contra la metadata derivada del mismo
esquema (`contracts/sqlalchemy_adapter.py`) — nunca se editan a mano.

**Migración futura documentada, no implementada:** cuando los datos superen
~1 GB o se necesiten consultas en vivo, el almacenamiento pasa a Neon
Postgres (free tier permanente, escala a cero) y la API a Hugging Face
Spaces. Al depender todo de `storage/protocols.py`, ese cambio es
implementar una nueva clase que satisfaga las interfaces existentes.

## Empezar en local

Requisitos: Python 3.11, [`uv`](https://docs.astral.sh/uv/).

```bash
make install       # entorno con las versiones exactas del lockfile + pre-commit
cp .env.example .env   # rellena las claves que tengas (ninguna es obligatoria en Fase 0)
make check          # lint + typecheck + tests
```

Comandos disponibles: `make help`.

### Base de datos local (opcional, solo si trabajas contra Postgres)

```bash
make db-up          # Postgres 16 + Adminer vía docker-compose
make migrate         # aplica alembic/versions/ sobre él
```

## Estructura del repositorio

```
config/                  YAML: competiciones, mercados, umbrales de decisión
src/deportivas/
  config/                 Settings (Pydantic) + carga validada de config/*.yaml
  domain/                 Enums cerrados + guardián de leakage temporal
  contracts/               El esquema, declarado una vez, y sus tres adaptadores
  storage/                 Interfaces de repositorio (implementaciones: Fase 1)
  ingest/ features/ models/ backtest/ signals/ api/ export/   (Fase 1+)
alembic/                  Migraciones sobre la metadata de contracts/
frontend/                 React + Vite + TS + Tailwind (Fase 7)
tests/
  unit/ contracts/ backtest/ fixtures/
.github/workflows/        CI ahora; daily/odds/deploy/sources-health en fases posteriores
```

## Variables de entorno

Ver [`.env.example`](.env.example) — documentado línea por línea. Ninguna
clave real vive en este repositorio; en GitHub Actions viven como Secrets.
