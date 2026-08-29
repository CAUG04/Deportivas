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
repositorio abstracta (interfaces), y el guardián de leakage temporal.

**Fase 1 — Ingesta.** Implementaciones reales de la capa de repositorio
(DuckDB/Parquet y Postgres, ambas con upsert idempotente sobre la clave
natural de cada tabla), la capa cruda append-only (`storage/duckdb_repo/raw_store.py`),
y un adaptador por fuente detrás de la interfaz común `DataSource`
(`src/deportivas/ingest/`): FBref, Understat, ESPN y football-data.co.uk vía
`soccerdata`; NFL vía `nfl_data_py`; NBA/NHL vía `sportsdataverse`; MLB vía
`pybaseball`; cuotas en vivo vía The Odds API. Un CLI (`deportivas ingest ...`)
expone cada adaptador como comando. Ver [limitaciones conocidas](#limitaciones-conocidas-de-la-fase-1)
abajo — documentadas, no escondidas.

**Fase 2 — Features.** Un pipeline de features real y punto-en-tiempo por
cada uno de los 5 deportes (`src/deportivas/features/`), todos con la misma
disciplina: procesan los partidos en orden cronológico, cada fila lleva
`as_of_timestamp` (nunca "ahora", sino el límite real de información
usada), y `domain/leakage.py` bloquea cualquier escritura que lo viole —
`features/writer.py` es el único punto de entrada a la tabla `features` y
lo llama siempre. Un CLI (`deportivas features compute-...`) expone cada
pipeline como comando. Ver [alcance de la Fase 2](#alcance-de-la-fase-2)
abajo para qué se simplificó y por qué.

- **`football_v1`** (`features/football/`): Elo con ventaja de localía,
  ataque/defensa estilo Dixon-Coles (GLM de Poisson, reajustado cada N
  partidos), xG rolling con decaimiento exponencial (ventanas 5/10/20),
  descanso y congestión de calendario, y defensa ajustada por la fuerza del
  rival — cinco módulos combinados en un solo vector por partido
  (`pipeline.py`).
- **`nfl_v1`** (`features/nfl/`): EPA/jugada y tasa de éxito rolling,
  ofensivos y defensivos (agregados desde play-by-play de nflfastR vía
  `nfl_data_py`, tabla nueva `nfl_team_game_stats`), días de descanso, y una
  "DVOA aproximada" — una transformación simple sobre el EPA rolling
  (ataque propio vs. defensa típica del rival), no la metodología real de
  Football Outsiders.
- **`nba_v1`, `nhl_v1`, `mlb_v1`** (`features/nba/`, `features/nhl/`,
  `features/mlb/`): descanso, back-to-back (configurable por deporte) y
  margen de anotación rolling, calculados directamente sobre el marcador de
  `fixtures` — la única señal por partido ingerida hasta ahora para estos
  tres deportes. Comparten el mismo módulo (`features/rest_and_margin.py`).

**Fase 3 — Modelos (en curso).** Núcleo compartido (`src/deportivas/models/`):
ventanas walk-forward por temporada (`walkforward.py`, regla #3), calibración
isotónica/Platt ajustada solo con datos de entrenamiento (`calibration.py`,
regla #4), y Brier score/log loss/curva de fiabilidad (`metrics.py`). Primer
modelo real: **fútbol**, un Poisson bivariante (`models/football/`) que
reutiliza el mismo ajuste de GLM que la feature `strength.py`
(`features/football/dixon_coles_glm.py`, ahora compartido) pero como modelo
predictivo — con `home_advantage` e intercepto — en vez de una calificación.
Cada partido produce una matriz de goles de la que se leen **1x2**,
**over/under** y **btts** (`config/markets.yaml`'s `derived_from:
score_matrix`), con una fila en `model_registry` por ventana (métricas por
temporada de validación) y una fila en `predictions` por partido, mercado,
selección y línea. Ver [alcance de la Fase 3](#alcance-de-la-fase-3) para
qué queda fuera todavía (modelos de NFL/NBA/NHL/MLB, hándicap asiático).

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

- **Fútbol europeo (5 grandes ligas):** FBref, Understat, ESPN y
  football-data.co.uk vía `soccerdata`, con identificadores de liga
  verificados contra `soccerdata.LEAGUE_DICT` y los mapeos de columnas
  verificados leyendo el código fuente instalado de `soccerdata` (no
  adivinados) — ver los docstrings de módulo en `src/deportivas/ingest/sources/`.
- **Eredivisie, Primeira Liga, UEFA, Liga BetPlay:** los identificadores de
  fuente en `competitions.yaml` están declarados pero **no verificados
  contra las fuentes en vivo** todavía (esta sesión de desarrollo no tiene
  acceso de red a FBref, ESPN ni The Odds API). El workflow
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

## Limitaciones conocidas de la Fase 1

Documentadas explícitamente porque el proyecto prefiere una limitación
visible a una silenciosa:

- **pybaseball (MLB): columnas no verificadas contra un fetch real.**
  `schedule_and_record` raspa una tabla HTML de baseball-reference.com; sus
  nombres de columna viven en el marcado, no en el código fuente de
  `pybaseball`, así que no se pudieron confirmar sin red. El mapeo usa el
  formato documentado y estable desde hace años (`Date`, `Home_Away`, `Opp`,
  `R`, `RA`), de forma defensiva (nunca lanza sobre una fila con forma
  inesperada), pero debe verificarse contra un fetch real antes de
  confiarle producción — ver el docstring de
  `src/deportivas/ingest/sources/pybaseball_source.py`. Exactamente el tipo
  de riesgo que `sources-health.yml` (Fase 10) está pensado para atrapar.
- **ESPN no publica marcador final en `read_schedule()`.** Toda fixture de
  este adaptador queda `status="scheduled"`, incluso partidos ya jugados.
  Es la única fuente para Liga BetPlay, así que sus resultados históricos
  dependen de un futuro uso de `read_matchsheet()` (una llamada HTTP por
  partido) — no implementado en esta fase.
- **football-data.co.uk: solo se mapea el mercado 1X2.** Los nombres de
  columna de hándicap asiático y over/under han cambiado más de una vez
  entre temporadas; los del 1X2 llevan 20+ años estables. Esos mercados
  llegan por The Odds API en su lugar.
- **Club Elo: no ingerido en esta fase.** No hay tabla en el esquema para
  ratings externos por equipo — Elo es algo que el propio sistema calcula
  en la Fase 2 (features), no un dato que haga falta ingerir ya. Añadirlo
  como fuente de validación es una decisión de la Fase 2, no un olvido.
- **`captured_at` de football-data.co.uk es una aproximación, no una
  observación real.** Esa fuente no publica timestamp de captura: la cuota
  "pre-cierre" se marca en kickoff menos un día y la de cierre en el propio
  kickoff. Documentado en el código (`is_closing` sigue siendo fiable; el
  timestamp exacto no). The Odds API sí captura timestamps reales.

## Alcance de la Fase 2

Documentado explícitamente porque el proyecto prefiere una simplificación
visible a una silenciosa — el usuario pidió explícitamente construir los 5
deportes ya en esta fase, y así se hizo, pero con distinto nivel de
profundidad según qué datos ya están ingeridos:

- **DVOA aproximado (NFL) no es DVOA real.** La metodología real de Football
  Outsiders es una regresión iterativa de fuerza de rival sobre toda la liga,
  con valores de jugada situacionalmente neutrales y ponderación semana a
  semana. `dvoa_approx.py` es una transformación mucho más simple y explícita
  sobre el EPA rolling ya calculado (ataque propio vs. defensa típica de
  *este* rival concreto) — un proxy razonable, no una reimplementación.
- **NBA/NHL/MLB: sin rating ajustado por posesión ni por pitcheo.** Ninguno
  de los tres tiene boxscore ni play-by-play ingerido todavía (Fase 1 solo
  trajo calendario + marcador vía `sportsdataverse`/`pybaseball`), así que
  un net rating ajustado por pace, o el abridor probable de MLB, no son
  viables con lo que hay en la base hoy. `rest_and_margin.py` calcula lo que
  sí es real con el marcador final: descanso, back-to-back y margen de
  anotación rolling.
- **Ninguna feature de estas tres se recalcula sobre la marcha.** Igual que
  fútbol y NFL, cada vector se calcula una vez, en orden walk-forward, y se
  persiste con su `as_of_timestamp` — la simplificación está en qué señales
  entran al vector, nunca en la disciplina de cuándo se calculan.

## Alcance de la Fase 3

- **Solo fútbol por ahora.** El modelo Poisson (`models/football/`) es lo que
  se propuso explícitamente y se aprobó para este primer corte de la Fase 3.
  Modelos para NFL/NBA/NHL/MLB (moneyline/spread/total — `derived_from:
  classifier` / `margin_regression` en `config/markets.yaml`, un enfoque
  distinto al de matriz de goles) quedan para una siguiente entrega dentro
  de la misma fase, no descartados.
- **Poisson independiente, no Dixon-Coles completo.** Sin el ajuste tau de
  baja puntuación (0-0/1-0/0-1/1-1) del paper original — la misma
  simplificación que `dixon_coles_glm.py` ya nombra en su propio docstring.
  Es una razón conocida por la que los empates suelen quedar
  sub-estimados; la curva de fiabilidad por ventana en `model_registry` es
  justamente lo que permite verlo, no algo escondido.
- **`asian_handicap` queda fuera**, aunque `config/markets.yaml` también lo
  marca `derived_from: score_matrix`. Su liquidación con push y medio-gane
  pertenece a la fase de señales/backtest (donde vive `RESULTS.outcome`),
  no al entrenamiento del modelo.
- **Calibración in-sample dentro de la ventana de entrenamiento**, no
  validación cruzada. Ajustar el calibrador con predicciones
  cross-validated dentro de la propia ventana de entrenamiento sería más
  riguroso; queda como mejora futura documentada, no como una limitación
  escondida — la validación *fuera* de muestra (lo que de verdad importa
  para no hacer trampa) sigue siendo estrictamente walk-forward por
  temporada.

## CLI de ingesta

```bash
uv run deportivas ingest --help          # lista cada adaptador como comando
uv run deportivas seed-competitions      # carga config/competitions.yaml en la tabla competitions

# Ejemplo: backfill de 3 temporadas de Premier League vía FBref
uv run deportivas ingest fbref-schedule \
  --competition-id eng-premier-league --fbref-league "ENG-Premier League" \
  --seasons 2223,2324,2425
```

Backfill (histórico, `--seasons` explícito) e ingesta incremental son el
mismo comando con una lista de temporadas distinta — la Fase 8 es la que
programa las corridas diarias de "temporada actual"; este CLI no adivina
ventanas de fechas por su cuenta.

## CLI de features

```bash
uv run deportivas features --help              # lista cada pipeline como comando

# Requiere que fixtures (y, para football/NFL, team_match_stats /
# nfl_team_game_stats) ya esten ingeridos para esa competicion.
uv run deportivas features compute-football --competition-id eng-premier-league
uv run deportivas features compute-nfl --competition-id usa-nfl
uv run deportivas features compute-nba --competition-id usa-nba
uv run deportivas features compute-nhl --competition-id usa-nhl
uv run deportivas features compute-mlb --competition-id usa-mlb
```

Cada comando recalcula el vector completo de esa competición y lo escribe
bajo su propio `feature_set` (`football_v1`, `nfl_v1`, ...) — re-ejecutarlo
tras ingerir partidos nuevos es idempotente (`write_features` hace upsert
sobre `(fixture_id, feature_set)`).

## CLI de modelos

```bash
uv run deportivas models --help                # lista cada modelo como comando

# Requiere que fixtures ya este ingerido, con al menos dos temporadas
# terminadas (una para entrenar, una para validar).
uv run deportivas models train-football --competition-id eng-premier-league

# Metodo de calibracion explicito en vez del de config/thresholds.yaml:
uv run deportivas models train-football \
  --competition-id eng-premier-league --calibration-method platt
```

A diferencia de `features compute-...`, esto **no** es idempotente por
diseño: `model_registry` es `append_only` (una fila por corrida de
entrenamiento, nunca sobreescrita) para conservar el historial completo de
cada ventana entrenada, incluso si se re-ejecuta el mismo comando dos veces.

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
competición y temporada (`storage/duckdb_repo/`). El *upsert* sobre la clave
natural de cada tabla (lo que hace idempotente reejecutar una fuente) se
implementa a mano — Parquet no tiene `ON CONFLICT`. Los Parquet se
publicarán como *assets* de GitHub Releases (Fase 9), no como archivos del
repositorio.

**Capa cruda append-only** (`storage/duckdb_repo/raw_store.py`): cada
respuesta de una fuente se guarda tal cual, con hash de contenido y
timestamp real de captura, antes de parsear nada. Nunca se sobrescribe.
Features y modelos deben poder reconstruirse desde aquí sin volver a
raspar — la decisión más importante de la Fase 10, construida desde ya.

### Backend alternativo: PostgreSQL (desarrollo local / migración futura)

`DEPORTIVAS_STORAGE_BACKEND=postgres`. `docker-compose.yml` levanta un
Postgres 16 local. Las migraciones viven en `alembic/`, generadas por
`alembic revision --autogenerate` contra la metadata derivada del mismo
esquema (`contracts/sqlalchemy_adapter.py`) — nunca se editan a mano. El
*upsert* usa `INSERT ... ON CONFLICT (clave_natural) DO UPDATE`; las tablas
append-only (`odds_snapshots`, `raw_documents`, `model_registry`) no llevan
esa restricción — exigirla rechazaría capturas legítimas repetidas.
`tests/unit/storage/test_sql_repository.py` (marcador `postgres`, activo en
CI) prueba este backend contra un Postgres real, no solo contra DuckDB.

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
  domain/                 Enums cerrados, ids deterministas, guardián de leakage temporal
  contracts/              El esquema, declarado una vez, y sus tres adaptadores
  storage/
    protocols.py           Interfaces de repositorio
    duckdb_repo/            Implementación DuckDB/Parquet + capa cruda append-only
    sql_repo/                Implementación Postgres
    factory.py               Elige backend según DEPORTIVAS_STORAGE_BACKEND
    unit_of_work.py           Escrituras agrupadas: todo o nada
    validation.py             Pandera: fila mala se rechaza y se registra, nunca se propaga
  ingest/
    base.py                  Interfaz DataSource: rate limiting + archivado en capa cruda
    ratelimit.py cache.py aliases.py
    sources/                 Un adaptador por fuente (fbref, understat, espn,
                              footballdata, nfl, sportsdataverse_source,
                              pybaseball_source, theoddsapi)
  features/
    asof.py                  Carga fixtures/stats punto-en-tiempo, dedup por fuente
    writer.py                Unico punto de escritura a features, bloqueado por leakage.py
    merge.py                 Fusiona los vectores de varios modulos en uno por partido
    rest_and_margin.py       Descanso/back-to-back/margen rolling (NBA, NHL, MLB)
    football/                Elo, ataque/defensa (GLM), xG rolling, descanso, opponent_adjusted
    nfl/                     EPA/jugada rolling, descanso, DVOA aproximado
    nba/ nhl/ mlb/           pipeline.py sobre rest_and_margin.py, config por deporte
  models/
    walkforward.py           Ventanas walk-forward por temporada (regla #3)
    calibration.py           Calibracion isotonica/Platt, solo con datos de entrenamiento
    metrics.py                Brier score, log loss, curva de fiabilidad
    football/                 Poisson bivariante: matriz de goles -> 1x2/over_under/btts
  cli.py                   Un comando por adaptador de ingesta, pipeline de features y modelo
  backtest/ signals/ api/ export/   (Fase 4+)
alembic/                  Migraciones sobre la metadata de contracts/
frontend/                 React + Vite + TS + Tailwind (Fase 7)
tests/
  unit/ contracts/ backtest/ fixtures/
.github/workflows/        CI (incluye servicio Postgres); daily/odds/deploy/sources-health en fases posteriores
```

## Variables de entorno

Ver [`.env.example`](.env.example) — documentado línea por línea. Ninguna
clave real vive en este repositorio; en GitHub Actions viven como Secrets.
