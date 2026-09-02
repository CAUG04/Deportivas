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

**Fase 3 — Modelos.** Núcleo compartido (`src/deportivas/models/`): ventanas
walk-forward por temporada (`walkforward.py`, regla #3), calibración
isotónica/Platt ajustada solo con datos de entrenamiento (`calibration.py`,
regla #4), y Brier score/log loss/curva de fiabilidad (`metrics.py`). Un
modelo real por cada uno de los 5 deportes:

- **Fútbol**: un Poisson bivariante (`models/football/`) que reutiliza el
  mismo ajuste de GLM que la feature `strength.py`
  (`features/football/dixon_coles_glm.py`, ahora compartido) pero como
  modelo predictivo — con `home_advantage` e intercepto — en vez de una
  calificación. Cada partido produce una matriz de goles de la que se leen
  **1x2**, **over/under** y **btts** (`config/markets.yaml`'s
  `derived_from: score_matrix`).
- **NFL, NBA, NHL, MLB**: un clasificador logístico de **moneyline**
  (`models/moneyline.py`, `models/moneyline_training.py`, compartidos;
  `models/nfl/`, `models/nba/`, `models/nhl/`, `models/mlb/` solo aportan su
  `feature_set`) entrenado sobre los vectores ya calculados en la Fase 2
  (`nfl_v1`, `nba_v1`, `nhl_v1`, `mlb_v1`) — a diferencia de fútbol, este no
  ajusta coeficientes propios desde el marcador: usa las señales que la
  feature de cada deporte ya produjo.

Cada modelo escribe una fila en `model_registry` por ventana (métricas por
temporada de validación) y una fila en `predictions` por partido, mercado,
selección y línea. Ver [alcance de la Fase 3](#alcance-de-la-fase-3) para
qué queda fuera todavía (hándicap asiático, spread/total de los deportes
americanos).

**Fase 4 — Señales.** `src/deportivas/signals/` convierte una predicción en
una decisión: `devig.py` quita el margen de una cuota (multiplicativo, power
o Shin — los tres coinciden exactamente cuando la cuota no trae margen),
`tiers.py` clasifica la confianza (alta/media/baja/**descartar**, evaluado
de arriba a abajo contra `config/thresholds.yaml`'s `tiers`) y `staking.py`
calcula el stake de Kelly fraccionado (regla #6: nunca Kelly completo, nunca
progresiones). `generate.py` los une: por cada fila de `predictions`, busca
en `odds_snapshots` la cuota de Pinnacle (o, si falta, la del primer
bookmaker de la lista de reserva) capturada entre el `as_of_timestamp` de la
predicción y el kickoff del partido — nunca antes (no se puede actuar sobre
una cuota anterior a la propia predicción) ni después (nada de cuotas en
vivo) —, calcula el edge (`prob_model - prob_fair`) y escribe una fila en
`signals`, incluidas las que se **descartan**: saber a qué no apostar se
persiste igual que una apuesta accionable, no se descarta en silencio. Ver
[alcance de la Fase 4](#alcance-de-la-fase-4) para las decisiones de diseño
detrás de esa ventana de precio.

**Fase 5 — Backtest.** `src/deportivas/backtest/` liquida las señales y mide
si el sistema funciona, con **CLV como métrica principal** (no el pnl): un
acierto de suerte o una mala racha mueven el pnl en cualquier dirección,
pero el CLV pregunta algo que no depende de la varianza — ¿se movió el
mercado para confirmar el precio que conseguimos, antes de que el partido
siquiera empezara? `settlement.py` liquida cada señal cuyo partido ya
terminó (incluidas las `descartar`/`baja`: su CLV es la verificación
honesta de si el sistema de tiers está descartando lo que debía) contra la
cuota de cierre del mismo bookmaker de entrada. `bootstrap.py` calcula el
intervalo de confianza del CLV por remuestreo — nunca asumiendo una campana
que una muestra chica y sesgada por unas pocas cuotas altas no tiene.
`baselines.py` responde la pregunta que un edge real tiene que superar para
valer algo: ¿qué hubiera devuelto apostar siempre al favorito, o al azar,
sobre exactamente los mismos partidos, en el mismo instante de cuota que ya
usó la señal real? `report.py` junta todo: CLV medio, su intervalo de
confianza, pnl y ROI — global, por tier, por mercado y contra cada
baseline. Ver [alcance de la Fase 5](#alcance-de-la-fase-5) para las
decisiones de diseño (el fallback del precio de cierre cuando `mark_closing`
todavía no corrió para un fixture, el stake plano de las baselines, y qué
significa aquí `min_matches_per_window`).

**Fase 6 — API y export.** `src/deportivas/api/views.py` es la única fuente
de verdad de "qué ve el consumidor": funciones puras (sin importar
`fastapi`) que arman modelos Pydantic de solo lectura directamente desde
los repositorios — competiciones, señales (enriquecidas con nombre de
equipos y kickoff, para que nadie tenga que volver a unir contra `fixtures`)
y el reporte de backtest. Dos superficies la consumen, nunca la
reimplementan:

- **`api/app.py`** — una API FastAPI de solo lectura (`GET /health`,
  `/competitions`, `/competitions/{id}/signals`, `/competitions/{id}/backtest`)
  para desarrollo local. Sin autenticación: es una herramienta de un solo
  operador, no un servicio público.
- **`export/json_export.py`** — lo que `frontend/` (Fase 7) realmente
  consume en producción: JSON pre-calculado bajo `Settings.export_dir`
  (`frontend/public/data/`), sin necesitar un servidor corriendo — coherente
  con el despliegue a coste $0. `deportivas export run` lo genera.

Ver [alcance de la Fase 6](#alcance-de-la-fase-6) para por qué existen las
dos vías y en qué difieren (`only_actionable` por defecto en la API viva
contra todos los tiers en el export).

**Fase 7 — Frontend.** `frontend/` es un sitio estático (Vite + React 19 +
TypeScript + Tailwind v4) que lee el JSON de `export/json_export.py` bajo
`public/data/` — sin llamar a ningún servidor, ni siquiera en desarrollo.
Un selector de competición, una tabla de señales (filtro por tier, incluido
`descartar`) y el resumen de backtest (CLV medio con su intervalo de
confianza cuando lo hay, ROI, desglose por tier/mercado, comparación contra
las baselines) — deliberadamente minimalista: ver
[alcance de la Fase 7](#alcance-de-la-fase-7) para qué queda fuera y por qué.

**Fase 8 — Automatización diaria.** `scripts/run_daily_pipeline.sh` es el
único lugar que decide, por competición, qué comando de ingesta correr y en
qué orden — el CLI sigue siendo primitivas puras, la orquestación vive en
el script, no en código Python nuevo (ver el docstring de `cli.py`).
Refresca calendario/stats + entrena modelos solo para las competiciones
cuya cadencia toca hoy (`daily` siempre, `weekly` una vez por semana — la
palanca contra el límite de tiempo del runner), y corre cierre de línea +
señales + liquidación **todos los días, para toda competición habilitada**,
sin importar si su calendario se refrescó hoy: un partido arranca y termina
entre dos refrescos de una competición `weekly` igual. Dos piezas nuevas lo
hacen posible:

- **`storage/protocols.py`'s `mark_closing`** — la única excepción,
  deliberada y acotada, a "las tablas append-only nunca se actualizan"
  (regla de la Fase 0/1). Marca `is_closing=True` sobre el último snapshot
  antes del kickoff de cada fixture ya arrancado. Es una optimización, no
  una dependencia de corrección: `closing_price()` (`backtest/settlement.py`)
  sigue cayendo al mismo fallback documentado en la
  [Fase 5](#alcance-de-la-fase-5) si esto nunca corre — `ingest/closing.py`
  es el job que lo ejecuta, idempotente (salta un fixture ya marcado).
- **`domain/seasons.py`** — la temporada actual de una competición, en el
  formato que cada fuente espera (código de dos años para fútbol, año
  simple para el resto), para que `deportivas current-seasons` alimente
  `--seasons` sin que un humano lo calcule a mano cada día. `deportivas
  list-competitions` completa el trío: vuelca `competitions.yaml` como
  JSON para que el script lo recorra con `jq`.

**Fase 9 — Arquitectura de despliegue gratuito.** Ver
[la sección dedicada](#arquitectura-de-despliegue-gratuito) más abajo para
el detalle completo (el Release de GitHub como disco persistente, el
presupuesto de créditos de The Odds API, el encadenado hacia GitHub Pages).
En una frase: `daily.yml`/`odds.yml` producen datos y los publican como el
asset de un Release fijo; `deploy.yml` los descarga, exporta el JSON
estático y construye/publica `frontend/`.

**Fase 10 — Salud de las fuentes.** `ingest/sources_health.py` reutiliza
los mismos adaptadores que la ingesta real (mismo rate limiting, mismo
archivado en la capa cruda) para intentar, contra la fuente real, exactamente
los identificadores que la [Fase 1](#limitaciones-conocidas-de-la-fase-1)
declaró "no verificados todavía" — y descubrió, de paso, que estaban rotos
de verdad: nada en el código generaba el `league_dict.json` personalizado
que `soccerdata` necesita para reconocer Eredivisie/Primeira
Liga/UEFA/Liga BetPlay Dimayor, así que esas seis competiciones fallaban
antes de intentar ninguna llamada de red.
`ingest/soccerdata_config.py` lo genera ahora desde `competitions.yaml`, y
`sources-health.yml` corre esa misma validación cada día, antes que
`daily.yml`/`odds.yml`, nombrando la competición y el campo exactos si algo
deja de cuadrar — nunca solo "algo salió mal".

## Cobertura objetivo

- **Fútbol europeo:** Premier League, La Liga, Serie A, Bundesliga,
  Ligue 1, Eredivisie, Primeira Liga
- **Competiciones UEFA:** Champions League, Europa League, Conference League
  — **deshabilitadas** (`enabled: false`), ver
  [Limitaciones conocidas](#limitaciones-conocidas-de-la-fase-1)
- **Fútbol colombiano:** Liga BetPlay Dimayor (Primera A)
- **Deportes americanos:** NFL, NBA, MLB, NHL

Las 15 competiciones están declaradas en
[`config/competitions.yaml`](config/competitions.yaml); 12 habilitadas hoy
(las 3 UEFA no lo están — ver arriba). Añadir una liga nueva es añadir un
bloque en ese YAML, no escribir código; volver a habilitar una ya declarada
es cambiar un flag, no reescribir nada.

## Sobre las fuentes de datos: qué existe y qué no

- **Fútbol europeo (5 grandes ligas):** FBref, Understat, ESPN y
  football-data.co.uk vía `soccerdata`, con identificadores de liga
  verificados contra `soccerdata.LEAGUE_DICT` y los mapeos de columnas
  verificados leyendo el código fuente instalado de `soccerdata` (no
  adivinados) — ver los docstrings de módulo en `src/deportivas/ingest/sources/`.
- **Eredivisie, Primeira Liga, Liga BetPlay:** los identificadores de fuente
  en `competitions.yaml` quedaron verificados contra la fuente en vivo por
  `sources-health.yml` (Fase 10) — siguen habilitadas y corriendo en el
  pipeline diario.
- **Las tres competiciones UEFA: deshabilitadas** (`enabled: false`).
  Confirmado en producción que ninguna fuente de calendario funciona para
  ellas hoy — ver el bullet de FBref/UEFA en
  [Limitaciones conocidas](#limitaciones-conocidas-de-la-fase-1). Sin
  calendario no hay nada que ingerir, entrenar ni liquidar; se quedan
  declaradas para no perder el mapeo de identificadores ya investigado.
- **Cuotas históricas de la Liga BetPlay Dimayor: no existen en ninguna
  fuente abierta**, y además `sources-health.yml` confirmó en producción
  que The Odds API tampoco cubre esta liga en absoluto — `soccer_colombia_
  primera_a` (la clave declarada originalmente) no aparece en `/v4/sports`.
  `odds.the_odds_api: null` en `competitions.yaml` desactiva la captura de
  cuotas para esta competición; calendario/resultados vía ESPN siguen
  intactos.
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
  **Actualización:** la primera corrida real lo atrapó, aunque no donde se
  esperaba — los nombres de columna resultaron correctos; lo que falla es el
  último paso de `schedule_and_record`. Ver abajo.
- **FBref bloquea con CAPTCHA a los runners de GitHub Actions.** Confirmado
  en producción (`sources-health.yml`), tres corridas seguidas: las 11
  competiciones de fútbol fallan de forma consistente al intentar FBref
  desde el runner — `ConnectionError` tras agotar los reintentos de
  `soccerdata`. Se probó evadirlo con `headless=False` + Xvfb (ver
  [Fase 9](#fbref-bloquea-con-captcha-a-los-runners-de-github-actions)) y se
  confirmó, también en producción, que no lo resuelve. Con el bloqueo ya
  confirmado y repetible, `daily.yml`/`sources-health.yml` desactivan FBref
  del todo (`DEPORTIVAS_FBREF_ENABLED=false`) en vez de seguir gastando
  minutos de CI en un resultado ya conocido — football-data.co.uk y ESPN
  (ligas domésticas) cubren calendario/resultados igual. El código
  (`Settings.fbref_enabled`/`fbref_headless`, `FBrefSource(headless=...)`)
  se queda en el repositorio para uso manual: `deportivas fbref-schedule`/
  `fbref-stats`, corridos a mano desde una IP normal (no de datacenter),
  suelen pasar sin problema.
- **`soccerdata.ESPN.read_schedule()` no soporta calendarios por etapas
  (UEFA).** Bug de la librería, no de este proyecto: para las tres
  competiciones UEFA (fase de grupos + eliminatorias), la API de ESPN
  devuelve `calendar` como una lista de objetos por etapa en vez de
  fechas planas, y `soccerdata` intenta `datetime.strptime()` sobre esos
  objetos directamente — `TypeError: strptime() argument 1 must be str,
  not dict`, confirmado también en producción. Con FBref bloqueado por
  CAPTCHA arriba, esto deja a las tres competiciones UEFA sin ninguna
  fuente de calendario funcional — por eso están `enabled: false` en
  `competitions.yaml` en vez de fallando en rojo cada corrida por un
  problema que ninguna de las dos fuentes va a resolver por sí sola.
  Documentado, no oculto ni parcheado por dentro de una librería de
  terceros; volver a habilitarlas es un flag, en cuanto una de las dos
  fuentes quede resuelta.
- **`pybaseball.schedule_and_record` revienta en cualquier temporada en
  curso — resuelto.** Bug de la librería: su `get_table` rellena con el
  centinela `"Unknown"` tres columnas cuando la celda viene vacía (un
  partido que aún no se juega no tiene marcador, entradas ni puesto), pero
  solo convierte una de las tres —`Attendance`— de vuelta a `NaN`. Su
  `make_numeric` llega después y hace `astype(float)` sobre
  `["R", "RA", "Inn", "Rank", "Attendance"]`, y falla con el `"Unknown"` que
  quedó en `Rank`: `ValueError: could not convert string to float:
  'Unknown'`. Convertir texto a número ahí es correcto y necesario —
  baseball-reference publica HTML y todo llega como texto; lo que falta es
  tolerar el centinela, algo que `pd.to_numeric(..., errors="coerce")` daría
  gratis. Este adaptador no necesita ese paso en absoluto (lee `R`/`RA` con
  `to_optional_int`, que ya devuelve `None` ante cualquier celda no
  numérica), así que llama a los pasos que sí sirven —`get_soup` y
  `get_table`— y omite `make_numeric`. Ver `_fetch_team_table` en
  `src/deportivas/ingest/sources/pybaseball_source.py`.
- **`seasons_back` era configuración muerta, y por eso ningún modelo de
  fútbol podía entrenar — resuelto.** `competitions.yaml` declaraba
  `seasons_back: 5` desde la Fase 1 y **nada en el código lo leía nunca**:
  `run_daily_pipeline.sh` llamaba a `current-seasons` con su default de 2.
  Con 2 temporadas el walk-forward produce una sola ventana, que entrena con
  una única temporada — y una liga europea son 306-380 partidos (18-20
  equipos), mientras `thresholds.yaml` pide `min_training_samples: 500` para
  calibrar. Ninguna temporada de liga europea alcanza 500, así que el fútbol
  no es que "todavía" no tuviera datos: era **estructuralmente imposible**
  que entrenara. NBA y NHL no lo notaron porque una temporada suya son
  ~1.300 partidos, muy por encima del umbral. Confirmado reproduciendo el
  entrenamiento contra el data lake real: con `min_training_samples=500`,
  cero ventanas; bajándolo a 300, una ventana y 220 predicciones. El arreglo
  no es bajar el umbral —500 es un piso razonable para calibrar— sino tener
  la historia que el propio YAML ya declaraba: `run_daily_pipeline.sh` acepta
  ahora `SEASONS_COUNT` (2 por defecto, el incremental de cada día) y
  `daily.yml` lo expone como input de `workflow_dispatch` para lanzar el
  backfill a 5 temporadas. Con 5, las ventanas entrenan con 380, 760, 1.140 y
  1.520 partidos: de la segunda en adelante, holgadamente por encima de 500.
- **Abreviaturas de equipo de MLB: las de baseball-reference, no las de
  prensa — resuelto.** `competitions.yaml` traía las de uso común (`CWS`,
  `KC`, `SD`, `SF`, `TB`, `WSH`) y baseball-reference usa otras (`CHW`,
  `KCR`, `SDP`, `SFG`, `TBR`, `WSN`). Y como el adaptador abortaba el bucle
  entero al primer fallo, una sola equivocada dejaba a MLB sin un solo
  partido — habiendo cargado bien los cinco equipos anteriores. Dos
  arreglos: las 30 abreviaturas verificadas contra
  `pybaseball.utils.first_season_map` (con un test que ancla el YAML a esa
  misma tabla, para que un typo futuro falle en CI y no en producción), y
  `fetch_schedule` ahora aísla cada equipo — que falle uno no toca a los
  otros 29, pero que fallen *todos* lanza error en vez de devolver un
  DataFrame vacío que se leería como "no hay partidos todavía".
- **`nfl_data_py.import_pbp_data` falla al pedir una temporada sin publicar
  — resuelto.** Bug de la librería, y doble: su rama de "no hay datos para
  este año" está escrita `except Error as e:` y `Error` no existe en el
  paquete, así que en cuanto falta el parquet de un año el propio manejador
  de error lanza `NameError`; y aunque eso se arreglara aguas arriba, pedir
  una lista de temporadas donde *ninguna* resuelve deja su variable `plays`
  sin asignar y lanza `UnboundLocalError`. La primera corrida real lo
  encontró con 2026, días antes de que nflverse lo publicara. Este
  adaptador pide **una temporada por llamada** y salta las que fallan, para
  que una sin publicar no se lleve por delante a la que sí tiene datos.
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

- **NFL/NBA/NHL/MLB: solo moneyline, no spread ni total.** Ambos son
  `derived_from: margin_regression` en `config/markets.yaml`, y ese propio
  archivo señala por qué no traen `default_lines`: la línea real la pone un
  libro de apuestas en vivo, no hay una rejilla fija que inventar aquí.
  Escribir una fila en `predictions` con una línea inventada sería un dato
  fabricado, no una simplificación honesta — así que spread/total esperan a
  la fase que cruce el modelo con una cuota real capturada.
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
- **El clasificador de moneyline es genérico, no ajustado a mano por
  deporte.** `models/moneyline.py` es una regresión logística que toma
  cualquier vector de features tal cual viene de la Fase 2 (imputando lo
  que falte con la media del propio entrenamiento) — no elige ni pondera
  columnas por deporte. Es honesto sobre sus límites: la calidad de la
  predicción depende enteramente de qué tan buenas sean las features de
  cada deporte, y NBA/NHL/MLB hoy solo tienen descanso/back-to-back/margen
  rolling (ver el alcance de la Fase 2), no una feature tan rica como el
  EPA de NFL o el Elo de fútbol.
- **Empates se excluyen del entrenamiento de moneyline**, no se fuerzan a
  "no-gana": el mercado `moneyline` solo tiene selecciones `home`/`away`,
  así que un empate (posible mayormente en NFL) no es una etiqueta valida
  para ese clasificador.

## Alcance de la Fase 4

- **El precio de entrada usa `as_of_timestamp`, no `predicted_at`.**
  `predicted_at` es el instante de reloj en que corrió el job de
  entrenamiento — para una ventana walk-forward histórica eso es siempre
  "ahora", sin importar qué temporada se está validando, así que acotar por
  ahí dejaría vacía la ventana de cuotas de cualquier predicción histórica y
  nunca se podría construir una señal para el backtest. `as_of_timestamp` es
  el corte de información real que el modelo usó (el propio significado que
  ya tiene esa columna en `predictions`), y por construcción del walk-forward
  siempre es anterior al kickoff de la temporada de validación — así se
  pueden generar señales tanto sobre partidos históricos (lo que necesita el
  backtest de CLV) como sobre los próximos.
- **Un movimiento de línea "a favor" solo se evalúa entre la cuota de
  entrada y la última cuota disponible antes del kickoff**, nunca contra una
  cuota post-kickoff (en vivo). Con menos de `line_move.min_snapshots`
  momentos capturados, la condición simplemente no se cumple — no se
  extrapola con un solo dato.
- **Una sola selección capturada en un instante no se devigea.** Sin al
  menos dos precios contemporáneos no hay margen que quitar, y normalizar
  un único precio devolvería una probabilidad "justa" de 1.0 — un dato
  fabricado, no una simplificación honesta. Esos instantes se descartan en
  `_market_snapshots` en vez de producir un número engañoso.
- **`sample_matches` sale de `model_registry.metrics.n_train_matches`**,
  buscado por `(model_name, model_version)` — esa tabla no tiene columna
  `competition_id` para filtrar, así que se lee entera una vez por corrida.
  Si dos filas comparten esa clave (posible porque `model_registry` es
  `append_only`, ver la Fase 3), gana la última en el orden de lectura del
  backend activo; no es un problema introducido aquí, y no se resuelve aquí.
- ~~No hay backtest todavía.~~ Resuelto en la Fase 5 (`backtest/settlement.py`
  liquida `signals` contra `results.clv`) — ver más abajo.

## Alcance de la Fase 5

- **El precio de cierre cae al último snapshot pre-kickoff cuando
  `mark_closing` todavía no corrió para ese fixture.** `results.clv` se mide
  contra una fila marcada `is_closing=True`; ese flag lo pone
  `ingest/closing.py` (Fase 8), corrido a diario por `daily.yml`/`odds.yml`
  después del kickoff. Mientras tanto — o si esa corrida todavía no llegó a
  este fixture — `closing_price` cae al último precio capturado antes del
  kickoff para el mismo bookmaker de entrada: la misma aproximación, nunca
  un dato inventado, documentada como tal en el docstring de
  `settlement.py`. El flag, cuando existe, gana automáticamente sin tocar
  una línea de código aquí — ver [Fase 8](#estado-del-proyecto) para el
  porqué `mark_closing` es una optimización, no una dependencia de
  corrección.
- **Las baselines usan un stake plano de 1 unidad**, no Kelly: no tienen una
  probabilidad propia contra la cual dimensionar una fracción de Kelly. Por
  eso su `pnl` medio ya *es* su ROI por apuesta, mientras que el ROI de la
  estrategia real se calcula ponderado por `stake_fraction` — son dos
  unidades distintas, comparables solo en CLV, no en pnl absoluto.
- **Las baselines resuelven la selección sobre el mismo instante exacto de
  cuota que ya usó la señal real** (mismo bookmaker, mismo `captured_at`),
  nunca resolviendo el mercado de nuevo por su cuenta. Comparar contra una
  cuota capturada en otro momento haría que cualquier diferencia de
  desempeño fuera un artefacto de qué precio se miró, no de las estrategias.
- **`min_matches_per_window` se reutiliza como el mínimo de apuestas
  liquidadas por grupo (tier, mercado o baseline) para reportar un intervalo
  de confianza de CLV**, no como una ventana de validación por temporada
  (que es como se usa en la Fase 3). El nombre del campo en
  `config/thresholds.yaml` es ambiguo entre ambos usos; se documenta aquí en
  vez de introducir un campo de configuración nuevo para lo mismo.
- **`empate` en moneyline liquida como `push`** (nadie gana, el stake se
  devuelve): el mercado `moneyline` solo tiene selecciones `home`/`away`, y
  un empate real (posible sobre todo en NFL) no favorece a ninguna.
- **Los partidos pospuestos o cancelados nunca se liquidan.** Sin marcador
  final no hay outcome que calcular; la liquidación explícita de esos casos
  (con `outcome=void`) queda para cuando el pipeline de ingesta distinga
  activamente "pospuesto para siempre" de "pospuesto, reprogramado" — hoy
  ese partido simplemente no genera una fila en `results`.

## Alcance de la Fase 6

- **Dos superficies, no una, porque leen datos en momentos distintos.** La
  API en vivo (`api/app.py`) consulta los repositorios en el instante de
  cada petición; el export estático (`export/json_export.py`) escribe una
  foto fija que el frontend sirve sin servidor. `Settings.export_dir`'s
  propio docstring ("Pre-computed JSON consumed by the static frontend") ya
  documentaba esta intención desde la Fase 0 — el export es lo que
  `frontend/` (Fase 7) realmente consume en producción; la API en vivo es
  para desarrollo local o como vía alternativa, nunca lo contrario.
- **`only_actionable=True` por defecto en la API viva, pero el export
  siempre escribe todos los tiers.** Una API se puede volver a consultar
  con otro filtro; un archivo JSON estático no — dejar `descartar` fuera
  del export lo haría invisible para siempre, no simplemente no-interesante
  por defecto. Pedir un `tier` explícito en la API (incluido `descartar`)
  sigue funcionando: es una consulta con sentido propio, no un accidente.
- **Sin autenticación en la API en vivo.** Es una herramienta de un solo
  operador (el propio dueño del proyecto), no un servicio multi-usuario;
  exponerla más allá de una red de confianza sin añadir autenticación
  propia queda fuera de este alcance, y se documenta así en el docstring
  del módulo en vez de fingir que ya está resuelto.
- **`teams` se lee entera, sin filtrar por competición**, para resolver
  nombre canónico en `list_signals`: esa tabla no tiene columna
  `competition_id` (un equipo puede jugar en más de una), así que no hay
  forma más barata de acotar la lectura sin arriesgarse a dejar un equipo
  sin nombre.
- **Una competición desconocida levanta `KeyError`** (404 en la API, fallo
  ruidoso en el export) en vez de devolver una lista vacía en silencio —
  igual que las demás validaciones del proyecto, una superficie de lectura
  pública no debería verse igual cuando el recurso no existe que cuando
  simplemente no tiene datos todavía.

## Alcance de la Fase 7

- **MVP deliberadamente minimalista.** Una sola vista (sin router:
  cambiar de competición es un `<select>`, no una URL distinta), sin
  librería de componentes ni gestor de estado — `fetch` + `useState` alcanza
  para leer un puñado de archivos JSON estáticos. Ordenar/paginar la tabla
  de señales, gráficas de la curva de fiabilidad o de CLV en el tiempo, y
  cualquier interactividad más allá del filtro por tier quedan fuera hasta
  que haya una razón concreta para añadirlas.
- **Sin suite de tests de frontend.** El rigor de 100% de cobertura es una
  regla del backend Python (`pyproject.toml`'s `[tool.pytest]`); el
  frontend se verifica con el build de TypeScript en modo `strict`, el
  linter (`oxlint`) y una verificación visual manual contra datos
  sintéticos — documentado aquí en vez de fingir una cobertura que no
  existe. Añadir Vitest/Testing Library es trabajo futuro razonable si el
  frontend crece más allá de este MVP.
- **`frontend/public/data/` es generado, no fuente** — `.gitignore` lo
  excluye salvo un `.gitkeep` que mantiene el directorio. `deportivas
  export run` lo puebla; sin haberlo corrido antes, el sitio carga y
  muestra "no se pudo cargar la lista de competiciones" en vez de fallar
  en blanco — un estado de error legible, no una pantalla vacía sin
  explicación.
- **Los tipos TypeScript en `src/types.ts` son un espejo manual** de los
  modelos Pydantic de `api/views.py`, no generados automáticamente. Si un
  campo cambia de un lado, hay que cambiarlo del otro a mano; generación
  automática de tipos (p.ej. desde el `openapi.json` que FastAPI ya expone
  en `api/app.py`) es una mejora futura razonable, no algo que este MVP
  necesitaba para demostrar que el patrón export-JSON-estático funciona.

## Alcance de la Fase 8

- **`mark_closing` es la única excepción a "append-only nunca se
  actualiza"**, y deliberadamente acotada: `TableRepository.mark_closing`
  solo existe para `odds_snapshots`, solo pone `is_closing=True`, y lanza
  `ValueError` si se llama sobre cualquier otra tabla. No es una puerta
  general para actualizar datos históricos — es una optimización de
  lectura (`closing_price` ya funciona sin ella, ver
  [Fase 5](#alcance-de-la-fase-5)) implementada en los dos backends
  (reescritura de partición en Parquet, `UPDATE` real en Postgres) y
  probada contra ambos.
- **`ingest/closing.py` decide "cerrado" por fixture, no por snapshot
  individual**: agrupa por `(bookmaker, market, selection, line)` y marca
  el último `captured_at` antes del kickoff de cada grupo — la línea de
  1x2-home de Pinnacle y la de asian_handicap del mismo bookmaker cierran
  en instantes distintos, y cada una necesita su propio "último antes del
  kickoff". Salta un fixture si ya tiene alguna fila `is_closing=True`
  (idempotente: seguro de correr a diario sin recalcular de más).
- **El formato de temporada de `domain/seasons.py` sale de los adaptadores
  ya existentes, no de una convención inventada aquí**: fútbol usa el
  código de dos años porque así lo esperan fbref/football-data.co.uk
  (`ingest/sources/fbref.py`, `footballdata.py`); el resto usa el año
  simple porque así lo esperan nfl_data_py/pybaseball/sportsdataverse. La
  fecha de corte entre una temporada y la siguiente es
  `season_start_month`, ya declarado por competición en
  `competitions.yaml` — no un segundo campo nuevo para lo mismo.
- **`list-competitions`/`current-seasons` son primitivas para que un
  workflow decida, no para decidir ellas mismas.** Devuelven datos
  (JSON, una lista de temporadas); qué hacer con esos datos —qué comando de
  ingesta correr, en qué orden— lo decide `scripts/run_daily_pipeline.sh`,
  siguiendo el mismo principio que el CLI de ingesta ya declara en su
  propio docstring desde la Fase 1.

## Alcance de la Fase 9

- **Por qué cuatro workflows y no uno.** `sources-health.yml` no toca datos;
  `daily.yml`/`odds.yml` producen datos pero no saben nada de Vite ni de
  Pages; `deploy.yml` sabe de Vite y de Pages pero no de ingesta. Cada uno
  puede fallar, reintentarse o cambiar de cadencia sin arrastrar a los
  demás — el costo es un `workflow_run` encadenando `deploy.yml`, no una
  gran orquestación central.
- **Por qué un Release fijo (`data-lake`) y no un tag por versión.** No hay
  "versiones" de un data lake que crece a diario — hay un estado actual.
  Un tag fijo con `--clobber` en cada publicación es más simple que idear un
  esquema de versionado para algo que nunca se necesita mirar hacia atrás
  (el historial real vive en la capa cruda append-only dentro del propio
  *asset*, no en Releases anteriores).
- **`--regions eu` en `run_odds_pipeline.sh`, no en el CLI.** El comando
  `ingest odds-snapshot` expone `--regions` con el default de tres regiones
  del adaptador (`uk,eu,us`) sin opinar — el guión de automatización es
  quien conoce el presupuesto real y elige recortarlo. Ver la sección
  ["El presupuesto de The Odds API"](#el-presupuesto-de-the-odds-api) para
  las cuentas exactas detrás de esa elección.
- **La cadencia `weekly` se decide por día de la semana dentro del script,
  no con un segundo cron.** `run_daily_pipeline.sh` calcula `date -u +%u` y
  compara contra `WEEKLY_DAY` (domingo por defecto) en vez de que
  `daily.yml` declare dos triggers de cron distintos — un solo punto de
  entrada, una sola variable de entorno como palanca, en vez de dos YAMLs
  que tienen que mantenerse en sync.
- **Nada de esto se ha ejecutado todavía de verdad.** Ver el aviso al final
  de ["Arquitectura de despliegue gratuito"](#arquitectura-de-despliegue-gratuito):
  esta sesión de desarrollo no tiene acceso a GitHub Actions, Releases ni
  red externa para probar la orquestación completa de punta a punta, más
  allá de lo que YAML/bash/pytest permiten verificar sin red.

## Alcance de la Fase 10

- **Una excepción real encontrada al construir el chequeo, no solo un
  chequeo nuevo.** `soccerdata` rechazaba `NED-Eredivisie`,
  `POR-Primeira Liga` y las tres claves `INT-*` de UEFA con `ValueError`
  antes de intentar ninguna llamada de red, porque nada generaba el
  `league_dict.json` personalizado que el propio paquete documenta
  (ver `ingest/soccerdata_config.py`). Las seis competiciones que la
  [Fase 1](#limitaciones-conocidas-de-la-fase-1) marcó como "no
  verificadas todavía" estaban, en la práctica, completamente rotas — no
  solo sin verificar.
- **Un DataFrame vacío no es un fallo.** `sources_health.py` solo reporta
  una excepción real (red, HTTP, "liga inválida"); una liga real fuera de
  temporada devuelve legítimamente cero partidos, y tratar eso como fallo
  produciría ruido constante en vez de señal.
- **Reutiliza los adaptadores de producción, no un cliente HTTP aparte.**
  Validar con el mismo rate limiting y el mismo archivado en capa cruda que
  `deportivas ingest` evita que "validar" y "correr de verdad" tengan
  comportamientos de red distintos — la única diferencia es que el
  resultado se descarta en vez de persistirse.
- **No participa del data lake.** No restaura ni publica el Release
  `data-lake`: lo que archiva en la capa cruda durante la corrida vive y
  muere con el runner efímero.

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
mismo comando con una lista de temporadas distinta — este CLI no adivina
ventanas de fechas por su cuenta. `scripts/run_daily_pipeline.sh` (Fase 8)
es lo que programa las corridas diarias de "temporada actual", apoyándose
en dos comandos más pensados para eso:

```bash
uv run deportivas list-competitions      # todas las competiciones habilitadas, como JSON
uv run deportivas current-seasons --competition-id eng-premier-league --count 2
uv run deportivas ingest mark-closing --competition-id eng-premier-league
uv run deportivas sources-health          # valida fuentes en vivo, sale con codigo 1 si algo no cuadra
```

`list-competitions` es lo que un workflow recorre con `jq` para decidir qué
comando de ingesta correr por competición, sin parsear YAML a mano en bash.
`current-seasons` calcula la temporada actual en el formato que cada fuente
espera (código de dos años para fútbol, año simple para el resto) y se pasa
directo a `--seasons`. `mark-closing` marca `is_closing=True` sobre el
último snapshot antes del kickoff de cada fixture ya arrancado — una
optimización sobre el fallback que `backtest/settlement.py` ya calcula en
tiempo de lectura, nunca una dependencia (ver
[Fase 8](#estado-del-proyecto)); seguro de correr a diario, salta los
fixtures ya marcados. `sources-health` reutiliza los mismos adaptadores
para validar identificadores y claves contra la fuente real sin persistir
nada (ver [Fase 10](#estado-del-proyecto)).

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

# NFL/NBA/NHL/MLB requieren ademas que su pipeline de features ya haya
# corrido (deportivas features compute-nfl, compute-nba, ...).
uv run deportivas models train-nfl --competition-id usa-nfl
uv run deportivas models train-nba --competition-id usa-nba
uv run deportivas models train-nhl --competition-id usa-nhl
uv run deportivas models train-mlb --competition-id usa-mlb

# Metodo de calibracion explicito en vez del de config/thresholds.yaml:
uv run deportivas models train-football \
  --competition-id eng-premier-league --calibration-method platt
```

A diferencia de `features compute-...`, esto **no** es idempotente por
diseño: `model_registry` es `append_only` (una fila por corrida de
entrenamiento, nunca sobreescrita) para conservar el historial completo de
cada ventana entrenada, incluso si se re-ejecuta el mismo comando dos veces.

## CLI de señales

```bash
uv run deportivas signals --help

# Requiere que ya existan predictions (deportivas models train-...) y
# odds_snapshots para la misma competicion.
uv run deportivas signals generate --competition-id eng-premier-league
```

Sí es idempotente: `signals.write` hace upsert sobre `id` (regla de todas
las tablas no `append_only`), así que re-ejecutarlo tras capturar cuotas
nuevas actualiza cada señal en vez de duplicarla.

## CLI de backtest

```bash
uv run deportivas backtest --help

# Requiere que ya existan signals (deportivas signals generate) para la
# misma competicion.
uv run deportivas backtest settle --competition-id eng-premier-league

# Requiere haber corrido "settle" primero.
uv run deportivas backtest report --competition-id eng-premier-league
```

`settle` es idempotente por el mismo motivo que `signals generate`
(`results.write` hace upsert sobre `signal_id`): volver a correrlo tras
capturar más cuotas de cierre actualiza `closing_price`/`clv` en vez de
duplicar la fila. `report` no escribe nada — solo lee `results` y `signals`,
agrega, y muestra el CLV medio (con su intervalo de confianza cuando hay
datos suficientes) y el ROI, global y desglosado, junto a las baselines.

## CLI de export

```bash
uv run deportivas export --help

# Una competicion, o todas las habilitadas si se omite --competition-id.
uv run deportivas export run --competition-id eng-premier-league
uv run deportivas export run
```

Escribe `frontend/public/data/competitions.json` y, por competición,
`{competition_id}/signals.json` + `{competition_id}/backtest.json` — lo que
el frontend estático (Fase 7) sirve sin necesitar ningún servidor corriendo.
Sobrescribe los ficheros existentes por completo en cada corrida; no hay
nada que "actualizar" en un JSON estático.

Para desarrollo local contra la API en vivo en lugar del export (ver
[alcance de la Fase 6](#alcance-de-la-fase-6) sobre cuándo usar cada una):

```bash
uv run uvicorn deportivas.api.app:app --reload
# docs interactivas en http://localhost:8000/docs
```

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
implementa a mano — Parquet no tiene `ON CONFLICT`. `data/raw/` y
`data/parquet/` se publican como el *asset* de un Release fijo de GitHub
(Fase 9, `scripts/{restore,publish}_data_lake.sh`), nunca como archivos del
repositorio — ver [Fase 9](#arquitectura-de-despliegue-gratuito).

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

## Arquitectura de despliegue gratuito

Sin servidor encendido, sin tarjeta de crédito: todo corre en runners
efímeros de GitHub Actions (gratis en un repositorio público) que no
conservan nada entre corridas, y el resultado se sirve desde GitHub Pages
(gratis, estático). Cuatro workflows, cada uno con una sola responsabilidad
— la orquestación vive en `scripts/*.sh` y en el YAML, nunca en código
Python nuevo, siguiendo el mismo principio que ya rige `cli.py`:

| Workflow | Cadencia | Qué hace |
|---|---|---|
| `sources-health.yml` (Fase 10) | diario, 05:00 UTC | Valida en vivo los identificadores de fuente y las claves de The Odds API. No toca el data lake. |
| `daily.yml` (Fase 8) | diario, 06:00 UTC | Ingesta incremental + features + modelos (según cadencia `daily`/`weekly`) + cierre de línea + señales + liquidación, para toda competición habilitada. |
| `odds.yml` (Fase 9) | lunes y jueves, 12:00 UTC | Captura de cuotas (The Odds API) + la misma cadena de cierre/señales/liquidación. |
| `deploy.yml` (Fase 7/9) | tras `daily.yml`/`odds.yml`, o al tocar `frontend/` | Exporta el JSON estático más fresco y publica `frontend/` en GitHub Pages. |

### El Release de GitHub como disco persistente

Un runner de GitHub Actions no conserva disco entre corridas — la pieza que
hace que esto funcione igual es un único Release fijo, con el tag literal
`data-lake` (no es una versión de software), cuyo único *asset*
(`deportivas-data-lake.tar.gz`, empaquetando `data/raw/` y `data/parquet/`
— nunca `data/cache/`, que es regenerable) se reemplaza en cada corrida:

- `scripts/restore_data_lake.sh` lo descarga y desempaqueta al principio de
  `daily.yml`/`odds.yml`. Si el Release todavía no existe (primera corrida
  jamás), lo avisa y arranca con `data/` vacío — nunca falla por esto.
- `scripts/publish_data_lake.sh` lo vuelve a subir (`--clobber`) al final,
  siempre (`if: always()`), incluso si el pipeline falló a mitad de camino:
  lo que sí se alcanzó a ingerir no se pierde.

`daily.yml` y `odds.yml` comparten el grupo de concurrencia
`deportivas-data-lake` (`cancel-in-progress: false`): dos corridas escribiendo
el mismo *asset* a la vez se pisarían una a otra, así que la segunda espera
a que la primera termine de publicar en vez de correr en paralelo.

### Del dato al sitio: `deploy.yml`

`daily.yml`/`odds.yml` no saben nada de Vite ni de GitHub Pages — solo
producen datos. `deploy.yml` es el único que sí, encadenado con
`workflow_run` justo después de que cualquiera de los otros dos termine (mas
`workflow_dispatch` y un `push` a `frontend/`, para que un cambio de UI no
espere a la próxima corrida de datos): descarga el data lake más reciente,
corre `deportivas export run` para regenerar `frontend/public/data/*.json`
fresco, construye el frontend y lo publica. `frontend/public/data/` sigue
siendo generado, no fuente ([Fase 7](#alcance-de-la-fase-7)) — este es
literalmente el único lugar donde se genera de verdad en producción.

### El presupuesto de The Odds API

El plan gratuito son 500 créditos/mes, y cada llamada a
`/v4/sports/{sport}/odds` cuesta `mercados × regiones` créditos — el número
de partidos que devuelve no importa. `scripts/run_odds_pipeline.sh` fija
`--regions eu` (Pinnacle, la referencia de este proyecto, vive ahí) en vez
del default de tres regiones del CLI: una corrida completa de las
competiciones con captura de cuotas activa (12 habilitadas menos Colombia,
que no tiene `odds.the_odds_api` — ver
["Sobre las fuentes de datos"](#sobre-las-fuentes-de-datos-qué-existe-y-qué-no);
las 3 UEFA ni siquiera se intentan, están deshabilitadas del todo) cuesta
~32 créditos con una región, ~96 con tres. A ~32 créditos/corrida,
`odds.yml` corriendo lunes y jueves (~9 corridas/mes) usa ~288
créditos/mes — deja bastante margen para reintentos manuales
(`workflow_dispatch`) sin tocar el tope. Mover la cadencia o las regiones es
la misma palanca que ya existe para el tiempo de `daily.yml`
(`config/competitions.yaml`'s `refresh: daily | weekly`): un número en un
sitio, no una reescritura.

### FBref bloquea con CAPTCHA a los runners de GitHub Actions

Descubierto en la primera corrida real de `sources-health.yml`: FBref le
sirve un CAPTCHA al runner (IP de datacenter), y el solver de `soccerdata`
(basado en PyAutoGUI) es un *no-op* en modo headless — solo lo intenta de
verdad con `headless=False`, que a su vez necesita una pantalla virtual o
Chrome no tiene dónde dibujar.

Se probó evadirlo: `Settings.fbref_headless=False` + un paso que instala
Xvfb + `xvfb-run -a` envolviendo la corrida real, en `daily.yml` y
`sources-health.yml`. Confirmado en producción que el cambio funcionaba al
nivel de infraestructura — el solver ya no hacía el no-op, intentaba
resolver el CAPTCHA de verdad — pero perdió los 5 reintentos en las 11
competiciones de fútbol igual, y quedó ~25-30s más lento por competición
por el overhead de Xvfb, sin ninguna ganancia a cambio.

Con el bloqueo confirmado y repetible (tres corridas reales seguidas, misma
lista exacta de 11 fallos cada vez), seguir intentando FBref en cada
corrida automatizada es tiempo de CI gastado en un resultado ya conocido —
no solo revertir a `headless=True`, sino desactivar FBref del todo ahí:
`daily.yml`/`sources-health.yml` ponen `DEPORTIVAS_FBREF_ENABLED=false`
(ver `Settings.fbref_enabled` en `config/settings.py`). El código que se
probó (`FBrefSource(headless=...)`, `Settings.fbref_headless`) se queda en
el repositorio a propósito, como constancia probada y con tests de que ese
camino no funciona, y sigue disponible para uso manual: `deportivas
fbref-schedule`/`fbref-stats`, corridos a mano desde una máquina con IP
normal (no de datacenter), suelen pasar sin que FBref los bloquee — solo el
runner de GitHub Actions está identificado y bloqueado.

Como FBref sigue sin funcionar desde el runner, football-data.co.uk y ESPN
cubren calendario/resultados de forma redundante para 8 de las 11
competiciones de fútbol (ver
["Sobre las fuentes de datos"](#sobre-las-fuentes-de-datos-qué-existe-y-qué-no)).
Las tres competiciones UEFA, que dependían de FBref *o* de ESPN (bloqueado
por un bug propio de `soccerdata` con calendarios por etapas — ver
[Limitaciones conocidas](#limitaciones-conocidas-de-la-fase-1)), se quedan
sin ninguna fuente funcional y por eso están `enabled: false` en
`competitions.yaml` en vez de generando un `FALLO` sin remedio en cada
corrida.

### Notificación de fallos

Los cuatro workflows —y `ci.yml`— comparten un job reutilizable
(`.github/workflows/notify-failure.yml`, invocado con `uses:` en vez de
copiar el mismo bloque cuatro veces) que abre un issue etiquetado, o
comenta en el ya abierto para ese workflow, en vez de fallar en silencio
(regla de la Fase 10, aplicada desde la Fase 0 en `ci.yml`).

### Configuración manual, una sola vez

Nada de esto corre solo con el código en el repositorio — hace falta, una
vez, fuera de este repositorio:

1. **Settings → Pages → Source = "GitHub Actions"** — sin esto,
   `deploy.yml` construye el sitio pero no tiene dónde publicarlo.
2. **Settings → Secrets and variables → Actions**, crear `THE_ODDS_API_KEY`
   (la clave de [the-odds-api.com](https://the-odds-api.com), plan
   gratuito) — sin ella, `odds.yml` y la mitad de `sources-health.yml` se
   saltan en silencio (documentado así en sus propios docstrings), no
   fallan.
3. **Settings → Actions → General → Workflow permissions** — confirmar que
   permite "Read and write permissions", para que `publish_data_lake.sh`
   pueda crear/actualizar el Release y `deploy.yml` pueda publicar en
   Pages.

Ninguno de los cuatro workflows se ha ejecutado todavía de verdad: esta
sesión de desarrollo no tiene acceso a GitHub Actions, a Releases reales ni
a red externa (FBref, The Odds API) para probarlos de punta a punta. Cada
uno está verificado por separado hasta donde es posible sin red — sintaxis
YAML, la lógica de cada script contra `deportivas list-competitions` real
y con `uv`/dobles simulando cada comando, y el código Python detrás de cada
CLI con cobertura del 100 % — pero la primera corrida real, después de los
tres pasos de arriba, es la que confirma que la orquestación completa
funciona de punta a punta.

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

### Frontend (opcional, Fase 7)

Requisitos: Node 22+.

```bash
uv run deportivas export run           # puebla frontend/public/data/
cd frontend && npm install && npm run dev
```

## Estructura del repositorio

```
config/                  YAML: competiciones, mercados, umbrales de decisión
src/deportivas/
  config/                 Settings (Pydantic) + carga validada de config/*.yaml
  domain/                 Enums cerrados, ids deterministas, guardián de leakage temporal,
                          seasons.py (temporada actual por competición, Fase 8)
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
    closing.py                Marca is_closing sobre el ultimo snapshot pre-kickoff (Fase 8)
    soccerdata_config.py      Genera el league_dict.json personalizado que soccerdata necesita
    sources_health.py         Valida fuentes en vivo sin persistir nada (Fase 10)
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
    features_loader.py        Une la tabla features con el resultado real del fixture
    feature_matrix.py         Vectoriza dicts de features, imputa con la media de entrenamiento
    moneyline.py               Clasificador logistico generico P(home win)
    moneyline_training.py      Walk-forward compartido por NFL/NBA/NHL/MLB
    football/                 Poisson bivariante: matriz de goles -> 1x2/over_under/btts
    nfl/ nba/ nhl/ mlb/        train.py: wrapper fino sobre moneyline_training.py
  odds/
    resolve.py                Resolucion de mercado punto-en-tiempo, compartida por signals y backtest
  signals/
    devig.py                 Quita el margen de la cuota (multiplicativo, power, Shin)
    tiers.py                  Clasifica alta/media/baja/descartar contra thresholds.yaml
    staking.py                 Stake de Kelly fraccionado, tope duro
    generate.py                 Une predictions + odds_snapshots -> signals
  backtest/
    settlement.py             Liquida signals contra el marcador final -> results, CLV incluido
    bootstrap.py               Intervalo de confianza por remuestreo, generico (CLV o pnl)
    baselines.py                always_favourite / random, mismo instante de cuota que la senal real
    report.py                   CLV/ROI global, por tier, por mercado y contra cada baseline
  api/
    views.py                  Capa de vistas pydantic, sin fastapi -- usada por app.py y export/
    app.py                     FastAPI de solo lectura, para desarrollo local
  export/
    json_export.py            Escribe los mismos datos de views.py como JSON estatico para frontend/
  cli.py                   Un comando por adaptador de ingesta, pipeline de features, modelo,
                          senal, backtest, export, list-competitions/current-seasons/
                          mark-closing/sources-health (Fase 8/10)
alembic/                  Migraciones sobre la metadata de contracts/
frontend/                 Vite + React 19 + TS + Tailwind v4, lee public/data/*.json
  src/
    types.ts                 Espejo manual de los modelos pydantic de api/views.py
    api.ts                    fetch de competitions.json / {id}/signals.json / {id}/backtest.json
    useFetch.ts                Hook generico de carga/error para los tres
    components/                CompetitionSelector, SignalsTable, BacktestSummary, TierBadge
  public/data/                Generado por "deportivas export run" -- gitignored salvo .gitkeep
scripts/                  Orquestacion de la Fase 9 (bash, no Python nuevo)
  lib_common.sh             Listado de competiciones + cadena de cierre/senales/liquidacion
  run_daily_pipeline.sh      Ingesta/features/modelos por cadencia + liquidacion diaria
  run_odds_pipeline.sh       Captura de cuotas + liquidacion
  restore_data_lake.sh       Descarga data/raw + data/parquet del Release "data-lake"
  publish_data_lake.sh       Los vuelve a publicar ahi
tests/
  unit/ contracts/ fixtures/
.github/workflows/        ci.yml, daily.yml, odds.yml, deploy.yml, sources-health.yml
                          (Fase 8/9/10) y notify-failure.yml (reutilizable, ver
                          "Arquitectura de despliegue gratuito")
```

## Variables de entorno

Ver [`.env.example`](.env.example) — documentado línea por línea. Ninguna
clave real vive en este repositorio; en GitHub Actions viven como Secrets.
