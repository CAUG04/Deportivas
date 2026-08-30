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
decisiones de diseño (el precio de cierre sin el flag de Fase 8 todavía,
el stake plano de las baselines, y qué significa aquí
`min_matches_per_window`).

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

- **El precio de cierre cae al último snapshot pre-kickoff sin el flag de
  Fase 8 todavía.** `results.clv` se mide contra una fila marcada
  `is_closing=True`, pero ese flag lo pone un job de liquidación de la Fase
  8 que corre después del kickoff y que todavía no existe. Mientras tanto,
  `closing_price` cae al último precio capturado antes del kickoff para el
  mismo bookmaker de entrada — la mejor aproximación disponible a "el precio
  en el que se cerró el mercado", documentada como tal en el docstring de
  `settlement.py`, no un dato inventado. En cuanto la Fase 8 empiece a
  marcar `is_closing`, ese flag gana automáticamente sin tocar una línea de
  código aquí.
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
  cli.py                   Un comando por adaptador de ingesta, pipeline de features, modelo, senal, backtest y export
alembic/                  Migraciones sobre la metadata de contracts/
frontend/                 Vite + React 19 + TS + Tailwind v4, lee public/data/*.json
  src/
    types.ts                 Espejo manual de los modelos pydantic de api/views.py
    api.ts                    fetch de competitions.json / {id}/signals.json / {id}/backtest.json
    useFetch.ts                Hook generico de carga/error para los tres
    components/                CompetitionSelector, SignalsTable, BacktestSummary, TierBadge
  public/data/                Generado por "deportivas export run" -- gitignored salvo .gitkeep
tests/
  unit/ contracts/ fixtures/
.github/workflows/        CI (incluye servicio Postgres); daily/odds/deploy/sources-health en fases posteriores
```

## Variables de entorno

Ver [`.env.example`](.env.example) — documentado línea por línea. Ninguna
clave real vive en este repositorio; en GitHub Actions viven como Secrets.
