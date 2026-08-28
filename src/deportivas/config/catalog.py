"""Loads and validates ``config/*.yaml`` into typed models.

This is the boundary between "change a YAML line" and "write Python": every
other module reads competitions, markets and thresholds through here, never
by parsing YAML itself. Validation happens once, at load time, so a typo in a
threshold fails loudly instead of silently disabling a tier.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deportivas.config.settings import CONFIG_DIR
from deportivas.domain.enums import DevigMethod, RefreshCadence, Sport

# --------------------------------------------------------------------------
# competitions.yaml
# --------------------------------------------------------------------------


class CompetitionSources(BaseModel):
    model_config = ConfigDict(extra="allow")  # nfl_data_py, sportsdataverse, pybaseball...

    soccerdata_key: str | None = None
    fbref: str | None = None
    understat: str | None = None
    club_elo: str | None = None
    match_history: str | None = None
    espn: str | None = None


class CompetitionOdds(BaseModel):
    the_odds_api: str


class Competition(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    country: str
    sport: Sport
    tier: int = Field(ge=1)
    season_start_month: int = Field(ge=1, le=12)
    season_end_month: int = Field(ge=1, le=12)
    seasons_back: int = Field(ge=1)
    refresh: RefreshCadence
    enabled: bool
    sources: CompetitionSources
    odds: CompetitionOdds


class CompetitionsCatalog(BaseModel):
    version: int
    competitions: tuple[Competition, ...]

    @classmethod
    def from_raw(cls, raw: dict[str, object]) -> CompetitionsCatalog:
        """Applies ``defaults:`` to each competition before validation.

        Plain YAML has no notion of defaults; this is where that convenience
        from ``competitions.yaml`` actually gets implemented. A competition's
        own value always wins over the default.
        """
        defaults = raw.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ValueError("competitions.yaml: 'defaults' debe ser un mapeo")
        raw_competitions = raw.get("competitions", [])
        if not isinstance(raw_competitions, list):
            raise ValueError("competitions.yaml: 'competitions' debe ser una lista")
        merged = [{**defaults, **comp} for comp in raw_competitions]
        return cls.model_validate({"version": raw.get("version"), "competitions": merged})

    @model_validator(mode="after")
    def _unique_ids(self) -> CompetitionsCatalog:
        ids = [c.id for c in self.competitions]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"competition ids duplicados en competitions.yaml: {sorted(dupes)}")
        return self

    @property
    def enabled(self) -> tuple[Competition, ...]:
        return tuple(c for c in self.competitions if c.enabled)

    def get(self, competition_id: str) -> Competition:
        for c in self.competitions:
            if c.id == competition_id:
                return c
        raise KeyError(f"competicion desconocida: {competition_id!r}")


# --------------------------------------------------------------------------
# markets.yaml
# --------------------------------------------------------------------------


class MarketSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    sports: tuple[Sport, ...]
    selections: tuple[str, ...]
    needs_line: bool
    default_lines: tuple[float, ...] = ()
    derived_from: str

    @model_validator(mode="after")
    def _lines_consistency(self) -> MarketSpec:
        # default_lines es una rejilla fija para evaluar la matriz de goles
        # (over/under, hándicap asiatico). spread/total en deportes americanos
        # tambien necesitan linea, pero la trae el mercado en vivo: no hay una
        # rejilla fija que declarar aqui.
        if self.derived_from == "score_matrix" and self.needs_line and not self.default_lines:
            raise ValueError(f"{self.id}: needs_line=true con score_matrix requiere default_lines")
        if not self.needs_line and self.default_lines:
            raise ValueError(f"{self.id}: needs_line=false no deberia declarar default_lines")
        return self


class Bookmakers(BaseModel):
    sharp: tuple[str, ...]
    soft: tuple[str, ...]

    @model_validator(mode="after")
    def _no_overlap(self) -> Bookmakers:
        overlap = set(self.sharp) & set(self.soft)
        if overlap:
            raise ValueError(f"casas en sharp y soft a la vez: {sorted(overlap)}")
        return self


class MarketsCatalog(BaseModel):
    version: int
    markets: tuple[MarketSpec, ...]
    bookmakers: Bookmakers

    def get(self, market_id: str) -> MarketSpec:
        for m in self.markets:
            if m.id == market_id:
                return m
        raise KeyError(f"mercado desconocido: {market_id!r}")

    def for_sport(self, sport: Sport) -> tuple[MarketSpec, ...]:
        return tuple(m for m in self.markets if sport in m.sports)


# --------------------------------------------------------------------------
# thresholds.yaml
# --------------------------------------------------------------------------


class DevigConfig(BaseModel):
    method: DevigMethod
    reference_bookmaker: str
    fallback_bookmakers: tuple[str, ...]


class TierConfig(BaseModel):
    min_edge: float
    min_sample_matches: int = Field(ge=0)
    requires_calibration: bool
    requires_sharp_price: bool
    requires_favourable_line_move: bool


class TiersConfig(BaseModel):
    alta: TierConfig
    media: TierConfig
    baja: TierConfig

    @model_validator(mode="after")
    def _decreasing_edge(self) -> TiersConfig:
        if not (self.alta.min_edge >= self.media.min_edge >= self.baja.min_edge):
            raise ValueError("min_edge debe decrecer: alta >= media >= baja")
        return self


class LineMoveConfig(BaseModel):
    favourable_drop: float = Field(gt=0)
    min_snapshots: int = Field(ge=1)


class CalibrationConfig(BaseModel):
    method: str
    min_training_samples: int = Field(gt=0)
    max_brier_score: float = Field(gt=0, le=1)
    reliability_bins: int = Field(gt=0)


class StakingConfig(BaseModel):
    kelly_fraction: float = Field(gt=0, le=1)
    max_stake_per_bet: float = Field(gt=0, le=1)
    min_stake_per_bet: float = Field(ge=0)
    bankroll_unit: float = Field(gt=0)

    @model_validator(mode="after")
    def _min_below_max(self) -> StakingConfig:
        if self.min_stake_per_bet >= self.max_stake_per_bet:
            raise ValueError("min_stake_per_bet debe ser menor que max_stake_per_bet")
        return self


class BacktestConfig(BaseModel):
    bootstrap_iterations: int = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)
    min_matches_per_window: int = Field(gt=0)
    baselines: tuple[str, ...]


class DataSufficiencyConfig(BaseModel):
    min_matches_with_odds: int = Field(ge=0)
    min_seasons: int = Field(ge=0)


class ThresholdsCatalog(BaseModel):
    version: int
    devig: DevigConfig
    tiers: TiersConfig
    line_move: LineMoveConfig
    calibration: CalibrationConfig
    staking: StakingConfig
    backtest: BacktestConfig
    data_sufficiency: DataSufficiencyConfig


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: se esperaba un mapeo en la raiz del YAML")
    return data


@functools.lru_cache(maxsize=1)
def load_competitions(config_dir: Path = CONFIG_DIR) -> CompetitionsCatalog:
    return CompetitionsCatalog.from_raw(_load_yaml(config_dir / "competitions.yaml"))


@functools.lru_cache(maxsize=1)
def load_markets(config_dir: Path = CONFIG_DIR) -> MarketsCatalog:
    return MarketsCatalog.model_validate(_load_yaml(config_dir / "markets.yaml"))


@functools.lru_cache(maxsize=1)
def load_thresholds(config_dir: Path = CONFIG_DIR) -> ThresholdsCatalog:
    return ThresholdsCatalog.model_validate(_load_yaml(config_dir / "thresholds.yaml"))
