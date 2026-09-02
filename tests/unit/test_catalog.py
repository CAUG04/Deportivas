"""Tests for the config catalog loader: both that the real config/*.yaml files
validate, and that the Pydantic models reject malformed configuration instead
of silently accepting a bad threshold."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from deportivas.config.catalog import (
    CompetitionsCatalog,
    MarketsCatalog,
    ThresholdsCatalog,
    load_competitions,
    load_markets,
    load_thresholds,
)
from deportivas.config.settings import CONFIG_DIR
from deportivas.domain.enums import Sport

# --------------------------------------------------------------------------
# Los YAML reales del repo deben cargar sin errores.
# --------------------------------------------------------------------------


def test_real_competitions_yaml_loads() -> None:
    load_competitions.cache_clear()
    catalog = load_competitions()
    assert len(catalog.competitions) == 15
    ids = {c.id for c in catalog.competitions}
    assert "eng-premier-league" in ids
    assert "col-primera-a" in ids
    assert "usa-nfl" in ids


def test_uefa_competitions_are_disabled() -> None:
    """Confirmado en produccion (sources-health.yml): ni FBref (CAPTCHA) ni
    ESPN (TypeError de soccerdata con calendarios por etapas) tienen fuente
    de calendario funcional para UEFA hoy -- se quedan declaradas (no se
    borran) pero fuera del pipeline hasta que una de las dos se resuelva."""
    load_competitions.cache_clear()
    catalog = load_competitions()
    uefa_ids = {"uefa-champions-league", "uefa-europa-league", "uefa-conference-league"}
    assert {c.id for c in catalog.competitions if c.id in uefa_ids} == uefa_ids
    assert not any(c.enabled for c in catalog.competitions if c.id in uefa_ids)
    assert uefa_ids.isdisjoint({c.id for c in catalog.enabled})


def test_mlb_team_abbreviations_are_the_ones_baseball_reference_uses() -> None:
    """Regresion del fallo real de daily.yml: competitions.yaml traia las
    abreviaturas de uso comun (CWS, KC, SD, SF, TB, WSH) y baseball-reference
    usa otras (CHW, KCR, SDP, SFG, TBR, WSN). Una sola equivocada dejaba a MLB
    sin un solo partido. first_season_map es la misma tabla contra la que
    pybaseball valida, asi que un typo futuro falla aqui y no en produccion."""
    from pybaseball.utils import first_season_map

    load_competitions.cache_clear()
    sources = load_competitions().get("usa-mlb").sources
    abbreviations = getattr(sources, "team_abbreviations", None)

    assert isinstance(abbreviations, list)
    assert len(abbreviations) == 30
    assert sorted(set(abbreviations)) == sorted(abbreviations)  # sin duplicados
    assert [t for t in abbreviations if t not in first_season_map] == []


def test_colombia_has_no_odds_capture() -> None:
    """Confirmado en produccion: 'soccer_colombia_primera_a' no existe en
    /v4/sports de The Odds API. Calendario/resultados (ESPN) siguen
    intactos -- col-primera-a sigue habilitada."""
    load_competitions.cache_clear()
    catalog = load_competitions()
    colombia = catalog.get("col-primera-a")
    assert colombia.enabled is True
    assert colombia.odds.the_odds_api is None


def test_real_markets_yaml_loads() -> None:
    load_markets.cache_clear()
    catalog = load_markets()
    assert catalog.get("1x2").sports == (Sport.FOOTBALL,)
    assert catalog.get("moneyline").sports != (Sport.FOOTBALL,)


def test_for_sport_returns_only_markets_applicable_to_that_sport() -> None:
    load_markets.cache_clear()
    catalog = load_markets()

    football_markets = catalog.for_sport(Sport.FOOTBALL)

    assert all(Sport.FOOTBALL in m.sports for m in football_markets)
    assert {m.id for m in football_markets} >= {"1x2", "btts"}
    assert "moneyline" not in {m.id for m in football_markets}


def test_for_sport_excludes_markets_for_other_sports() -> None:
    load_markets.cache_clear()
    catalog = load_markets()

    baseball_markets = catalog.for_sport(Sport.BASEBALL)

    assert "1x2" not in {m.id for m in baseball_markets}
    assert "spread" not in {m.id for m in baseball_markets}  # spread no aplica a baseball


def test_real_thresholds_yaml_loads() -> None:
    load_thresholds.cache_clear()
    catalog = load_thresholds()
    assert catalog.tiers.alta.min_edge >= catalog.tiers.media.min_edge
    assert catalog.staking.kelly_fraction == pytest.approx(0.25)


def test_competitions_yaml_has_no_duplicate_odds_keys() -> None:
    """the_odds_api sport keys deben ser unicos: una colision silenciosa
    mezclaria las cuotas de dos competiciones distintas. None (sin captura
    de cuotas) no cuenta -- varias competiciones pueden compartirlo sin que
    sea una colision real."""
    load_competitions.cache_clear()
    catalog = load_competitions()
    keys = [c.odds.the_odds_api for c in catalog.competitions if c.odds.the_odds_api is not None]
    assert len(keys) == len(set(keys))


def test_config_dir_points_at_real_directory() -> None:
    assert CONFIG_DIR.is_dir()
    assert (CONFIG_DIR / "competitions.yaml").exists()
    assert (CONFIG_DIR / "markets.yaml").exists()
    assert (CONFIG_DIR / "thresholds.yaml").exists()


# --------------------------------------------------------------------------
# Defaults de competitions.yaml: cada competicion hereda de 'defaults' y puede
# sobreescribir.
# --------------------------------------------------------------------------


def test_defaults_are_applied_and_overridable() -> None:
    raw = {
        "version": 1,
        "defaults": {"seasons_back": 5, "refresh": "daily", "enabled": True},
        "competitions": [
            {
                "id": "a",
                "name": "A",
                "country": "X",
                "sport": "football",
                "tier": 1,
                "season_start_month": 8,
                "season_end_month": 5,
                "sources": {},
                "odds": {"the_odds_api": "sport_a"},
            },
            {
                "id": "b",
                "name": "B",
                "country": "Y",
                "sport": "football",
                "tier": 1,
                "season_start_month": 8,
                "season_end_month": 5,
                "enabled": False,  # sobreescribe el default
                "sources": {},
                "odds": {"the_odds_api": "sport_b"},
            },
        ],
    }
    catalog = CompetitionsCatalog.from_raw(raw)
    assert catalog.get("a").enabled is True
    assert catalog.get("b").enabled is False
    assert len(catalog.enabled) == 1


def test_duplicate_competition_ids_rejected() -> None:
    raw = {
        "version": 1,
        "defaults": {"seasons_back": 5, "refresh": "daily", "enabled": True},
        "competitions": [
            {
                "id": "dup",
                "name": "A",
                "country": "X",
                "sport": "football",
                "tier": 1,
                "season_start_month": 8,
                "season_end_month": 5,
                "sources": {},
                "odds": {"the_odds_api": "sport_a"},
            },
            {
                "id": "dup",
                "name": "B",
                "country": "Y",
                "sport": "football",
                "tier": 1,
                "season_start_month": 8,
                "season_end_month": 5,
                "sources": {},
                "odds": {"the_odds_api": "sport_b"},
            },
        ],
    }
    with pytest.raises(ValidationError, match="duplicados"):
        CompetitionsCatalog.from_raw(raw)


def test_unknown_competition_id_raises_keyerror() -> None:
    load_competitions.cache_clear()
    catalog = load_competitions()
    with pytest.raises(KeyError):
        catalog.get("does-not-exist")


# --------------------------------------------------------------------------
# markets.yaml: reglas de consistencia
# --------------------------------------------------------------------------


def test_market_needing_line_without_default_lines_rejected() -> None:
    raw = {
        "version": 1,
        "markets": [
            {
                "id": "over_under",
                "name": "OU",
                "sports": ["football"],
                "selections": ["over", "under"],
                "needs_line": True,
                "derived_from": "score_matrix",
                # falta default_lines
            }
        ],
        "bookmakers": {"sharp": ["pinnacle"], "soft": ["bet365"]},
    }
    with pytest.raises(ValidationError, match="default_lines"):
        MarketsCatalog.model_validate(raw)


def test_sharp_and_soft_bookmaker_overlap_rejected() -> None:
    raw = {
        "version": 1,
        "markets": [
            {
                "id": "1x2",
                "name": "1X2",
                "sports": ["football"],
                "selections": ["home", "draw", "away"],
                "needs_line": False,
                "derived_from": "score_matrix",
            }
        ],
        "bookmakers": {"sharp": ["pinnacle"], "soft": ["pinnacle"]},
    }
    with pytest.raises(ValidationError, match="sharp y soft"):
        MarketsCatalog.model_validate(raw)


# --------------------------------------------------------------------------
# thresholds.yaml: reglas de consistencia
# --------------------------------------------------------------------------


def test_tier_edges_must_decrease() -> None:
    base = yaml.safe_load((CONFIG_DIR / "thresholds.yaml").read_text(encoding="utf-8"))
    base["tiers"]["media"]["min_edge"] = 0.10  # ahora media > alta, invalido
    with pytest.raises(ValidationError, match="min_edge debe decrecer"):
        ThresholdsCatalog.model_validate(base)


def test_stake_bounds_must_be_ordered() -> None:
    base = yaml.safe_load((CONFIG_DIR / "thresholds.yaml").read_text(encoding="utf-8"))
    base["staking"]["min_stake_per_bet"] = base["staking"]["max_stake_per_bet"]
    with pytest.raises(ValidationError, match="min_stake_per_bet"):
        ThresholdsCatalog.model_validate(base)


def test_kelly_fraction_out_of_range_rejected() -> None:
    base = yaml.safe_load((CONFIG_DIR / "thresholds.yaml").read_text(encoding="utf-8"))
    base["staking"]["kelly_fraction"] = 1.5
    with pytest.raises(ValidationError):
        ThresholdsCatalog.model_validate(base)


def test_defaults_must_be_a_mapping() -> None:
    with pytest.raises(ValueError, match="'defaults' debe ser un mapeo"):
        CompetitionsCatalog.from_raw({"version": 1, "defaults": [], "competitions": []})


def test_competitions_must_be_a_list() -> None:
    with pytest.raises(ValueError, match="'competitions' debe ser una lista"):
        CompetitionsCatalog.from_raw({"version": 1, "defaults": {}, "competitions": {}})


def test_market_without_line_rejects_default_lines() -> None:
    raw = {
        "version": 1,
        "markets": [
            {
                "id": "1x2",
                "name": "1X2",
                "sports": ["football"],
                "selections": ["home", "draw", "away"],
                "needs_line": False,
                "default_lines": [1.5],  # inconsistente con needs_line=False
                "derived_from": "score_matrix",
            }
        ],
        "bookmakers": {"sharp": ["pinnacle"], "soft": ["bet365"]},
    }
    with pytest.raises(ValidationError, match="no deberia declarar default_lines"):
        MarketsCatalog.model_validate(raw)


def test_unknown_market_id_raises_keyerror() -> None:
    load_markets.cache_clear()
    catalog = load_markets()
    with pytest.raises(KeyError):
        catalog.get("does-not-exist")


def test_malformed_yaml_root_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "competitions.yaml"
    bad.write_text(textwrap.dedent("- just\n- a\n- list\n"), encoding="utf-8")
    load_competitions.cache_clear()
    with pytest.raises(ValueError, match="mapeo"):
        load_competitions(config_dir=tmp_path)
    load_competitions.cache_clear()
