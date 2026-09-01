// Espejo de los modelos pydantic en src/deportivas/api/views.py. Si un
// campo cambia ahi, cambia aqui: no hay generacion automatica de tipos
// todavia (fuera de alcance del MVP de la Fase 7).

export interface Competition {
  id: string
  name: string
  country: string
  sport: string
  tier: number
}

export type Tier = 'alta' | 'media' | 'baja' | 'descartar'

export interface Signal {
  id: string
  fixture_id: string
  home_team: string
  away_team: string
  kickoff_utc: string
  market: string
  selection: string
  line: number | null
  model_name: string
  model_version: string
  prob_model: number
  prob_fair: number
  fair_price: number
  entry_price: number
  entry_bookmaker: string
  entry_captured_at: string
  edge: number
  tier: Tier
  tier_reasons: Record<string, boolean>
  stake_fraction: number
  created_at: string
}

export interface MetricSummary {
  n: number
  mean_clv: number | null
  clv_ci_low: number | null
  clv_ci_high: number | null
  mean_pnl: number
  roi: number | null
}

export interface BacktestReport {
  overall: MetricSummary
  by_tier: Record<string, MetricSummary>
  by_market: Record<string, MetricSummary>
  baselines: Record<string, MetricSummary>
}
