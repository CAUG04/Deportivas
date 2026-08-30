// Lee el JSON pre-calculado que "deportivas export run" escribe bajo
// frontend/public/data/ (deportivas.export.json_export). No hay servidor
// detras de esto en produccion: cada archivo es una foto fija, no una
// consulta en vivo -- ver el docstring de Settings.export_dir.
import type { BacktestReport, Competition, Signal } from './types'

const DATA_BASE = `${import.meta.env.BASE_URL}data`

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`)
  }
  return (await response.json()) as T
}

export function fetchCompetitions(): Promise<Competition[]> {
  return fetchJson(`${DATA_BASE}/competitions.json`)
}

export function fetchSignals(competitionId: string): Promise<Signal[]> {
  return fetchJson(`${DATA_BASE}/${competitionId}/signals.json`)
}

export function fetchBacktestReport(competitionId: string): Promise<BacktestReport> {
  return fetchJson(`${DATA_BASE}/${competitionId}/backtest.json`)
}
