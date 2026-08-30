import { type ReactNode, useState } from 'react'
import { fetchBacktestReport, fetchCompetitions, fetchSignals } from './api'
import { BacktestSummary } from './components/BacktestSummary'
import { CompetitionSelector } from './components/CompetitionSelector'
import { SignalsTable } from './components/SignalsTable'
import { useFetch } from './useFetch'

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="mb-3 text-lg font-semibold text-slate-800">{title}</h2>
      {children}
    </section>
  )
}

export default function App() {
  const { data: competitions, error: competitionsError } = useFetch(fetchCompetitions, [])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  // Ninguna competicion elegida todavia por el usuario: cae a la primera de
  // la lista una vez cargada, derivado en el render en vez de sincronizado
  // con un efecto -- no hay estado externo que sincronizar, solo un default.
  const effectiveId = selectedId ?? competitions?.[0]?.id ?? null

  const signalsFetch = useFetch(
    () => (effectiveId !== null ? fetchSignals(effectiveId) : Promise.resolve([])),
    [effectiveId],
  )
  const backtestFetch = useFetch(
    () =>
      effectiveId !== null
        ? fetchBacktestReport(effectiveId)
        : Promise.reject(new Error('sin competición seleccionada')),
    [effectiveId],
  )

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Deportivas</h1>
          <p className="text-sm text-slate-500">
            Señales y backtest generados por el pipeline — datos pre-calculados, sin servidor.
          </p>
        </div>
        {competitions !== null && competitions.length > 0 && (
          <CompetitionSelector
            competitions={competitions}
            selectedId={effectiveId}
            onSelect={setSelectedId}
          />
        )}
      </header>

      {competitionsError !== null && (
        <p className="rounded border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          No se pudo cargar la lista de competiciones ({competitionsError}). ¿Corriste{' '}
          <code>deportivas export run</code>?
        </p>
      )}

      {competitions !== null && competitions.length === 0 && (
        <p className="text-sm text-slate-500">
          No hay competiciones habilitadas en <code>config/competitions.yaml</code>.
        </p>
      )}

      {effectiveId !== null && (
        <>
          <Section title="Señales">
            {signalsFetch.loading && <p className="text-sm text-slate-400">Cargando…</p>}
            {signalsFetch.error !== null && (
              <p className="rounded border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
                No se pudo cargar señales para esta competición ({signalsFetch.error}).
              </p>
            )}
            {signalsFetch.data !== null && <SignalsTable signals={signalsFetch.data} />}
          </Section>

          <Section title="Backtest">
            {backtestFetch.loading && <p className="text-sm text-slate-400">Cargando…</p>}
            {backtestFetch.error !== null && (
              <p className="rounded border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
                No se pudo cargar el reporte de backtest ({backtestFetch.error}).
              </p>
            )}
            {backtestFetch.data !== null && <BacktestSummary report={backtestFetch.data} />}
          </Section>
        </>
      )}
    </div>
  )
}
