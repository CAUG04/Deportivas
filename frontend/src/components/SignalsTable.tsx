import { useMemo, useState } from 'react'
import { formatKickoff, formatOdds, formatPercent } from '../format'
import type { Signal, Tier } from '../types'
import { TierBadge } from './TierBadge'

const TIER_ORDER: Tier[] = ['alta', 'media', 'baja', 'descartar']
const ALL = 'todas'

interface Props {
  signals: Signal[]
}

export function SignalsTable({ signals }: Props) {
  const [tierFilter, setTierFilter] = useState<Tier | typeof ALL>(ALL)

  const filtered = useMemo(
    () => (tierFilter === ALL ? signals : signals.filter((s) => s.tier === tierFilter)),
    [signals, tierFilter],
  )

  if (signals.length === 0) {
    return <p className="text-sm text-slate-500">No hay señales para esta competición todavía.</p>
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-2">
        {([ALL, ...TIER_ORDER] as const).map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => setTierFilter(option)}
            className={`rounded-full border px-3 py-1 text-xs font-medium capitalize ${
              tierFilter === option
                ? 'border-slate-800 bg-slate-800 text-white'
                : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'
            }`}
          >
            {option}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto rounded border border-slate-200">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-left text-xs font-medium tracking-wide text-slate-500 uppercase">
            <tr>
              <th className="px-3 py-2">Partido</th>
              <th className="px-3 py-2">Mercado</th>
              <th className="px-3 py-2">Selección</th>
              <th className="px-3 py-2 text-right">Cuota entrada</th>
              <th className="px-3 py-2 text-right">Edge</th>
              <th className="px-3 py-2 text-right">Stake</th>
              <th className="px-3 py-2">Tier</th>
              <th className="px-3 py-2">Kickoff</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {filtered.map((signal) => (
              <tr key={signal.id} className="hover:bg-slate-50">
                <td className="px-3 py-2 font-medium text-slate-800">
                  {signal.home_team} vs {signal.away_team}
                </td>
                <td className="px-3 py-2 text-slate-600">
                  {signal.market}
                  {signal.line !== null ? ` (${signal.line})` : ''}
                </td>
                <td className="px-3 py-2 text-slate-600 capitalize">{signal.selection}</td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {formatOdds(signal.entry_price)}
                  <span className="ml-1 text-xs text-slate-400">{signal.entry_bookmaker}</span>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {formatPercent(signal.edge)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {formatPercent(signal.stake_fraction)}
                </td>
                <td className="px-3 py-2">
                  <TierBadge tier={signal.tier} />
                </td>
                <td className="px-3 py-2 text-slate-500">{formatKickoff(signal.kickoff_utc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
