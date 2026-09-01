import { formatPercent } from '../format'
import type { BacktestReport, MetricSummary } from '../types'
import { MetricSummaryCard } from './MetricSummaryCard'

function BreakdownTable({ title, rows }: { title: string; rows: [string, MetricSummary][] }) {
  if (rows.length === 0) return null
  return (
    <div>
      <h3 className="mb-2 text-sm font-medium text-slate-700">{title}</h3>
      <table className="min-w-full divide-y divide-slate-200 text-sm">
        <thead className="bg-slate-50 text-left text-xs font-medium tracking-wide text-slate-500 uppercase">
          <tr>
            <th className="px-3 py-2">Grupo</th>
            <th className="px-3 py-2 text-right">n</th>
            <th className="px-3 py-2 text-right">CLV medio</th>
            <th className="px-3 py-2 text-right">ROI</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map(([key, summary]) => (
            <tr key={key}>
              <td className="px-3 py-2 font-medium text-slate-800 capitalize">{key}</td>
              <td className="px-3 py-2 text-right tabular-nums">{summary.n}</td>
              <td className="px-3 py-2 text-right tabular-nums">
                {formatPercent(summary.mean_clv)}
              </td>
              <td className="px-3 py-2 text-right tabular-nums">{formatPercent(summary.roi)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function BacktestSummary({ report }: { report: BacktestReport }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <MetricSummaryCard label="Estrategia real (global)" summary={report.overall} />
        {Object.entries(report.baselines).map(([name, summary]) => (
          <MetricSummaryCard key={name} label={`Baseline: ${name}`} summary={summary} />
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <BreakdownTable title="Por tier" rows={Object.entries(report.by_tier)} />
        <BreakdownTable title="Por mercado" rows={Object.entries(report.by_market)} />
      </div>
    </div>
  )
}
