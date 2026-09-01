import { formatPercent } from '../format'
import type { MetricSummary } from '../types'

export function MetricSummaryCard({ label, summary }: { label: string; summary: MetricSummary }) {
  if (summary.n === 0) {
    return (
      <div className="rounded border border-slate-200 bg-white p-4">
        <p className="text-xs font-medium tracking-wide text-slate-500 uppercase">{label}</p>
        <p className="mt-1 text-sm text-slate-400">Sin liquidaciones todavía</p>
      </div>
    )
  }

  const ciText =
    summary.clv_ci_low !== null && summary.clv_ci_high !== null
      ? `IC [${formatPercent(summary.clv_ci_low)}, ${formatPercent(summary.clv_ci_high)}]`
      : 'IC no disponible (pocos datos)'

  return (
    <div className="rounded border border-slate-200 bg-white p-4">
      <p className="text-xs font-medium tracking-wide text-slate-500 uppercase">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-slate-800">
        {formatPercent(summary.mean_clv)}
      </p>
      <p className="text-xs text-slate-400">CLV medio · {ciText}</p>
      <p className="mt-2 text-sm text-slate-600">ROI {formatPercent(summary.roi)}</p>
      <p className="text-xs text-slate-400">n={summary.n}</p>
    </div>
  )
}
