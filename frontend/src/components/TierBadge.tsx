import type { Tier } from '../types'

const STYLES: Record<Tier, string> = {
  alta: 'bg-emerald-100 text-emerald-800',
  media: 'bg-amber-100 text-amber-800',
  baja: 'bg-slate-100 text-slate-600',
  descartar: 'bg-rose-50 text-rose-500',
}

export function TierBadge({ tier }: { tier: Tier }) {
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium capitalize ${STYLES[tier]}`}
    >
      {tier}
    </span>
  )
}
