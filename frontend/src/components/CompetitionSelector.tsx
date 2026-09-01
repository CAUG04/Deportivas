import type { Competition } from '../types'

interface Props {
  competitions: Competition[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export function CompetitionSelector({ competitions, selectedId, onSelect }: Props) {
  return (
    <label className="flex items-center gap-2 text-sm text-slate-700">
      <span className="font-medium">Competición</span>
      <select
        className="rounded border border-slate-300 bg-white px-2 py-1"
        value={selectedId ?? ''}
        onChange={(event) => onSelect(event.target.value)}
      >
        {competitions.map((competition) => (
          <option key={competition.id} value={competition.id}>
            {competition.name} ({competition.country})
          </option>
        ))}
      </select>
    </label>
  )
}
