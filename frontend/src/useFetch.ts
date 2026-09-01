import { useEffect, useState } from 'react'

interface FetchState<T> {
  data: T | null
  error: string | null
  loading: boolean
}

/**
 * Runs `fetcher` whenever `deps` changes, tracking loading/error/data state
 * and ignoring a stale response that resolves after a newer request already
 * started (e.g. switching competitions quickly).
 */
export function useFetch<T>(fetcher: () => Promise<T>, deps: unknown[]): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({ data: null, error: null, loading: true })

  // deps es deliberadamente dinamico: cada llamador declara sus propias
  // dependencias (p.ej. [competitionId]), no las de este hook generico.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    let cancelled = false
    setState({ data: null, error: null, loading: true })
    fetcher()
      .then((data) => {
        if (!cancelled) setState({ data, error: null, loading: false })
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : String(err)
          setState({ data: null, error: message, loading: false })
        }
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}
