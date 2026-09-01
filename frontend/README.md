# Deportivas — frontend

Sitio estático (Vite + React 19 + TypeScript + Tailwind v4) que lee el JSON
pre-calculado bajo `public/data/` — nunca llama a un servidor. Ver la raíz
del repo para el resto: [`../README.md`](../README.md)'s
["Fase 7 — Frontend"](../README.md#estado-del-proyecto) y
["Alcance de la Fase 7"](../README.md#alcance-de-la-fase-7).

## Desarrollo local

```bash
# desde la raiz del repo: puebla public/data/ con datos reales
uv run deportivas export run

cd frontend
npm install
npm run dev       # http://localhost:5173
```

## Comandos

- `npm run dev` — servidor de desarrollo con recarga en caliente.
- `npm run build` — `tsc -b` (typecheck estricto) + `vite build` a `dist/`.
- `npm run lint` — `oxlint`.
- `npm run preview` — sirve `dist/` localmente, como en producción.
