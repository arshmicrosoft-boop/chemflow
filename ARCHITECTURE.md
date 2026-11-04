# ChemFlow Prototype — Architecture

## Existing Engine (Preserved)
- `chemflow/agent.py`, `sandbox.py`, `memory.py`, `hypotheses.py`, `experiments.py`, `evaluation.py`, `learning.py`, `gaps.py`, `reporting.py`
- SQLite session memory with dataset partition tracking, budget enforcement, evidence thresholds, reproducibility seeds
- CLI: `python -m chemflow init/run/status/report/approvals/propose/approve/ingest`

## New Prototype Components

### Backend (`backend/app/`)
- `main.py`: FastAPI server with CORS
- `database.py`: SQLite `projects` and `settings` tables
- `routes.py`: REST endpoints for projects, status, run, hypotheses, experiments, timeline, results, report, knowledge, health
- Uses existing `ResearchAgent` directly (no mocked science)

### Frontend (`frontend/`)
- React + Vite + React Router + Tailwind CSS (compiled via PostCSS)
- Pages: Workspace (`/`) and Results (`/projects/:id/results`: verdict, model leaderboard, structure effects, findings, experiment log)
- Liquid glass styling lives in `src/index.css`; refraction uses an SVG backdrop filter on Chromium and falls back to blur elsewhere
- All values fetched from `/projects/*` endpoints

### Deployment
- `docker-compose.yml`: backend (8000) + frontend (3000)
- `docker/backend.Dockerfile`: Python 3.11 with backend requirements
- `docker/frontend.Dockerfile`: Node 20 build + Vite preview

## Notes
- The existing scientific logic remains the source of truth
- No simulated successes; experiment results come from `Sandbox.execute()`
- Holdout labels locked; never exposed to agent or frontend
