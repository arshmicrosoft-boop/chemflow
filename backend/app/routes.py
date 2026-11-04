from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, json, time

router = APIRouter()

# Import existing chemflow modules directly
import sys
sys.path.insert(0, "/workspaces/project/chemflow")
from chemflow.agent import CONFIG_FILE, MEMORY_FILE, ResearchAgent
from chemflow.config import SessionConfig
from chemflow.memory import ResearchMemory
from chemflow.reporting import build_report, build_results

class ProjectCreate(BaseModel):
    name: str
    objective: str
    dataset: str = "datasets/delaney-processed.csv"

class ProjectResponse(BaseModel):
    id: int
    name: str
    objective: str
    dataset: str
    active: int

# In-memory session tracking (prototype)
_sessions: Dict[int, ResearchAgent] = {}

@router.get("/health")
async def health():
    return {"status": "ok", "service": "chemflow-api"}

@router.post("/projects", response_model=ProjectResponse)
async def create_project(p: ProjectCreate):
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO projects (name, objective, dataset) VALUES (?, ?, ?)",
        (p.name, p.objective, p.dataset),
    )
    pid = cursor.lastrowid
    conn.commit()
    session_dir = f"sessions/project_{pid}"
    os.makedirs(session_dir, exist_ok=True)
    cursor.execute(
        "UPDATE projects SET session_dir = ? WHERE id = ?", (session_dir, pid)
    )
    conn.commit()
    conn.close()
    # Initialize session using existing config
    config_path = "examples/chemflow/solubility.json"
    if not os.path.exists(config_path):
        config_path = "chemflow/config.json"
    # Build a basic session config for prototype
    cfg = {
        "objective": {
            "name": p.objective,
            "question": "Predict aqueous solubility from molecular structure",
            "dataset": p.dataset,
            "smiles_column": "smiles",
            "target_column": "measured log solubility in mols per litre",
            "task": "regression",
            "primary_metric": "rmse",
            "success_threshold": 0.75,
            "desirable_direction": "high",
        },
        "sandbox": {
            "seed": 42,
            "max_cycles_per_run": 10,
            "max_experiments_total": 50,
            "assay_budget": 100,
        },
    }
    import tempfile, json
    tmp_cfg = tempfile.mktemp(suffix=".json", dir=session_dir)
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f)
    agent = ResearchAgent.create(session_dir, SessionConfig.from_json(tmp_cfg))
    agent.close()
    return ProjectResponse(id=pid, name=p.name, objective=p.objective, dataset=p.dataset, active=1)

@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects():
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM projects").fetchall()
    conn.close()
    return [ProjectResponse(**dict(r)) for r in rows]

@router.get("/projects/{pid}")
async def get_project(pid: int):
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return dict(row)

@router.get("/projects/{pid}/status")
async def project_status(pid: int):
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
    session_dir = row["session_dir"]
    if session_dir and os.path.exists(os.path.join(session_dir, "config.json")):
        agent = ResearchAgent.resume(session_dir)
        status = agent.status()
        agent.close()
        return status
    return {"cycle": 0, "experiments": 0, "objective": row["objective"]}

@router.post("/projects/{pid}/run")
async def run_project(pid: int, cycles: Optional[int] = 5):
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
    session_dir = row["session_dir"]
    agent = ResearchAgent.resume(session_dir)
    reason = agent.run(max_cycles=cycles)
    agent.close()
    return {"pid": pid, "reason": reason, "message": "Run completed"}

@router.get("/projects/{pid}/hypotheses")
async def get_hypotheses(pid: int):
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    row = conn.execute("SELECT session_dir FROM projects WHERE id = ?", (pid,)).fetchone()
    conn.close()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="No session")
    agent = ResearchAgent.resume(row[0])
    hyps = agent.memory.hypotheses()
    agent.close()
    return hyps

@router.get("/projects/{pid}/experiments")
async def get_experiments(pid: int):
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    row = conn.execute("SELECT session_dir FROM projects WHERE id = ?", (pid,)).fetchone()
    conn.close()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="No session")
    agent = ResearchAgent.resume(row[0])
    exps = agent.memory.experiments()
    agent.close()
    return exps

@router.get("/projects/{pid}/timeline")
async def get_timeline(pid: int):
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    row = conn.execute("SELECT session_dir FROM projects WHERE id = ?", (pid,)).fetchone()
    conn.close()
    if not row or not row[0]:
        return []
    agent = ResearchAgent.resume(row[0])
    events = agent.memory.events()
    agent.close()
    return events

def _open_session(pid: int):
    """Read-only view of a session: unlike ResearchAgent.resume this logs no
    event and does not load the dataset."""
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    row = conn.execute("SELECT session_dir FROM projects WHERE id = ?", (pid,)).fetchone()
    conn.close()
    if not row or not row[0] or not os.path.exists(os.path.join(row[0], CONFIG_FILE)):
        raise HTTPException(status_code=404, detail="No session")
    config = SessionConfig.from_json(os.path.join(row[0], CONFIG_FILE))
    return config, ResearchMemory(os.path.join(row[0], MEMORY_FILE))

@router.get("/projects/{pid}/results")
async def get_results(pid: int):
    config, memory = _open_session(pid)
    results = build_results(config, memory)
    memory.close()
    return results

@router.get("/projects/{pid}/report", response_class=PlainTextResponse)
async def get_report(pid: int):
    config, memory = _open_session(pid)
    report = build_report(config, memory)
    memory.close()
    return report

@router.get("/explain")
async def explain():
    return {
        "what_is_this": "ChemFlow backend using the real Python scientific engine (agent.py, sandbox.py, memory.py, evaluation.py, learning.py)",
        "how_research_works": "Agent generates hypotheses -> selects experiment -> runs DeepChem model in sandbox -> evaluates -> updates memory -> repeats",
        "what_is_real": "Experiments use actual DeepChem featurizers and sklearn models. Results are real predictions, not mocked.",
        "budget_enforced": "Sandbox limits experiments (max_experiments_total=50), assay budget (100), runtime limits, and locks holdout labels",
        "endpoints": {
            "/projects": "Create and list research projects",
            "/projects/{id}/status": "Current agent state (cycle, experiments, champion model, budget remaining)",
            "/projects/{id}/run": "Run autonomous research cycles with real model execution",
            "/projects/{id}/hypotheses": "Real generated hypotheses from memory",
            "/projects/{id}/experiments": "Every experiment including blocked/failed/completed",
            "/projects/{id}/timeline": "Event log from SQLite memory",
            "/projects/{id}/results": "Structured results: verdict, model leaderboard, structure effects, findings, next steps",
            "/projects/{id}/report": "Full Markdown research report",
            "/knowledge": "Findings and events",
        }
    }

@router.get("/knowledge")
async def get_knowledge(pid: Optional[int] = None):
    if pid is not None:
        import sqlite3
        conn = sqlite3.connect("chemflow.db")
        row = conn.execute("SELECT session_dir FROM projects WHERE id = ?", (pid,)).fetchone()
        conn.close()
        if row and row[0]:
            agent = ResearchAgent.resume(row[0])
            findings = agent.memory.findings()
            events = agent.memory.events()
            agent.close()
            return {"findings": findings, "events": events}
    return {"findings": [], "events": []}
