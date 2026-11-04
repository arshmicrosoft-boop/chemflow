from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import io, os, json, re, time
import pandas as pd

router = APIRouter()

# Import existing chemflow modules directly
import sys
sys.path.insert(0, "/workspaces/project/chemflow")
from chemflow.agent import CONFIG_FILE, MEMORY_FILE, ResearchAgent
from chemflow.config import SessionConfig
from chemflow.data import valid_smiles
from chemflow.memory import ResearchMemory
from chemflow.reporting import build_report, build_results
from . import topic_research

DATASETS_DIR = "datasets"
UPLOADS_FOLDER = "uploads"
DATASET_SUFFIXES = (".csv", ".csv.gz")
DATASET_PREVIEW_ROWS = 2000
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_TOPIC_CHARS = 500

class ProjectCreate(BaseModel):
    name: str
    objective: str
    dataset: str = "datasets/delaney-processed.csv"
    question: Optional[str] = None
    smiles_column: str = "smiles"
    target_column: str = "measured log solubility in mols per litre"
    task: str = "regression"
    primary_metric: str = "rmse"
    success_threshold: float = 0.75

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

def _dataset_path(name: str) -> str:
    """Resolve a dataset name, keeping it inside the datasets folder."""
    root = os.path.abspath(DATASETS_DIR)
    path = os.path.abspath(os.path.join(root, name.replace("\\", "/").removeprefix("datasets/")))
    if not path.startswith(root + os.sep) or not path.endswith(DATASET_SUFFIXES):
        raise HTTPException(status_code=400, detail="Datasets must be CSV files in the datasets folder")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Dataset not found: %s" % name)
    return path

def _column_kind(values: pd.Series) -> str:
    numbers = pd.to_numeric(values.dropna(), errors="coerce")
    if len(numbers) == 0 or numbers.isna().any():
        return "text"
    return "binary" if set(numbers.unique()) <= {0, 1} else "numeric"

def _smiles_share(values: pd.Series) -> float:
    sample = values.dropna().astype(str).head(50)
    return sum(valid_smiles(s) for s in sample) / max(1, len(sample))

def _describe_dataset(name: str) -> Dict[str, Any]:
    df = pd.read_csv(_dataset_path(name), nrows=DATASET_PREVIEW_ROWS)
    columns = [{"name": c, "kind": _column_kind(df[c])} for c in df.columns]
    text = [c["name"] for c in columns if c["kind"] == "text"]
    shares = {c: _smiles_share(df[c]) for c in text}
    best = max(shares, key=shares.get, default=None)
    return {
        "name": name,
        "rows": len(df),
        "rows_capped": len(df) == DATASET_PREVIEW_ROWS,
        "columns": columns,
        "smiles_column": best if best and shares[best] >= 0.5 else None,
    }

def _check_objective(p: ProjectCreate, path: str) -> None:
    """Refuse settings that would leave a project that cannot run."""
    if not p.name.strip():
        raise HTTPException(status_code=400, detail="Give the project a name")
    columns = list(pd.read_csv(path, nrows=0).columns)
    for column in (p.smiles_column, p.target_column):
        if column not in columns:
            raise HTTPException(status_code=400, detail="Column %r is not in this dataset" % column)
    if p.smiles_column == p.target_column:
        raise HTTPException(status_code=400, detail="The molecule and target columns must be different")
    kind = _column_kind(pd.read_csv(path, usecols=[p.target_column])[p.target_column])
    if kind == "text":
        raise HTTPException(status_code=400, detail="Target column %r must contain numbers" % p.target_column)
    if p.task == "classification" and kind != "binary":
        raise HTTPException(status_code=400, detail="Classification needs a target column of only 0 and 1")

@router.get("/datasets")
async def list_datasets():
    names = []
    for folder, _, files in os.walk(DATASETS_DIR):
        for f in files:
            if f.endswith(DATASET_SUFFIXES):
                names.append(os.path.relpath(os.path.join(folder, f), DATASETS_DIR).replace(os.sep, "/"))
    return sorted(names)

@router.get("/datasets/info")
async def dataset_info(name: str):
    return _describe_dataset(name)

@router.post("/datasets")
async def upload_dataset(filename: str, request: Request):
    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Datasets can be at most 100 MB")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(filename.replace("\\", "/")))
    if not stem.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")
    try:
        pd.read_csv(io.BytesIO(body), nrows=5)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="This file could not be read as a CSV")
    folder = os.path.join(DATASETS_DIR, UPLOADS_FOLDER)
    os.makedirs(folder, exist_ok=True)
    base, n = stem[:-4], 1
    while os.path.exists(os.path.join(folder, stem)):
        n += 1
        stem = "%s_%d.csv" % (base, n)
    with open(os.path.join(folder, stem), "wb") as f:
        f.write(body)
    return _describe_dataset("%s/%s" % (UPLOADS_FOLDER, stem))

@router.post("/projects", response_model=ProjectResponse)
async def create_project(p: ProjectCreate):
    import sqlite3, shutil
    dataset_path = _dataset_path(p.dataset)
    _check_objective(p, dataset_path)
    dataset = os.path.relpath(dataset_path).replace(os.sep, "/")
    conn = sqlite3.connect("chemflow.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO projects (name, objective, dataset) VALUES (?, ?, ?)",
        (p.name, p.objective, dataset),
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
            "question": p.question or p.objective,
            "dataset": dataset,
            "smiles_column": p.smiles_column,
            "target_column": p.target_column,
            "task": p.task,
            "primary_metric": p.primary_metric,
            "success_threshold": p.success_threshold,
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
    agent, failure = None, None
    try:
        agent = ResearchAgent.create(session_dir, SessionConfig.from_json(tmp_cfg))
    except ValueError as e:
        failure = str(e)
    if failure:
        # Outside the except block: its traceback keeps the session's database
        # open, and Windows cannot delete an open file.
        conn = sqlite3.connect("chemflow.db")
        conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
        conn.commit()
        conn.close()
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=failure)
    agent.close()
    return ProjectResponse(id=pid, name=p.name, objective=p.objective, dataset=dataset, active=1)

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

class TopicRequest(BaseModel):
    topic: str

def _topic_report(row) -> Dict[str, Any]:
    report = dict(row)
    report["sources"] = json.loads(report["sources"])
    report["search_errors"] = json.loads(report["search_errors"])
    return report

@router.get("/topics/status")
async def topics_status():
    return {"configured": topic_research.api_key_configured(), "model": topic_research.MODEL}

@router.get("/topics")
async def list_topics():
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, topic, created_at FROM topic_reports ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@router.get("/topics/{tid}")
async def get_topic(tid: int):
    import sqlite3
    conn = sqlite3.connect("chemflow.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM topic_reports WHERE id = ?", (tid,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return _topic_report(row)

# A plain def: research takes minutes, and FastAPI runs these in a worker
# thread so the rest of the app keeps responding.
@router.post("/topics")
def create_topic(t: TopicRequest):
    import sqlite3
    topic = t.topic.strip()
    if not topic or len(topic) > MAX_TOPIC_CHARS:
        raise HTTPException(status_code=400, detail="Enter a topic of up to %d characters" % MAX_TOPIC_CHARS)
    try:
        result = topic_research.research(topic)
    except topic_research.ResearchError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    conn = sqlite3.connect("chemflow.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "INSERT INTO topic_reports (topic, report, sources, searches, search_errors, model) VALUES (?, ?, ?, ?, ?, ?)",
        (topic, result["report"], json.dumps(result["sources"]), result["searches"],
         json.dumps(result["search_errors"]), result["model"]),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM topic_reports WHERE id = ?", (cursor.lastrowid,)).fetchone()
    conn.close()
    return _topic_report(row)

@router.get("/explain")
async def explain():
    return {
        "what_is_this": "ChemFlow backend using the real Python scientific engine (agent.py, sandbox.py, memory.py, evaluation.py, learning.py)",
        "how_research_works": "Agent generates hypotheses -> selects experiment -> runs DeepChem model in sandbox -> evaluates -> updates memory -> repeats",
        "what_is_real": "Experiments use actual DeepChem featurizers and sklearn models. Results are real predictions, not mocked.",
        "budget_enforced": "Sandbox limits experiments (max_experiments_total=50), assay budget (100), runtime limits, and locks holdout labels",
        "endpoints": {
            "/topics": "AI research on a written topic, using web search (needs an API key)",
            "/datasets": "List CSV datasets, or upload one",
            "/datasets/info": "Columns of a dataset and which one holds the molecules",
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
