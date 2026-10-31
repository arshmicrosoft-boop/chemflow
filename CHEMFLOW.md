# ChemFlow

ChemFlow is a research layer built on DeepChem. DeepChem predicts scientific
outcomes. ChemFlow adds an agent that keeps investigating a scientific
question over many cycles. It works inside a sandbox, so it can explore freely
but cannot claim anything the evidence does not support.

The code is in [`chemflow/`](chemflow/). DeepChem (`deepchem/`) is unchanged:
ChemFlow uses its featurizers, models, datasets and scaffold splitter.

## Architecture

| Component | Module | Role |
|---|---|---|
| Research Agent | `agent.py` | Runs the research loop and keeps sessions going across runs |
| Scientific Sandbox | `sandbox.py` | Enforces rules, limits, data boundaries and human approval |
| Persistent Memory | `memory.py` | SQLite store of hypotheses, experiments, findings, failures, data and approvals |
| Hypothesis Generation | `hypotheses.py` | Model, structure-activity (SAR) and data-acquisition hypotheses built from evidence |
| Experiment Engine | `experiments.py` | Computational experiments that run DeepChem models |
| Experiment Selection | `selection.py` | Picks the next experiment by expected value (expected improvement, exploration, budget) |
| Evaluation Engine | `evaluation.py` | Checks hypotheses against evidence, and checks the agent's own predictions |
| Learning Engine | `learning.py` | Champion model, retrained on validated data, with versions and regression guard |
| Knowledge-Gap Detection | `gaps.py` | Open questions, inconclusive results, under-represented chemistry, error hotspots, uncertainty |
| Research Reporting | `reporting.py` | Markdown report of findings, evidence, failures and limitations |

## The learning cycle

```
objective -> generate hypotheses -> detect gaps -> select experiment
  -> run DeepChem models (in sandbox) -> evaluate -> learn from validated data
  -> update memory -> choose next experiment -> repeat
```

## Sandbox rules

The table shows how each rule is enforced.

| Rule | Enforcement |
|---|---|
| Evidence requirements | A claim needs at least `min_supporting_experiments` supporting experiments |
| Dataset boundaries | The data is split into labeled, hidden pool, validation and locked hold-out sets. Pool labels are revealed only through budgeted virtual assays |
| Confidence thresholds | The pessimistic bootstrap confidence-interval bound must meet the success threshold. SAR tests use Bonferroni-corrected significance and a minimum effect size |
| Reproducibility | Every experiment is seeded and records its spec, data fingerprint and package versions. Model claims need a replicate run with a different seed to agree within tolerance |
| Experiment logging | Every action is written to memory, including blocked, failed and timed-out experiments |
| Compute/runtime limits | Limits per experiment, per run and on total experiments |
| Independent validation | A claim is validated only after a hold-out test. Hold-out access is counted and capped |
| Human approval | Lab experiments are never executed. They are queued as approval requests, and only results for approved molecules can be ingested. This rule cannot be switched off |

Each finding ends in one of four states:

- **validated**: every requirement above is met.
- **provisional**: the evidence supports it but is incomplete.
- **inconclusive**: the sandbox cannot get more evidence for it.
- **rejected**: the evidence contradicts it.

## Usage

```bash
pip install numpy scipy pandas scikit-learn joblib rdkit
python -m chemflow init sessions/solubility --config examples/chemflow/solubility.json
python -m chemflow run sessions/solubility --cycles 40     # run again later to continue
python -m chemflow status sessions/solubility
python -m chemflow report sessions/solubility             # writes report.md
python -m chemflow approvals sessions/solubility
python -m chemflow approve sessions/solubility <A-id> --by "Your Name"
python -m chemflow ingest sessions/solubility <A-id> lab_results.csv
```

Python API:

```python
from chemflow import ResearchAgent, SessionConfig
agent = ResearchAgent.create("sessions/s1", SessionConfig.from_json("examples/chemflow/solubility.json"))
agent.run(max_cycles=20)
agent = ResearchAgent.resume("sessions/s1")   # later, in another process
```

## Limitations

- Experiments are computational. In benchmark mode, virtual assays reveal
  measurements held back from the dataset.
- SAR findings are associations, not mechanisms.
- Models are scikit-learn estimators wrapped as `dc.models.SklearnModel`, so
  ChemFlow runs without PyTorch. The MLP family is the neural network. To add
  DeepChem Torch models, extend `chemflow/data.py:build_model`.
- Time limits are soft. A slow experiment is abandoned, but Python cannot
  interrupt it, so it keeps running in the background until it finishes.
