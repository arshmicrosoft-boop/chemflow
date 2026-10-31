"""Tests for the ChemFlow research layer.

They run a few real research cycles on a subset of the Delaney solubility
data bundled with DeepChem, so they need RDKit and scikit-learn.
"""
import os

import pandas as pd
import pytest

from chemflow import (ResearchAgent, ResearchObjective, SandboxPolicy,
                      SandboxViolation, SessionConfig)
from chemflow.memory import ResearchMemory

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DELANEY = os.path.join(REPO, "datasets", "delaney-processed.csv")
TARGET = "measured log solubility in mols per litre"


@pytest.fixture
def config(tmp_path):
    df = pd.read_csv(DELANEY).sample(300, random_state=0)
    data = tmp_path / "data.csv"
    df.to_csv(data, index=False)
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1"]
                 }).to_csv(candidates, index=False)
    return SessionConfig(
        objective=ResearchObjective(name="test",
                                    question="Is solubility predictable?",
                                    dataset=str(data),
                                    smiles_column="smiles",
                                    target_column=TARGET,
                                    success_threshold=0.5,
                                    candidates=str(candidates)),
        sandbox=SandboxPolicy(allowed_featurizers=["maccs"],
                              allowed_models=["ridge", "random_forest"],
                              assay_budget=30,
                              assay_batch_size=10,
                              max_holdout_evaluations=3,
                              max_cycles_per_run=6))


def make_agent(tmp_path, config, name="session"):
    return ResearchAgent.create(str(tmp_path / name),
                                config,
                                log=lambda *_: None)


def test_human_approval_cannot_be_disabled():
    with pytest.raises(ValueError):
        SandboxPolicy(require_human_approval_for_lab=False).validate()


def test_dataset_boundaries(tmp_path, config):
    agent = make_agent(tmp_path, config)
    part = {k: set(v) for k, v in agent.vault.partition.items()}
    names = list(part)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not part[a] & part[b], "%s overlaps %s" % (a, b)
    train = set(agent.vault.training_indices())
    assert train == part["labeled"]
    # Pool labels only enter training through a budgeted assay.
    pool_smiles = [agent.vault.smiles[i] for i in sorted(part["pool"])[:5]]
    agent.sandbox.assay(pool_smiles, None)
    assert len(agent.vault.training_indices()) == len(part["labeled"]) + 5
    assert agent.sandbox.assay_remaining() == 25
    with pytest.raises(SandboxViolation):
        agent.sandbox.assay([agent.vault.smiles[next(iter(part["holdout"]))]],
                            None)
    agent.close()


def test_sandbox_limits(tmp_path, config):
    agent = make_agent(tmp_path, config)
    with pytest.raises(SandboxViolation):
        agent.sandbox.check("train_evaluate", {"model": "mlp"})
    with pytest.raises(SandboxViolation):
        agent.sandbox.check("train_evaluate", {"featurizer": "ecfp"})
    with pytest.raises(SandboxViolation):
        agent.sandbox.check("delete_data", {})
    for _ in range(3):
        agent.sandbox.holdout("maccs", "test")
    with pytest.raises(SandboxViolation):
        agent.sandbox.holdout("maccs", "test")
    agent.close()


def test_certify_requires_all_evidence(tmp_path, config):
    agent = make_agent(tmp_path, config)
    full = {
        "supporting_experiments": ["E-1", "E-2"],
        "meets_confidence": True,
        "reproducible": True,
        "independent_validation": True,
    }
    assert agent.sandbox.certify(full)[0] == "validated"
    assert agent.sandbox.certify(dict(
        full, independent_validation=None))[0] == "provisional"
    assert agent.sandbox.certify(dict(
        full, supporting_experiments=["E-1"]))[0] == "provisional"
    assert agent.sandbox.certify(dict(full,
                                      reproducible=False))[0] == "rejected"
    agent.close()


def test_research_loop_and_resume(tmp_path, config):
    agent = make_agent(tmp_path, config)
    agent.run(max_cycles=4)
    assert agent.memory.cycle == 4
    n_experiments = len(agent.memory.experiments())
    assert n_experiments == 4
    assert agent.learning.champion is not None
    assert os.path.exists(os.path.join(agent.session_dir, "report.md"))
    session_dir = agent.session_dir
    agent.close()

    resumed = ResearchAgent.resume(session_dir, log=lambda *_: None)
    assert resumed.memory.cycle == 4
    assert len(resumed.memory.experiments()) == n_experiments
    resumed.run(max_cycles=3)
    assert resumed.memory.cycle == 7
    assert len(resumed.memory.runs()) == 2
    statuses = {h["status"] for h in resumed.memory.hypotheses()}
    assert statuses - {"proposed"}, "hypotheses should have been assessed"
    resumed.close()


def test_lab_experiments_need_approval(tmp_path, config):
    agent = make_agent(tmp_path, config)
    agent.run(max_cycles=2)
    pending = agent.memory.approvals("pending")
    assert len(pending) == 1
    aid = pending[0]["id"]
    results = pd.DataFrame({
        "smiles": ["CCO", "CCCCCCCC"],
        "value": [1.1, -5.0]
    })
    with pytest.raises(SandboxViolation):
        agent.ingest_lab_results(aid, results)
    agent.memory.decide_approval(aid, True, "tester")
    out = agent.ingest_lab_results(aid, results)
    # Only molecules named in the approved request are accepted.
    assert out["accepted"] == 1 and out["rejected"] == 1
    assert agent.memory.approval(aid)["status"] == "completed"
    assert len(agent.vault.training_set("maccs")) == \
        len(agent.vault.partition["labeled"]) + 1
    agent.close()


def test_memory_records_failures(tmp_path):
    memory = ResearchMemory(str(tmp_path / "m.db"))
    hid = memory.add_hypothesis("model", "x", {"a": 1}, "test", 0.5)
    eid = memory.add_experiment(hid, "train_evaluate", {}, 0)
    memory.update_experiment(eid, status="failed", error="boom")
    assert memory.experiment(eid)["error"] == "boom"
    assert memory.has_hypothesis("model", {"a": 1})
    memory.close()
