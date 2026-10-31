"""
The Research Agent: plans and coordinates autonomous research.

Each research cycle follows the ChemFlow learning loop::

    research objective
      -> generate hypotheses        (HypothesisGenerator)
      -> detect knowledge gaps      (KnowledgeGapDetector)
      -> select experiment          (ExperimentSelector)
      -> run DeepChem models        (ExperimentEngine inside the Sandbox)
      -> evaluate results           (EvaluationEngine)
      -> learn from validated data  (LearningEngine)
      -> update research memory     (ResearchMemory)
      -> choose next experiment ... repeat

A session lives in a directory (config, SQLite memory, feature cache, model
versions, report), so research can be stopped and resumed across processes
without losing progress.
"""
import os
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from chemflow.config import SessionConfig
from chemflow.evaluation import EvaluationEngine
from chemflow.experiments import ExperimentEngine
from chemflow.gaps import KnowledgeGapDetector
from chemflow.hypotheses import HypothesisGenerator
from chemflow.learning import LearningEngine
from chemflow.memory import ResearchMemory
from chemflow.reporting import build_report
from chemflow.sandbox import DataVault, Sandbox, SandboxViolation
from chemflow.selection import Action, ExperimentSelector

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.db"
REPORT_FILE = "report.md"


class ResearchAgent:

    def __init__(self, session_dir: str, config: SessionConfig,
                 log: Callable[[str], None] = print):
        config.validate()
        self.session_dir = session_dir
        self.config = config
        self.objective = config.objective
        self.policy = config.sandbox
        self.print = log
        self.memory = ResearchMemory(os.path.join(session_dir, MEMORY_FILE))
        self.vault = DataVault(self.objective, self.policy, self.memory,
                               session_dir)
        self.sandbox = Sandbox(self.objective, self.policy, self.memory,
                               self.vault)
        self.learning = LearningEngine(self.objective, self.policy,
                                       self.memory, self.vault, session_dir)
        self.engine = ExperimentEngine(self.objective, self.policy,
                                       self.sandbox, self.vault, self.learning)
        self.evaluation = EvaluationEngine(self.objective, self.policy,
                                           self.memory, self.sandbox)
        self.generator = HypothesisGenerator(self.objective, self.policy,
                                             self.memory, self.vault)
        self.selector = ExperimentSelector(self.objective, self.policy,
                                           self.memory, self.sandbox,
                                           self.evaluation, self.learning)
        self.gaps = KnowledgeGapDetector(self.objective, self.policy,
                                         self.memory, self.vault, self.sandbox,
                                         self.learning)

    # -- sessions --------------------------------------------------------
    @classmethod
    def create(cls, session_dir: str, config: SessionConfig,
               **kwargs) -> "ResearchAgent":
        """Start a new research session in an empty directory."""
        if os.path.exists(os.path.join(session_dir, CONFIG_FILE)):
            raise FileExistsError("%s already holds a session; use "
                                  "ResearchAgent.resume" % session_dir)
        config.validate()
        # Store absolute data paths so the session resumes from any cwd.
        obj = config.objective
        obj.dataset = os.path.abspath(obj.dataset)
        if obj.candidates:
            obj.candidates = os.path.abspath(obj.candidates)
        os.makedirs(session_dir, exist_ok=True)
        config.save(os.path.join(session_dir, CONFIG_FILE))
        agent = cls(session_dir, config, **kwargs)
        agent.memory.log("session_created", objective=config.objective.name)
        return agent

    @classmethod
    def resume(cls, session_dir: str, **kwargs) -> "ResearchAgent":
        """Reopen an existing session with all of its memory."""
        path = os.path.join(session_dir, CONFIG_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError("No ChemFlow session in %s" % session_dir)
        agent = cls(session_dir, SessionConfig.from_json(path), **kwargs)
        agent.memory.log("session_resumed", cycle=agent.memory.cycle)
        return agent

    def close(self) -> None:
        self.memory.close()

    # -- the research loop ----------------------------------------------
    def run(self,
            max_cycles: Optional[int] = None,
            max_runtime_seconds: Optional[float] = None) -> str:
        """Run research cycles until a limit is hit or nothing is left to
        learn. Returns the stop reason. Safe to call again later to
        continue the same session."""
        max_cycles = max_cycles or self.policy.max_cycles_per_run
        runtime = min(max_runtime_seconds or self.policy.max_runtime_seconds,
                      self.policy.max_runtime_seconds)
        self.sandbox.run_started = time.time()
        run_id = self.memory.start_run()
        self.print("ChemFlow: %s (cycle %d, %d experiments so far)" %
                   (self.objective.name, self.memory.cycle,
                    len(self.memory.experiments())))
        cycles, reason = 0, "cycle limit reached"
        try:
            while cycles < max_cycles:
                if self.sandbox.runtime_used() >= runtime:
                    reason = "runtime limit reached"
                    break
                if self.sandbox.experiments_used() >= \
                        self.policy.max_experiments_total:
                    reason = "experiment budget exhausted"
                    break
                if not self.step():
                    reason = "no informative experiments left"
                    break
                cycles += 1
        except KeyboardInterrupt:
            reason = "interrupted"
        finally:
            self.memory.end_run(run_id, cycles, reason)
            self.memory.log("run_ended", cycles=cycles, reason=reason)
        if self.objective.candidates and self.learning.champion and \
                not self.memory.approvals():
            try:
                self.propose_lab_experiments()
            except (SandboxViolation, RuntimeError) as e:
                self.memory.log("lab_proposal_skipped", reason=str(e))
        path = self.write_report()
        self.print("Stopped: %s. Report: %s" % (reason, path))
        return reason

    def step(self) -> bool:
        """One research cycle. Returns False when nothing is left to do."""
        cycle = self.memory.cycle + 1
        self.memory.set_state("cycle", cycle)

        self.generator.generate()
        self._act_on_gaps(self.gaps.detect())
        action = self.selector.select()
        if action is None:
            self.memory.set_state("cycle", cycle - 1)
            return False
        self.memory.log("experiment_selected",
                        kind=action.kind,
                        hypothesis_id=action.hypothesis_id,
                        utility=action.utility,
                        rationale=action.rationale)
        self._execute(action, cycle)
        return True

    def _execute(self, action: Action, cycle: int) -> None:
        h = self.memory.hypothesis(action.hypothesis_id)
        fn = self._experiment_fn(action, h)
        try:
            record = self.sandbox.execute(h["id"], action.kind,
                                          dict(action.spec, **h["params"]), fn,
                                          action.expected)
        except SandboxViolation as e:
            self.memory.set_hypothesis_status(h["id"], "abandoned")
            self.print("[%3d] %-18s blocked: %s" % (cycle, action.kind, e))
            return
        self.evaluation.record_surprise(record)
        if record["status"] == "completed" and action.kind in (
                "train_evaluate", "replicate"):
            self.learning.consider(h, record,
                                   self.engine.fitted.get(record["id"]))
        self.engine.fitted.clear()
        status = self.evaluation.assess(h)
        self.print("[%3d] %-18s %-11s %s -> %s" %
                   (cycle, action.kind, record["status"],
                    _short(h["statement"]), status or
                    self.memory.hypothesis(h["id"])["status"]))

    def _experiment_fn(self, action: Action, h: Dict[str, Any]):
        if action.kind == "train_evaluate":
            return self.engine.train_evaluate(h)
        if action.kind == "replicate":
            return self.engine.replicate(h)
        if action.kind == "holdout_validation":
            return self.engine.holdout_validation(h)
        if action.kind == "sar_test":
            return self.engine.sar_test(h, action.spec["split"])
        if action.kind == "virtual_assay":
            return self.engine.virtual_assay(h)
        raise ValueError("unknown experiment kind %s" % action.kind)

    def _act_on_gaps(self, gaps: List[Dict[str, Any]]) -> None:
        """Turn the most severe data gap into an acquisition hypothesis."""
        if self.learning.champion is None or \
                self.sandbox.assay_remaining() <= 0 or \
                self.generator.open_acquisitions():
            return
        # Finish the initial model screen first: acquiring data to improve a
        # model that may soon be replaced wastes assay budget.
        if any(h["origin"] == "initial search space"
               for h in self.memory.hypotheses(status="proposed",
                                               kind="model")):
            return
        used_groups = {
            h["params"].get("group")
            for h in self.memory.hypotheses(kind="acquisition")
        }
        for gap in gaps:
            s = gap["suggestion"]
            if not s:
                continue
            if s["strategy"] == "group" and s["group"] in used_groups:
                continue
            self.generator.acquisition_hypothesis(
                s["strategy"], "knowledge gap: %s" % gap["description"],
                gap["severity"], s.get("group"))
            return

    # -- real-world experiments (human in the loop) ----------------------
    def propose_lab_experiments(self, n: int = 5) -> str:
        """Rank candidate molecules and request approval to test them.

        Nothing is executed: the request waits for a human decision.
        """
        if not self.vault.candidates:
            raise RuntimeError("objective has no candidates file")
        champ = self.learning.champion
        if champ is None:
            raise RuntimeError("no model to rank candidates with yet")
        smiles = self.vault.candidates
        pred, unc = self.learning.predict(smiles, self.policy.seed)
        df = pd.DataFrame({"smiles": smiles, "predicted": pred,
                           "uncertainty": unc})
        df = df.sort_values("predicted",
                            ascending=self.objective.desirable_direction ==
                            "low").head(n)
        validated = bool(
            self.memory.hypotheses(status="validated", kind="model"))
        description = (
            "Measure %s for %d candidate molecules predicted to have %s "
            "values by %s/%s (validation %s %.3f%s)." %
            (self.objective.target_column, len(df),
             self.objective.desirable_direction, champ["model"],
             champ["featurizer"], self.objective.primary_metric,
             champ["metric"], "" if validated else
             "; model NOT yet validated"))
        aid = self.sandbox.request_lab_experiment(
            description, {"molecules": df.to_dict("records"),
                          "model": champ})
        self.print("Requested human approval %s: %s" % (aid, description))
        return aid

    def ingest_lab_results(self, approval_id: str, results: pd.DataFrame,
                           smiles_column: str = "smiles",
                           value_column: str = "value") -> Dict[str, Any]:
        """Add measured results of an approved lab request and learn from
        them."""
        approval = self.memory.approval(approval_id)
        if approval is None:
            raise KeyError("No approval request %s" % approval_id)
        if approval["status"] != "approved":
            raise SandboxViolation(
                "results can only be ingested for approved requests "
                "(status: %s)" % approval["status"])
        allowed = {m["smiles"] for m in approval["payload"]["molecules"]}
        rows = results[[smiles_column, value_column]].dropna()
        accepted = rows[rows[smiles_column].isin(allowed)]
        if len(accepted) < len(rows):
            self.memory.log("lab_results_rejected",
                            count=len(rows) - len(accepted),
                            reason="molecules not in the approved request")
        self.memory.add_datapoints(accepted[smiles_column].tolist(),
                                   accepted[value_column].tolist(), "lab",
                                   approval_id)
        self.memory.complete_approval(approval_id)
        update = self.learning.retrain(
            self.policy.seed, "learned from %d lab results (%s)" %
            (len(accepted), approval_id))
        return {"accepted": len(accepted),
                "rejected": len(rows) - len(accepted),
                "model_update": update}

    # -- reporting -------------------------------------------------------
    def write_report(self) -> str:
        path = os.path.join(self.session_dir, REPORT_FILE)
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_report(self.config, self.memory))
        return path

    def status(self) -> Dict[str, Any]:
        hyps = self.memory.hypotheses()
        counts: Dict[str, int] = {}
        for h in hyps:
            counts[h["status"]] = counts.get(h["status"], 0) + 1
        return {
            "objective": self.objective.name,
            "cycle": self.memory.cycle,
            "runs": len(self.memory.runs()),
            "experiments": len(self.memory.experiments()),
            "hypotheses": counts,
            "champion": self.learning.champion,
            "assays_remaining": self.sandbox.assay_remaining(),
            "holdout_remaining": self.sandbox.holdout_remaining(),
            "pending_approvals": len(self.memory.approvals("pending")),
        }


def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[:n - 3] + "..."
