"""
Research Reporting: a Markdown report of findings, evidence and limitations.

The report distinguishes validated findings from provisional, inconclusive
and rejected ones, lists the evidence behind each claim, records failures and
blocked experiments, shows how the model learned over time and states the
limitations of the work honestly.
"""
import datetime
import os
from typing import Any, Dict, Iterable, List, Tuple

from chemflow.config import ResearchObjective, SessionConfig
from chemflow.memory import ResearchMemory

LIMITATIONS = [
    "All experiments are computational. Virtual assays reveal held-back "
    "measurements from the dataset; they are not new measurements.",
    "Structure-activity findings are statistical associations between a "
    "substructure and the target, not causal mechanisms, and can be "
    "confounded by molecular size or other correlated features.",
    "The SAR discovery test uses the data the hypothesis was generated "
    "from; only the validation and hold-out tests are independent.",
    "Validation and hold-out sets come from a scaffold split of a single "
    "dataset; performance on other chemistry or assays may differ.",
    "The sandbox time limit is enforced by abandoning slow experiments; "
    "Python cannot interrupt a running computation, so it may continue in "
    "the background until it finishes.",
    "Findings are only as reliable as the source measurements.",
]


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_None._\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(
            str(c).replace("|", "/").replace("\n", " ") for c in r) + " |")
    return "\n".join(out) + "\n"


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "%.*f" % (digits, x)
    return str(x)


def build_report(config: SessionConfig, memory: ResearchMemory) -> str:
    obj, pol = config.objective, config.sandbox
    findings = memory.findings()
    hyps = {h["id"]: h for h in memory.hypotheses()}
    exps = memory.experiments()
    champ = memory.get_state("champion")
    curve = memory.get_state("learning_curve", [])
    gaps = memory.get_state("knowledge_gaps", [])
    errors = memory.get_state("prediction_errors", [])
    runs = memory.runs()
    lines: List[str] = []
    add = lines.append

    add("# ChemFlow research report: %s\n" % obj.name)
    add("_Generated %s. Research cycles completed: %d across %d run(s)._\n" %
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), memory.cycle,
         len(runs)))
    add("## Research objective\n")
    add("**Question:** %s\n" % obj.question)
    add("- Data: `%s` (target `%s`, %s)" % (obj.dataset, obj.target_column,
                                             obj.task))
    op = "<=" if obj.lower_is_better else ">="
    add("- Success criterion: %d%% confidence bound of validation %s %s %s, "
        "reproduced and confirmed on a locked hold-out set\n" %
        (round(pol.confidence_level * 100), obj.primary_metric, op,
         obj.success_threshold))

    add("## Summary\n")
    validated = [f for f in findings if f["status"] == "validated"]
    if champ:
        add("- Best model: **%s on %s features**, validation %s = %.3f "
            "(CI %.3f to %.3f), trained on %d molecules." %
            (champ["model"], champ["featurizer"], obj.primary_metric,
             champ["metric"], champ["ci"][0], champ["ci"][1],
             champ["n_train"]))
    else:
        add("- No model has been trained yet.")
    add("- Findings: %d validated, %d provisional, %d inconclusive, "
        "%d rejected." %
        tuple(
            len([f for f in findings if f["status"] == s])
            for s in ("validated", "provisional", "inconclusive",
                      "rejected")))
    add("- Experiments: %d completed, %d failed or timed out, %d blocked by "
        "the sandbox." %
        (len([e for e in exps if e["status"] == "completed"]),
         len([e for e in exps if e["status"] in ("failed", "timeout")]),
         len([e for e in exps if e["status"] == "blocked"])))
    add("- Virtual assays used: %d of %d. Hold-out evaluations used: %d of "
        "%d.\n" % (len(memory.datapoints("virtual_assay")), pol.assay_budget,
                   memory.get_state("holdout_uses", 0),
                   pol.max_holdout_evaluations))

    add("## Validated findings\n")
    add("These claims met every sandbox requirement: confidence threshold, "
        "at least %d supporting experiments, reproducibility and independent "
        "hold-out validation.\n" % pol.min_supporting_experiments)
    add(_table(["Finding", "Claim", "Evidence"], [[
        f["id"], f["claim"], ", ".join(f["evidence"]["supporting_experiments"])
    ] for f in validated]))

    add("## Provisional and inconclusive findings\n")
    add("Supported by some evidence but not enough to claim. Reasons are the "
        "sandbox requirements still unmet.\n")
    add(_table(["Finding", "Status", "Claim", "Unmet requirements"], [[
        f["id"], f["status"], f["claim"],
        "; ".join(f["evidence"].get("certificate_reasons", []))
    ] for f in findings if f["status"] in ("provisional", "inconclusive")]))

    add("## Rejected hypotheses\n")
    add(_table(["Finding", "Claim", "Why rejected"], [[
        f["id"], f["claim"],
        "; ".join(f["evidence"].get("certificate_reasons", []))
    ] for f in findings if f["status"] == "rejected"]))

    add("## Model learning over time\n")
    add(_table(["Version", "Cycle", "Model", "Training molecules",
                obj.primary_metric, "Accepted", "Reason"],
               [[v["version"], v["cycle"], v["model"], v["n_train"],
                 _fmt(v["metric"]), "yes" if v["accepted"] else "no",
                 v["reason"]] for v in curve]))

    add("## Experiment log\n")
    add(_table(["Experiment", "Cycle", "Kind", "Hypothesis", "Status",
                "Result"], [[
                    e["id"], e["cycle"], e["kind"], e["hypothesis_id"],
                    e["status"],
                    _summarise(e)
                ] for e in exps]))

    if errors:
        abs_err = sum(abs(e["error"]) for e in errors) / len(errors)
        add("## Self-evaluation of the agent\n")
        add("Before each experiment the agent predicted its outcome. Mean "
            "absolute error of those predictions: %.3f %s over %d "
            "experiments.\n" % (abs_err, obj.primary_metric, len(errors)))

    add("## Knowledge gaps and open questions\n")
    add(_table(["Gap", "Severity", "Description"],
               [[g["kind"], g["severity"], g["description"]] for g in gaps]))

    approvals = memory.approvals()
    add("## Real-world experiments awaiting human decision\n")
    add("ChemFlow never runs laboratory experiments itself. Proposals below "
        "need a human decision (`python -m chemflow approve`).\n")
    add(_table(["Request", "Status", "Description"],
               [[a["id"], a["status"], a["description"]] for a in approvals]))

    add("## Limitations\n")
    for limitation in LIMITATIONS:
        add("- %s" % limitation)
    add("")
    return "\n".join(lines)


def build_results(config: SessionConfig, memory: ResearchMemory) -> Dict[str, Any]:
    """The same evidence as the report, as structured data for a results
    page. Per-molecule data (residuals, hold-out labels) is left out."""
    obj, pol = config.objective, config.sandbox
    findings = memory.findings()
    hyps = {h["id"]: h for h in memory.hypotheses()}
    exps = memory.experiments()
    errors = memory.get_state("prediction_errors", [])
    runs = memory.runs()
    models = _model_leaderboard(obj, hyps, exps, findings)
    return {
        "objective": {
            "name": obj.name,
            "question": obj.question,
            "dataset": os.path.basename(obj.dataset),
            "target": obj.target_column,
            "task": obj.task,
            "metric": obj.primary_metric,
            "lower_is_better": obj.lower_is_better,
            "threshold": obj.success_threshold,
            "confidence_level": pol.confidence_level,
        },
        "verdict": _verdict(obj, exps, models),
        "progress": {
            "cycles": memory.cycle,
            "runs": len(runs),
            "stop_reason": runs[-1]["stop_reason"] if runs else None,
        },
        "champion": memory.get_state("champion"),
        "counts": {
            "findings": _count(findings),
            "experiments": _count(exps),
            "hypotheses": _count(hyps.values()),
        },
        "budgets": {
            "experiments": {"used": len(exps),
                            "total": pol.max_experiments_total},
            "assays": {"used": len(memory.datapoints("virtual_assay")),
                       "total": pol.assay_budget},
            "holdout": {"used": memory.get_state("holdout_uses", 0),
                        "total": pol.max_holdout_evaluations},
        },
        "models": models,
        "learning_curve": memory.get_state("learning_curve", []),
        "structure_effects": _structure_effects(hyps, findings),
        "findings": [_finding(f, hyps) for f in findings],
        "experiments": [{
            "id": e["id"],
            "cycle": e["cycle"],
            "kind": e["kind"],
            "status": e["status"],
            "hypothesis_id": e["hypothesis_id"],
            "summary": _summarise(e),
        } for e in exps],
        "gaps": memory.get_state("knowledge_gaps", []),
        "approvals": [{
            "id": a["id"],
            "status": a["status"],
            "description": a["description"],
            "molecules": a["payload"].get("molecules", []),
        } for a in memory.approvals()],
        "self_evaluation": {
            "mean_absolute_error":
                sum(abs(e["error"]) for e in errors) / len(errors),
            "n": len(errors),
        } if errors else None,
        "limitations": LIMITATIONS,
    }


def _count(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


def _meets(obj: ResearchObjective, value: float) -> bool:
    t = obj.success_threshold
    return value <= t if obj.lower_is_better else value >= t


def _model_leaderboard(obj: ResearchObjective, hyps: Dict[str, Dict[str, Any]],
                       exps: List[Dict[str, Any]],
                       findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The best tested configuration of each model/featurizer pair, best
    first: a validated configuration if there is one, otherwise the highest
    scoring. A pair's metric is its first run; replicates and hold-out
    evaluations are reported beside it, as the evaluation engine does."""
    finding_of = {f["hypothesis_id"]: f for f in findings}
    pairs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for e in exps:
        h = hyps.get(e["hypothesis_id"])
        if e["kind"] != "train_evaluate" or e["status"] != "completed" or \
                h is None or h["kind"] != "model":
            continue
        pairs.setdefault((h["params"]["model"], h["params"]["featurizer"]),
                         []).append(e)
    sign = 1 if obj.lower_is_better else -1
    board = []
    for (model, featurizer), runs in pairs.items():
        best = min(runs, key=lambda e: (
            hyps[e["hypothesis_id"]]["status"] != "validated",
            sign * e["result"]["metric"]))
        r, hid = best["result"], best["hypothesis_id"]
        evidence = (finding_of.get(hid) or {}).get("evidence", {})
        board.append({
            "model": model,
            "featurizer": featurizer,
            "metric": r["metric"],
            "ci": [r["ci_low"], r["ci_high"]],
            "n_train": r["n_train"],
            "configs_tested": len(runs),
            "status": hyps[hid]["status"],
            "meets_target": _meets(obj, r["metric"]),
            "replicate_metric": evidence.get("replicate_metric"),
            "holdout_metric": evidence.get("holdout_metric"),
        })
    return sorted(board, key=lambda m: sign * m["metric"])


def _verdict(obj: ResearchObjective, exps: List[Dict[str, Any]],
             models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Whether the research objective has been answered, in one line."""
    name = "%s on %s" % (models[0]["model"],
                         models[0]["featurizer"]) if models else ""
    goal = "%s %s %s" % (obj.primary_metric, "<=" if obj.lower_is_better else
                         ">=", obj.success_threshold)
    validated = [m for m in models if m["status"] == "validated"]
    if validated:
        best = validated[0]
        return {
            "state": "validated",
            "headline": "Objective met: %s on %s is validated" %
                        (best["model"], best["featurizer"]),
            "detail": "Hit %s on validation data, reproduced it, and held up "
                      "on the locked hold-out set." % goal,
        }
    if not models:
        return {
            "state": "not_started",
            "headline": "No results yet",
            "detail": "%d experiment(s) recorded, none produced a trained "
                      "model. Run research to start." % len(exps),
        }
    if models[0]["meets_target"]:
        return {
            "state": "promising",
            "headline": "Promising, not yet validated: %s" % name,
            "detail": "The best model reaches %s, but the evidence does not "
                      "support a claim yet." % goal,
        }
    return {
        "state": "below_target",
        "headline": "Target not reached yet",
        "detail": "Best so far is %s; the target is %s." % (name, goal),
    }


def _structure_effects(hyps: Dict[str, Dict[str, Any]],
                       findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tested substructures, strongest discovery effect first."""
    effects = []
    for f in findings:
        h, ev = hyps[f["hypothesis_id"]], f["evidence"]
        if h["kind"] != "sar" or "discovery" not in ev:
            continue
        effects.append({
            "group": h["params"]["group"],
            "direction": h["params"]["direction"],
            "status": f["status"],
            "discovery": ev["discovery"],
            "replication": ev.get("replication"),
            "holdout": ev.get("holdout"),
        })
    return sorted(effects, key=lambda e: -abs(e["discovery"]["effect_size"]))


def _finding(f: Dict[str, Any], hyps: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    h, ev = hyps[f["hypothesis_id"]], f["evidence"]
    p = h["params"]
    return {
        "id": f["id"],
        "kind": h["kind"],
        "title": ("%s on %s" % (p["model"], p["featurizer"])
                  if h["kind"] == "model" else p["group"]),
        "status": f["status"],
        "claim": f["claim"],
        "confidence": f["confidence"],
        "checks": {
            "meets_confidence": ev.get("meets_confidence"),
            "reproducible": ev.get("reproducible"),
            "independent_validation": ev.get("independent_validation"),
        },
        "reasons": ev.get("certificate_reasons", []),
        "supporting_experiments": ev.get("supporting_experiments", []),
        "updated_cycle": f["updated_cycle"],
    }


def _summarise(e: Dict[str, Any]) -> str:
    r = e.get("result") or {}
    if e["status"] != "completed":
        return e.get("error") or ""
    if e["kind"] in ("train_evaluate", "replicate") or (
            e["kind"] == "holdout_validation" and "metric" in r):
        return "metric %.3f (CI %.3f to %.3f)" % (r["metric"], r["ci_low"],
                                                  r["ci_high"])
    if e["kind"] in ("sar_test", "holdout_validation"):
        return "%s: d=%.2f, p=%s, n=%d" % (r.get("split"),
                                           r["effect_size"],
                                           _fmt(r["p_value"], 4),
                                           r["n_with"])
    if e["kind"] == "virtual_assay":
        return "acquired %d; metric %.3f -> %.3f" % (
            r["acquired"], r["metric_before"], r["metric_after"])
    return ""
