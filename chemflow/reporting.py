"""
Research Reporting: a Markdown report of findings, evidence and limitations.

The report distinguishes validated findings from provisional, inconclusive
and rejected ones, lists the evidence behind each claim, records failures and
blocked experiments, shows how the model learned over time and states the
limitations of the work honestly.
"""
import datetime
from typing import Any, Dict, List

from chemflow.config import SessionConfig
from chemflow.memory import ResearchMemory


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
    add("- All experiments are computational. Virtual assays reveal held-back "
        "measurements from the dataset; they are not new measurements.")
    add("- Structure-activity findings are statistical associations between "
        "a substructure and the target, not causal mechanisms, and can be "
        "confounded by molecular size or other correlated features.")
    add("- The SAR discovery test uses the data the hypothesis was generated "
        "from; only the validation and hold-out tests are independent.")
    add("- Validation and hold-out sets come from a scaffold split of a "
        "single dataset; performance on other chemistry or assays may "
        "differ.")
    add("- The sandbox time limit is enforced by abandoning slow "
        "experiments; Python cannot interrupt a running computation, so it "
        "may continue in the background until it finishes.")
    add("- Findings are only as reliable as the source measurements.\n")
    return "\n".join(lines)


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
