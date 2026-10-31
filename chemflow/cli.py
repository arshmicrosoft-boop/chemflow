"""Command-line interface: ``python -m chemflow <command>``."""
import argparse
import json
import sys

import pandas as pd

from chemflow.agent import ResearchAgent
from chemflow.config import SessionConfig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="chemflow",
        description="ChemFlow: autonomous computational research on top of "
        "DeepChem.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a research session")
    p.add_argument("session")
    p.add_argument("--config", required=True, help="session config JSON")

    p = sub.add_parser("run", help="run (or continue) research cycles")
    p.add_argument("session")
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--minutes", type=float, default=None)

    for name, text in (("status", "show session status"),
                       ("report", "write the research report"),
                       ("approvals", "list human approval requests")):
        sub.add_parser(name, help=text).add_argument("session")

    p = sub.add_parser("propose",
                       help="rank candidates and request lab approval")
    p.add_argument("session")
    p.add_argument("-n", type=int, default=5)

    p = sub.add_parser("approve", help="approve or reject a lab request")
    p.add_argument("session")
    p.add_argument("approval_id")
    p.add_argument("--by", required=True, help="name of the approver")
    p.add_argument("--reject", action="store_true")
    p.add_argument("--note", default="")

    p = sub.add_parser("ingest", help="ingest results of an approved request")
    p.add_argument("session")
    p.add_argument("approval_id")
    p.add_argument("results_csv")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--value-column", default="value")

    args = parser.parse_args(argv)
    if args.command == "init":
        agent = ResearchAgent.create(args.session,
                                     SessionConfig.from_json(args.config))
        print("Created session in %s" % args.session)
    else:
        agent = ResearchAgent.resume(args.session)

    try:
        if args.command == "run":
            agent.run(args.cycles,
                      None if args.minutes is None else args.minutes * 60)
        elif args.command == "status":
            print(json.dumps(agent.status(), indent=2, default=str))
        elif args.command == "report":
            print(agent.write_report())
        elif args.command == "approvals":
            for a in agent.memory.approvals():
                print("%s  %-9s %s" % (a["id"], a["status"], a["description"]))
                for m in a["payload"].get("molecules", []):
                    print("    %-50s predicted %.3f +/- %.3f" %
                          (m["smiles"], m["predicted"], m["uncertainty"]))
        elif args.command == "propose":
            agent.propose_lab_experiments(args.n)
        elif args.command == "approve":
            agent.memory.decide_approval(args.approval_id, not args.reject,
                                         args.by, args.note)
            print("%s %s" % (args.approval_id,
                             "rejected" if args.reject else "approved"))
        elif args.command == "ingest":
            out = agent.ingest_lab_results(args.approval_id,
                                           pd.read_csv(args.results_csv),
                                           args.smiles_column,
                                           args.value_column)
            print(json.dumps(out, indent=2, default=str))
    finally:
        agent.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
