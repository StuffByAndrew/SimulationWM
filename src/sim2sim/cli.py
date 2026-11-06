"""sim2sim command line."""
import argparse
import json
from pathlib import Path
import sys
import tomllib

from .runtime import RuntimeFailure, run
from .audit import AuditFailure


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a sandboxed scene reconstruction agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run an agent from a TOML configuration")
    run_parser.add_argument("config", type=Path)
    replay_parser = sub.add_parser("replay", help="verify and replay exact archived frames without rerunning models or external calls")
    replay_parser.add_argument("run_dir", type=Path)
    replay_parser.add_argument("--output", type=Path, required=True)
    experiment_parser = sub.add_parser("experiment", help="prepare or run a private configured experiment")
    experiment_parser.add_argument("config", type=Path, nargs="?")
    experiment_parser.add_argument("--root", type=Path, default=Path("experiments"))
    experiment_parser.add_argument("--id", dest="experiment_id")
    experiment_parser.add_argument("--prepare-only", action="store_true")
    experiment_parser.add_argument("--development", action="store_true", help="explicitly use the approved shared-filesystem development backend")
    experiment_parser.add_argument("--start", metavar="ID", help="start an existing prepared experiment")
    experiment_parser.add_argument("--no-dashboard", action="store_true", help="defer dashboard generation until parallel runs finish")
    dashboard_parser = sub.add_parser("dashboard", help="refresh the private multi-experiment dashboard")
    dashboard_parser.add_argument("--root", type=Path, action="append", help="experiment root; repeat for several (default experiments)")
    dashboard_parser.add_argument("--output", type=Path)
    dashboard_parser.add_argument("--workers", type=int, default=1, help="parallel run snapshots (NFS-bound; e.g. 16)")
    inspect_parser = sub.add_parser("inspect", help="serve the live dashboard with interactive physics (physics on Slurm test nodes only)")
    inspect_parser.add_argument("--root", type=Path, action="append", help="experiment root; repeat for several (default experiments)")
    inspect_parser.add_argument("--dashboard", type=Path)
    inspect_parser.add_argument("--ports", type=int, nargs=5, default=(8000, 8001, 8002, 8003, 8004), metavar=("HTTP", "REFERENCE", "CANDIDATE", "OVERLAY", "GALLERY"))
    inspect_parser.add_argument("--no-physics", action="store_true", help="live recordings only, without viewers (login node allowed); uses only the HTTP port")
    inspect_parser.add_argument("--session-seconds", type=float, default=2 * 3600)
    inspect_parser.add_argument("--ready-file", type=Path)
    cost_parser = sub.add_parser("cost", help="print host-only spend of a shared inference cost pool")
    cost_parser.add_argument("--pool", required=True)
    cost_parser.add_argument("--ledger", type=Path, default=Path("experiments/cost-ledger"))
    args = parser.parse_args(argv)
    try:
        if args.command == "experiment":
            from .experiments import prepare_experiment, run_experiment, start_experiment
            if args.start:
                if args.config or args.prepare_only or args.experiment_id:
                    raise ValueError('--start takes an existing id without config/--id/--prepare-only')
                result = start_experiment(args.start, args.root, development=args.development, refresh_dashboard=not args.no_dashboard)
            else:
                if args.config is None:
                    raise ValueError('experiment requires a configuration or --start ID')
                function = prepare_experiment if args.prepare_only else run_experiment
                result = function(args.config, args.root, args.experiment_id, **({'development': args.development, 'refresh_dashboard': not args.no_dashboard} if not args.prepare_only else {}))
            print(json.dumps(result, indent=2))
            return 2 if result.get('status') == 'blocked' else 1 if result.get('status') == 'failed' else 0
        elif args.command == "dashboard":
            from .dashboard import build_dashboard
            print(json.dumps(build_dashboard(args.root or [Path("experiments")], args.output, args.workers), indent=2))
        elif args.command == "inspect":
            from .inspector import serve
            roots = args.root or [Path("experiments")]
            serve(roots, args.dashboard or roots[0] / "dashboard", args.ports, args.session_seconds, args.ready_file, physics=not args.no_physics)
        elif args.command == "cost":
            from .runtime import read_cost_pool
            if not (args.ledger / args.pool).is_dir():
                raise ValueError("No cost pool directory " + str(args.ledger / args.pool))
            runs = read_cost_pool(args.ledger / args.pool)
            print(json.dumps({"pool": args.pool, "runs": runs, **{"total_" + k: sum(r[k] for r in runs.values())
                              for k in ("cost_usd", "inflight_reserve_usd", "calls", "tokens")}}, indent=2))
        elif args.command == "replay":
            from .replay import replay_run
            print(json.dumps(replay_run(args.run_dir, args.output), indent=2))
        else:
            with args.config.open("rb") as handle:
                config = tomllib.load(handle)
            print(json.dumps(run(config), indent=2))
    except (RuntimeFailure, AuditFailure, OSError, ValueError, KeyError) as exc:
        print("sim2sim: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
