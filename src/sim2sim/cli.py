"""sim2sim command line."""
import argparse
import json
from pathlib import Path
import sys
import tomllib

from .runtime import RuntimeFailure, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a sandboxed scene reconstruction agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run an agent from a TOML configuration")
    run_parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.config.open("rb") as handle:
            config = tomllib.load(handle)
        print(json.dumps(run(config), indent=2))
    except (RuntimeFailure, OSError, ValueError, KeyError) as exc:
        print("sim2sim: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
