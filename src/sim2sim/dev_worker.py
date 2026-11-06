"""Explicit development worker: resource constrained, NOT filesystem isolated."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import resource


def apply_limits(limits):
    for key, resource_id, default in (("address_space_bytes", resource.RLIMIT_AS, 4*1024**3),
                                      ("data_bytes", resource.RLIMIT_DATA, 1024**3),
                                      ("file_bytes", resource.RLIMIT_FSIZE, 128*1024**2),
                                      ("open_files", resource.RLIMIT_NOFILE, 256)):
        value = int(limits.get(key, default))
        if value <= 0:
            raise ValueError("Development worker limits must be positive")
        resource.setrlimit(resource_id, (value, value))
    seconds = max(1, math.ceil(float(limits["cpu_seconds"])))
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import os
    if limits.get("pin_cpu", True) and hasattr(os, "sched_getaffinity"):
        available = os.sched_getaffinity(0)
        if available:
            os.sched_setaffinity(0, {min(available)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("candidate", "evaluation", "inspect"))
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--limits", required=True)
    args = parser.parse_args()
    import os
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") not in ("test", "sapphire"):
        raise RuntimeError("Development native workers require a Slurm test or sapphire allocation")
    apply_limits(json.loads(args.limits))
    if args.role == "inspect":
        from .inspector_worker import main as inspector_main
        inspector_main(workspace=args.workspace)
    elif args.role == "candidate":
        from .candidate_worker import main as candidate_main
        candidate_main(project_dir=args.workspace/"project")
    else:
        from .worker import main as evaluation_main
        evaluation_main(workspace=args.workspace)


if __name__ == "__main__":
    main()
