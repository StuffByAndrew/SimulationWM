#!/usr/bin/env bash
# Keep the lightweight job monitor (src/sim2sim/monitor.py) up on this login node at http://127.0.0.1:PORT/ (default
# 8090; also reachable over this node's Tailscale address). Idempotent: does nothing if it already runs. Its state
# (strict-job CPU/RAM history, spend snapshots) is kept in experiments/monitor/. Run from the main checkout.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
port="${SIM2SIM_MONITOR_PORT:-8090}"
if pgrep -u "$(id -un)" -f "sim2sim.monitor --port $port" >/dev/null; then
    echo "monitor already running on port $port"
    exit 0
fi
mkdir -p "$repo_root/experiments/monitor"
cd "$repo_root"
PYTHONPATH="$repo_root/src" setsid nohup "${SIM2SIM_PYTHON:-$repo_root/.venv/bin/python}" -m sim2sim.monitor --port "$port" \
    >> experiments/monitor/monitor.log 2>&1 < /dev/null &
echo "monitor started on http://127.0.0.1:$port/ (pid $!, log experiments/monitor/monitor.log)"
