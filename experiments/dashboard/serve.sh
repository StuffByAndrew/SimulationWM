#!/usr/bin/env bash
# Usage: ./serve.sh [--static] [SSH_DESTINATION]
# Requests a Slurm test CPU node for the dashboard and interactive physics.
# --static serves recordings only from this login node.
dashboard_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM2SIM_DASHBOARD_DIR="$dashboard_dir" exec "$dashboard_dir/../../scripts/serve-dashboard.sh" "$@"
