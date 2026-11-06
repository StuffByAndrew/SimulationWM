#!/usr/bin/env bash
# Serve the private dashboard plus interactive physics from a Slurm test CPU node.
# Run this on a login node; the physics server itself never runs there.
#
# Usage: scripts/serve-dashboard.sh [--static] [SSH_DESTINATION]
#   SSH_DESTINATION  client-side SSH target for this login node (default USER@HOST)
#   --static         recordings only: serve the snapshot from this login node, no physics
# Environment: SIM2SIM_PYTHON, SIM2SIM_EXPERIMENTS, SIM2SIM_DASHBOARD_DIR,
#   SIM2SIM_INSPECT_TIME (02:00:00), SIM2SIM_INSPECT_CPUS (4), SIM2SIM_INSPECT_MEM (8G)
set -euo pipefail

static=0
if [[ "${1:-}" == --static ]]; then static=1; shift; fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SIM2SIM_PYTHON:-$repo_root/.venv/bin/python}"
root="${SIM2SIM_EXPERIMENTS:-$repo_root/experiments}"  # colon-separated for several roots; the first holds dashboard/
IFS=: read -r -a roots <<<"$root"
root_args=(); for r in "${roots[@]}"; do root_args+=(--root "$r"); done
dashboard="$(cd -- "${SIM2SIM_DASHBOARD_DIR:-${roots[0]}/dashboard}" && pwd)"
wall="${SIM2SIM_INSPECT_TIME:-02:00:00}"
cpus="${SIM2SIM_INSPECT_CPUS:-4}"
mem="${SIM2SIM_INSPECT_MEM:-8G}"
destination="${1:-$(id -un)@$(hostname)}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo 'Run this from a login node; it requests its own Slurm test allocation.' >&2
    exit 2
fi
if [[ ! -f "$dashboard/catalog.js" ]]; then
    echo "No dashboard snapshot in $dashboard. Generate it first: $python_bin -m sim2sim.cli dashboard ${root_args[*]}" >&2
    exit 2
fi

# Three consecutive ports free on this login node: HTTP, reference viewer, reconstruction viewer.
read -r http reference candidate < <("$python_bin" - <<'PY'
import socket
for base in range(8000, 8100, 3):
    sockets = []
    try:
        for port in range(base, base + 3):
            sockets.append(socket.socket())
            sockets[-1].bind(('127.0.0.1', port))
    except OSError:
        continue
    finally:
        for sock in sockets:
            sock.close()
    print(base, base + 1, base + 2)
    break
else:
    raise SystemExit('No three consecutive free ports in 8000-8099')
PY
)

print_client() {
    echo "On your client machine, run:"
    echo "  ssh -N -o ExitOnForwardFailure=yes $1 ${2:-$destination}"
    echo
    echo "Then open in your browser:"
    echo "  http://127.0.0.1:$http/"
    echo
    echo "Keep both terminals open. Press Ctrl-C in each to stop."
}

if (( static )); then
    print_client "-L 127.0.0.1:$http:127.0.0.1:$http"
    echo "Static mode: recordings only; the Interactive physics tab reports that no inspector is running."
    exec "$python_bin" -m http.server "$http" --bind 127.0.0.1 --directory "$dashboard"
fi

IFS=: read -r -a parts <<<"$wall"
seconds=0
for part in "${parts[@]}"; do seconds=$(( seconds * 60 + 10#$part )); done

mkdir -p "$HOME/.cache"
state_dir="$(mktemp -d "$HOME/.cache/sim2sim-dashboard.XXXXXX")"  # shared FS, visible to the node
ready="$state_dir/ready.json"
srun_pid=
tunnel_pid=
cleanup() {
    [[ -n "$tunnel_pid" ]] && kill "$tunnel_pid" 2>/dev/null || true
    if [[ -n "$srun_pid" ]] && kill -0 "$srun_pid" 2>/dev/null; then
        kill "$srun_pid" 2>/dev/null || true
        wait "$srun_pid" 2>/dev/null || true
    fi
    rm -rf "$state_dir"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Value of KEY=VALUE in one-line `scontrol show ... -o` output.
slurm_field() { tr ' ' '\n' <<<"$1" | sed -n "s/^$2=//p" | head -1; }
gpu_count() { local count; count="$(tr ',' '\n' <<<"$1" | sed -n 's/^gres\/gpu[^=]*=//p' | head -1)"; echo "${count:-0}"; }

cat <<EOF
Requesting a Slurm allocation for the dashboard and physics inspector:
  partition    test
  nodes        1 (one task)
  CPUs         $cpus
  GPUs         0 (CPU-only; none requested)
  memory       $mem
  time limit   $wall
  job name     sim2sim-dashboard
EOF
srun --partition=test --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem" --time="$wall" \
    --job-name=sim2sim-dashboard --chdir="$repo_root" \
    --export="ALL,MUJOCO_GL=disable,PYTHONPATH=$repo_root/src" \
    "$python_bin" -u -m sim2sim.cli inspect "${root_args[@]}" --dashboard "$dashboard" \
    --ports "$http" "$reference" "$candidate" --session-seconds "$seconds" --ready-file "$ready" </dev/null &
srun_pid=$!

until [[ -s "$ready" ]]; do
    if ! kill -0 "$srun_pid" 2>/dev/null; then
        echo 'The inspector job ended before it became ready; see the output above.' >&2
        exit 1
    fi
    sleep 1
done
read -r node job < <("$python_bin" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["node"], r["job"])' "$ready")

# Report what Slurm actually allocated, not just what was requested.
job_info="$(scontrol show job -o "$job" 2>/dev/null || true)"
node_info="$(scontrol show node -o "$(slurm_field "$job_info" NodeList || true)" 2>/dev/null || true)"
if [[ -n "$job_info" ]]; then
    tres="$(slurm_field "$job_info" AllocTRES)"; tres="${tres:-$(slurm_field "$job_info" TRES)}"
    node_gres="$(slurm_field "$node_info" Gres)"
    [[ "$node_gres" == "(null)" ]] && node_gres=none
    cat <<EOF

Allocated Slurm job $job:
  partition    $(slurm_field "$job_info" Partition)
  node         $(slurm_field "$job_info" NodeList) ($(slurm_field "$node_info" CPUTot) CPUs total, GPUs on node: ${node_gres:-unknown})
  CPUs         $(slurm_field "$job_info" NumCPUs) allocated
  GPUs         $(gpu_count "$tres") allocated
  resources    $tres
  time limit   $(slurm_field "$job_info" TimeLimit) (started $(slurm_field "$job_info" StartTime))
EOF
else
    echo "Allocated Slurm job $job on $node (scontrol details unavailable)."
fi
echo

# The node listens only on its loopback; forward the same ports from this login node.
forwards=()
for port in "$http" "$reference" "$candidate"; do forwards+=(-L "127.0.0.1:$port:127.0.0.1:$port"); done
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new "${forwards[@]}" "$node" &
tunnel_pid=$!
sleep 3
if ! kill -0 "$tunnel_pid" 2>/dev/null; then
    tunnel_pid=
    echo "Could not forward ports from $(hostname) to $node. Tunnel through both hops instead:" >&2
    print_client "-J $destination ${forwards[*]}" "$(id -un)@$node"
else
    echo "Inspector ready on $node (Slurm job $job)."
    echo
    print_client "${forwards[*]}"
fi
wait "$srun_pid"
