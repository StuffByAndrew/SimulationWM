#!/usr/bin/env bash
#SBATCH --job-name=inkling-service
#SBATCH --partition=kempner_gpu_priority
#SBATCH --account=kempner_qianqian_lab
#SBATCH --qos=kempner_qianqian_priority
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:2
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=inkling-service-%j.log
# Researcher-run Inkling 2 Mini service for strict runs with [inference] api = "local_openai" (zero cost; never H100/H200).
# vLLM and SGLang run this checkpoint only on Hopper/Blackwell (vLLM's inkling2 quantization needs capability >= 9.0), so
# on A100s llama.cpp (INKLING_ROOT/src, b12706, sm_80) serves the official checkpoint converted to GGUF by llama.cpp's own
# converter: its NVFP4/MXFP8 experts dequantized exactly into BF16, computed with BF16 activations (two 40 GB A100s, layer
# split). While healthy it rewrites the private record INKLING_ROOT/service/service.json (0600 in a 0700 dir: address,
# per-job bearer token, heartbeat, engine and checkpoint) every 5 s; the strict controller's broker reads it per call, so
# a dead or requeued service is a provider outage there. Start it before the experiment jobs and keep -t longer than they
# run. Also works on kempner_requeue (-p kempner_requeue, no --qos): preemption then shows up as an outage that
# provider_hold_seconds holds until the service is back.
set -euo pipefail
root="${INKLING_ROOT:-/n/netscratch/qianqian_lab/Lab/aejxie/inkling}"
slots="${INKLING_SLOTS:-5}"              # concurrent requests (one per strict run)
context="${INKLING_CONTEXT:-262144}"     # unified KV cache shared by the slots; each request may use all of it
record="$root/service/service.json"
gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd,)"
[[ "$gpus" != *H100* && "$gpus" != *H200* ]] || { echo "refusing H100/H200 GPU: $gpus" >&2; exit 2; }
(cd "$root/gguf" && sha256sum -c --quiet sha256.txt) || { echo "GGUF files do not match $root/gguf/sha256.txt" >&2; exit 2; }
mkdir -p -m 700 "$root/service"
[[ -z "$(find "$record" -mmin -2 2>/dev/null)" ]] || { echo "another service is heartbeating $record" >&2; exit 2; }
source /etc/profile.d/lmod.sh
module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01
LLAMA_API_KEY="$(openssl rand -hex 32)"
export LLAMA_API_KEY  # environment, not argv: other users on the node cannot read it from ps
port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
host="$(hostname -f)"
server_bin="$root/src/llama.cpp-b12706/build/bin/llama-server"
engine="llama.cpp $("$server_bin" --version 2>&1 | grep -o 'version: [^ ]*' | cut -d' ' -f2) (b12706, CUDA sm_80)"
printf 'node=%s job=%s gpus=%s port=%s engine=%s slots=%s context=%s\n' "$host" "$SLURM_JOB_ID" "$gpus" "$port" "$engine" "$slots" "$context"
# Sampling: the checkpoint's generation_config (temperature 1.0) without truncation, as the model card recommends.
# The layer split counts 48 cache slots (24 physical layers, then 24 loop-pass slots that reuse layers 6-17's weights).
# Flash attention needs each loop-pass slot on the device of its layer's attention weights, so GPU 0 takes slots 0-5
# (layers 0-5) and GPU 1 the rest; the experts of layers 6-13 move to GPU 0 to balance memory (MoE ops run there).
# The relative-position bias is gathered as an f32 [cache cells x ubatch x heads] tensor per full-attention layer, so the
# physical batch stays at 64 tokens (about 3 GB of compute buffer at 262144 cells). Idle slots swap to host RAM.
"$server_bin" -m "$root/gguf/Inkling-2-Mini-BF16.gguf" --mmproj "$root/gguf/mmproj-Inkling-2-Mini-BF16.gguf" \
    --alias inkling-2-mini --host 0.0.0.0 --port "$port" --jinja --reasoning-format deepseek \
    -ngl 999 --split-mode layer --tensor-split 1,7 -ot 'blk\.([6-9]|1[0-3])\.ffn_(gate_up|down)_exps\.weight=CUDA0' -fa on -ub 64 \
    -c "$context" -np "$slots" --kv-unified --cache-ram 65536 \
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --no-webui --metrics ${INKLING_EXTRA_ARGS:-} &
server=$!
trap 'rm -f "$record"; kill $server 2>/dev/null; wait $server' EXIT
until curl -fsS -m 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 $server 2>/dev/null || { echo "llama-server exited before becoming healthy" >&2; exit 1; }
    sleep 5
done
echo "healthy at $(date -u +%FT%TZ)"
umask 077
while kill -0 $server 2>/dev/null; do
    if curl -fsS -m 10 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        printf '{"endpoint":"http://%s:%s/v1/chat/completions","model":"inkling-2-mini","token":"%s","heartbeat_unix":%s,"job":"%s","node":"%s","gpus":"%s","engine":"%s","checkpoint":"thinkingmachines/Inkling-2-Mini@c9d028302cd0f035fdf8e7e91f089a4ce94d2195","precision":"official NVFP4/MXFP8 expert weights dequantized exactly to BF16 GGUF (llama.cpp converter); BF16 weights and activations, F16 KV cache","slots":%s,"context":%s}\n' \
            "$host" "$port" "$LLAMA_API_KEY" "$(date +%s.%N)" "$SLURM_JOB_ID" "$host" "$gpus" "$engine" "$slots" "$context" > "$record.tmp"
        mv -f "$record.tmp" "$record"
    fi
    sleep 5
done
echo "llama-server exited at $(date -u +%FT%TZ)" >&2
exit 1
