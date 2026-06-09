#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  start_servers.sh  —  Launch llama-server + supertonic serve
#  Both run as independent OS processes; Python's GIL cannot affect them.
#
#  Usage:
#    chmod +x start_servers.sh
#    ./start_servers.sh            # uses config.yaml next to this script
#    ./start_servers.sh stop       # kill all managed servers
#    ./start_servers.sh status     # show PID / port status
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${SCRIPT_DIR}/.venv/bin:${PATH}"
CONFIG="${SCRIPT_DIR}/config.yaml"
PIDFILE="${SCRIPT_DIR}/.server_pids"
LLAMA_LOG="${SCRIPT_DIR}/logs/llama_server.log"
SUPERTONIC_LOG="${SCRIPT_DIR}/logs/supertonic.log"

mkdir -p "${SCRIPT_DIR}/logs"

# ── YAML parser (awk-based, handles indented keys correctly) ──────────────────
yaml_get() {
    # Usage: yaml_get <key> [section]
    # key:     the field name (e.g. "port")
    # section: optional parent block (e.g. "supertonic")
    local key="$1"
    local section="${2:-}"

    if [[ -z "$section" ]]; then
        awk -v k="$key" '
            /^[[:space:]]*'"$key"':[[:space:]]/ {
                sub(/^[^:]*:[[:space:]]*/, ""); gsub(/["'\'']/, ""); sub(/[[:space:]]+$/, ""); print; exit
            }' "$CONFIG"
    else
        awk -v sec="$section" -v k="$key" '
            $0 ~ "^"sec":" { in_sec=1; next }
            in_sec && /^[^ \t]/ { exit }
            in_sec && $0 ~ "^[[:space:]]+"k":" {
                sub(/^[^:]*:[[:space:]]*/, ""); gsub(/["'\'']/, ""); sub(/[[:space:]]+$/, ""); print; exit
            }' "$CONFIG"
    fi
}

LLM_GGUF=$(yaml_get "llm_gguf"    "models")
MMPROJ=$(yaml_get   "mmproj_gguf" "models")
LLAMA_HOST=$(yaml_get "host"      "llama_server")
LLAMA_PORT=$(yaml_get "port"      "llama_server")
GPU_LAYERS=$(yaml_get "gpu_layers" "llama_server")
CTX_SIZE=$(yaml_get  "ctx_size"   "llama_server")
ENABLE_THINKING=$(yaml_get "enable_thinking" "llama_server")
SUPERTONIC_PORT=$(yaml_get "port" "supertonic")

# ── stop command ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "stop" ]]; then
    echo "🛑  Stopping Polyglot servers..."
    if [[ -f "$PIDFILE" ]]; then
        while read -r pid; do
            kill "$pid" 2>/dev/null && echo "   Killed PID $pid" || true
        done < "$PIDFILE"
        rm -f "$PIDFILE"
    fi
    # Belt-and-suspenders
    pkill -f "llama-server.*${LLAMA_PORT}"    2>/dev/null || true
    pkill -f "supertonic serve.*${SUPERTONIC_PORT}" 2>/dev/null || true
    echo "✅  Done."
    exit 0
fi

# ── status command ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "status" ]]; then
    echo "── Server Status ──────────────────────────────"
    for name in "llama-server:${LLAMA_PORT}" "supertonic:${SUPERTONIC_PORT}"; do
        svc="${name%%:*}"; port="${name##*:}"
        if fuser "${port}/tcp" &>/dev/null 2>&1; then
            echo "  ✅  $svc  (port $port)"
        else
            echo "  ❌  $svc  (port $port)"
        fi
    done
    exit 0
fi

# ── validate paths ────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🌍  Polyglot Server Launcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  LLM  %-12s %s\n" "GGUF:"  "$LLM_GGUF"
printf "  LLM  %-12s %s\n" "mmproj:" "$MMPROJ"
printf "  Port %-12s %s\n" "llama:"  "$LLAMA_PORT"
printf "  Port %-12s %s\n" "tts:"    "$SUPERTONIC_PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[[ -f "$LLM_GGUF" ]] || { echo "❌  llm_gguf not found: $LLM_GGUF"; exit 1; }
[[ -f "$MMPROJ"   ]] || { echo "❌  mmproj_gguf not found: $MMPROJ"; exit 1; }

# ── clear ports ───────────────────────────────────────────────────────────────
for port in "$LLAMA_PORT" "$SUPERTONIC_PORT"; do
    fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 1
> "$PIDFILE"

# ── CUDA path (Colab / NVIDIA) ────────────────────────────────────────────────
if [[ -d "/content/llama-bin/cuda-12.8" ]]; then
    export LD_LIBRARY_PATH="/content/llama-bin/cuda-12.8:${LD_LIBRARY_PATH:-}"
fi

# Add PyPI CUDA 12 libraries from virtual environment to LD_LIBRARY_PATH for onnxruntime-gpu
NVIDIA_DIR="${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/nvidia"
if [[ -d "$NVIDIA_DIR" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_DIR}/cublas/lib:${NVIDIA_DIR}/cufft/lib:${NVIDIA_DIR}/cuda_runtime/lib:${NVIDIA_DIR}/cuda_nvrtc/lib:${NVIDIA_DIR}/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi


# ── 1. llama-server ───────────────────────────────────────────────────────────
echo ""
echo "🚀  Starting llama-server..."

THINKING_FLAG=""
if [[ "$ENABLE_THINKING" == "false" ]]; then
    THINKING_FLAG="--chat-template-kwargs {\"enable_thinking\":false}"
fi

# shellcheck disable=SC2086
nohup llama-server \
    -m          "$LLM_GGUF"   \
    --mmproj    "$MMPROJ"     \
    --host      "$LLAMA_HOST" \
    --port      "$LLAMA_PORT" \
    -ngl        "$GPU_LAYERS" \
    -c          "$CTX_SIZE"   \
    --flash-attn on           \
    --no-mmap                 \
    $THINKING_FLAG            \
    > "$LLAMA_LOG" 2>&1 &

LLAMA_PID=$!
echo "$LLAMA_PID" >> "$PIDFILE"
echo "   PID $LLAMA_PID  →  $LLAMA_LOG"

# ── 2. supertonic serve ───────────────────────────────────────────────────────
echo ""
echo "🔊  Starting Supertonic V3 serve..."

nohup supertonic serve --port "$SUPERTONIC_PORT" \
    > "$SUPERTONIC_LOG" 2>&1 &

SUPERTONIC_PID=$!
echo "$SUPERTONIC_PID" >> "$PIDFILE"
echo "   PID $SUPERTONIC_PID  →  $SUPERTONIC_LOG"

# ── 3. Health-check loop ──────────────────────────────────────────────────────
echo ""
echo "⏳  Waiting for servers..."

LLAMA_READY=0
SUPERTONIC_READY=0
DEADLINE=$((SECONDS + 180))

while [[ $SECONDS -lt $DEADLINE ]]; do

    # llama-server: check log for idle signal OR HTTP /health
    if [[ $LLAMA_READY -eq 0 ]]; then
        if grep -q "all slots are idle" "$LLAMA_LOG" 2>/dev/null || \
           curl -sf "http://localhost:${LLAMA_PORT}/health" -o /dev/null 2>/dev/null; then
            echo "   ✅  llama-server  ready  (port $LLAMA_PORT)"
            LLAMA_READY=1
        elif ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "   ❌  llama-server CRASHED"; tail -n 15 "$LLAMA_LOG"; exit 1
        fi
    fi

    # supertonic: no /health endpoint — use a real TTS probe request instead
    if [[ $SUPERTONIC_READY -eq 0 ]]; then
        if curl -sf -X POST "http://localhost:${SUPERTONIC_PORT}/v1/tts" \
               -H "Content-Type: application/json"                        \
               -d '{"text":"ok","lang":"en","voice":"F1"}'                \
               --max-time 5 -o /dev/null 2>/dev/null; then
            echo "   ✅  Supertonic V3  ready  (port $SUPERTONIC_PORT)"
            SUPERTONIC_READY=1
        # fallback: check uvicorn startup line in log
        elif grep -qE "Uvicorn running|Application startup complete" \
                  "$SUPERTONIC_LOG" 2>/dev/null; then
            echo "   ✅  Supertonic V3  ready  (log confirm, port $SUPERTONIC_PORT)"
            SUPERTONIC_READY=1
        elif ! kill -0 "$SUPERTONIC_PID" 2>/dev/null; then
            echo "   ❌  Supertonic CRASHED"; tail -n 15 "$SUPERTONIC_LOG"; exit 1
        fi
    fi

    [[ $LLAMA_READY -eq 1 && $SUPERTONIC_READY -eq 1 ]] && break
    printf "."
    sleep 2
done

echo ""
if [[ $LLAMA_READY -eq 0 || $SUPERTONIC_READY -eq 0 ]]; then
    [[ $LLAMA_READY      -eq 0 ]] && echo "⏰  llama-server timed out → $LLAMA_LOG"
    [[ $SUPERTONIC_READY -eq 0 ]] && echo "⏰  Supertonic timed out   → $SUPERTONIC_LOG"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Both servers ready!"
echo "  llama-server  →  http://localhost:${LLAMA_PORT}"
echo "  Supertonic V3 →  http://localhost:${SUPERTONIC_PORT}"
echo ""
echo "  Next: python main.py              (terminal UI)"
echo "        python main.py --web        (web UI on port 7860)"
echo "        ./start_servers.sh stop     (shutdown)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
