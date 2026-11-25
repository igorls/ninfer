#!/usr/bin/env bash
set -euo pipefail

# General-purpose Google Colab G4 (RTX PRO 6000 Blackwell 96 GB, sm_120a)
# bootstrapper for NInfer (`ninfer-serve`) listening privately on 127.0.0.1.
#
# Fast path (~90s): pulls a precompiled `ninfer-serve-linux-x86_64-sm120a.tar.gz`
# from GitHub Releases (or NINFER_BINARY_URL) while downloading the `.ninfer`
# model in parallel.
#
# Fallback / rebuild path (NINFER_BUILD_FROM_SOURCE=1): installs CUDA 13.3 +
# build deps, compiles `ninfer-serve` from source (-DNINFER_TARGET=rtx5090),
# and launches it.
#
# Never installs Tailscale or binds a public interface — bridge 127.0.0.1:$PORT
# privately via `scripts/colab/colab-ninfer.ts tunnel` or `bridge`.

WORKDIR="${NINFER_WORKDIR:-/content/ninfer-colab}"
GITHUB_REPO="${NINFER_GITHUB_REPO:-igorls/ninfer}"
REPO_URL="${NINFER_REPO_URL:-https://github.com/${GITHUB_REPO}.git}"
REPO_BRANCH="${NINFER_BRANCH:-workstation}"
RELEASE_TAG="${NINFER_RELEASE_TAG:-colab-linux-sm120a}"
BINARY_ASSET="${NINFER_BINARY_ASSET:-ninfer-serve-linux-x86_64-sm120a.tar.gz}"
BINARY_URL="${NINFER_BINARY_URL:-https://github.com/${GITHUB_REPO}/releases/download/${RELEASE_TAG}/${BINARY_ASSET}}"
BUILD_FROM_SOURCE="${NINFER_BUILD_FROM_SOURCE:-0}"

MODEL_REPO="${NINFER_MODEL_REPO:-neroued/Qwen3.8-27B-nvfp4-NInfer}"
MODEL_FILE="${NINFER_MODEL_FILE:-qwen3_8_27b_nvfp4.ninfer}"
MODEL_ID="${NINFER_MODEL_ID:-qwen3.8-27b}"
PORT="${NINFER_PORT:-8010}"
MAX_CONTEXT="${NINFER_MAX_CONTEXT:-65536}"
CONTEXT_CAPACITY="${NINFER_CONTEXT_CAPACITY:-auto}"
MAX_CONCURRENCY="${NINFER_MAX_CONCURRENCY:-8}"
MAX_PENDING="${NINFER_MAX_PENDING:-64}"
KV_DTYPE="${NINFER_KV_DTYPE:-fp8}"
PREFILL_CHUNK="${NINFER_PREFILL_CHUNK:-2048}"
EXTRA_ARGS="${NINFER_EXTRA_ARGS:-}"

mkdir -p "$WORKDIR/bin" "$WORKDIR/models" "$WORKDIR/logs"
cd "$WORKDIR"

echo "=== [1/5] GPU & host inventory ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
free -h

echo "=== [2/5] Starting model download in background (${MODEL_REPO}/${MODEL_FILE}) ==="
export DEBIAN_FRONTEND=noninteractive
python3 -m pip install -q --upgrade huggingface_hub >/dev/null 2>&1 || true
(
  python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='${MODEL_REPO}', filename='${MODEL_FILE}', local_dir='${WORKDIR}/models')
print('MODEL_DOWNLOADED=' + path, flush=True)
"
) >"$WORKDIR/logs/model-download.log" 2>&1 &
DL_PID=$!
echo "Model download started (pid=$DL_PID, log=$WORKDIR/logs/model-download.log)"

SERVE_BIN=""
LIB_DIR=""

if [[ "$BUILD_FROM_SOURCE" != "1" ]]; then
  echo "=== [3/5] Checking for precompiled ninfer-serve (${BINARY_URL}) ==="
  TARBALL="$WORKDIR/${BINARY_ASSET}"
  if curl -fL --retry 2 --connect-timeout 15 -o "$TARBALL" "$BINARY_URL" >"$WORKDIR/logs/binary-download.log" 2>&1; then
    rm -rf "$WORKDIR/prebuilt"
    mkdir -p "$WORKDIR/prebuilt"
    tar -xzf "$TARBALL" -C "$WORKDIR/prebuilt"
    if [[ -x "$WORKDIR/prebuilt/ninfer-serve" ]]; then
      SERVE_BIN="$WORKDIR/prebuilt/ninfer-serve"
      [[ -d "$WORKDIR/prebuilt/lib" ]] && LIB_DIR="$WORKDIR/prebuilt/lib"
      echo "Using precompiled binary: $SERVE_BIN"
      [[ -f "$WORKDIR/prebuilt/build-info.json" ]] && cat "$WORKDIR/prebuilt/build-info.json"
      # Install only lightweight runtime shared libs if missing on the Colab image.
      apt-get update -qq
      apt-get install -y -qq --no-install-recommends \
        libhwloc15 \
        liburing2 \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libswscale7 \
        libcurl4 \
        libjpeg8 \
        >"$WORKDIR/logs/apt-runtime.log" 2>&1 || true
    else
      echo "Downloaded archive did not contain executable ninfer-serve; falling back to source build."
    fi
  else
    echo "No precompiled binary available at ${BINARY_URL}; falling back to source build."
  fi
fi

if [[ -z "$SERVE_BIN" ]]; then
  echo "=== [3/5] Installing CUDA 13.3 toolkit & building latest ${GITHUB_REPO} (${REPO_BRANCH}) from source ==="
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    cuda-toolkit-13-3 \
    cmake \
    ninja-build \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libcurl4-openssl-dev \
    libhwloc-dev \
    liburing-dev \
    libjpeg-dev \
    >"$WORKDIR/logs/apt.log" 2>&1

  export PATH="/usr/local/cuda-13.3/bin:/usr/local/cuda/bin:$PATH"
  export CUDACXX="/usr/local/cuda-13.3/bin/nvcc"
  if [[ ! -x "$CUDACXX" ]]; then
    CUDACXX="$(command -v nvcc)"
  fi
  echo "Using nvcc: $CUDACXX ($($CUDACXX --version | tail -n 1))"

  if [[ -d "$WORKDIR/ninfer/.git" ]]; then
    git -C "$WORKDIR/ninfer" fetch --depth 1 origin "$REPO_BRANCH"
    git -C "$WORKDIR/ninfer" checkout -B "$REPO_BRANCH" "origin/$REPO_BRANCH"
  else
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR/ninfer"
  fi
  echo "NInfer commit: $(git -C "$WORKDIR/ninfer" rev-parse --short HEAD)"

  cmake -S "$WORKDIR/ninfer" -B "$WORKDIR/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DNINFER_TARGET=rtx5090 \
    -DNINFER_BUILD_APPS=ON \
    -DBUILD_TESTING=OFF \
    -DNINFER_BUILD_BENCHMARKS=OFF \
    >"$WORKDIR/logs/cmake-config.log" 2>&1

  cmake --build "$WORKDIR/build" --parallel --target ninfer-serve >"$WORKDIR/logs/cmake-build.log" 2>&1
  SERVE_BIN="$WORKDIR/build/apps/ninfer-serve"
  echo "Build complete: $SERVE_BIN"
fi

echo "=== [4/5] Waiting for model download (pid=$DL_PID) ==="
if ! wait "$DL_PID"; then
  echo "Model download failed; log:"
  cat "$WORKDIR/logs/model-download.log"
  exit 1
fi
tail -n 5 "$WORKDIR/logs/model-download.log"
MODEL_PATH="$WORKDIR/models/$MODEL_FILE"
ls -lh "$MODEL_PATH"

echo "=== [5/5] Starting ninfer-serve on 127.0.0.1:${PORT} ==="
if [[ -f "$WORKDIR/ninfer-serve.pid" ]]; then
  OLD_PID="$(cat "$WORKDIR/ninfer-serve.pid" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]]; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
  fi
fi

export LD_LIBRARY_PATH="${LIB_DIR:+${LIB_DIR}:}/usr/local/cuda-13.3/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# shellcheck disable=SC2086
nohup "$SERVE_BIN" "$MODEL_PATH" \
  --max-context "$MAX_CONTEXT" \
  --context-capacity "$CONTEXT_CAPACITY" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --max-pending-requests "$MAX_PENDING" \
  --prefill-chunk "$PREFILL_CHUNK" \
  --kv-dtype "$KV_DTYPE" \
  --model-id "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --desktop-reserve 0B \
  $EXTRA_ARGS \
  >"$WORKDIR/logs/ninfer-serve.log" 2>&1 </dev/null &
SERVE_PID=$!
echo "$SERVE_PID" >"$WORKDIR/ninfer-serve.pid"
echo "ninfer-serve started (pid=$SERVE_PID, bin=$SERVE_BIN)"

for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >"$WORKDIR/logs/models.json"; then
    echo "READY: $(cat "$WORKDIR/logs/models.json")"
    exit 0
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "ninfer-serve exited early; log:"
    tail -n 60 "$WORKDIR/logs/ninfer-serve.log"
    exit 1
  fi
  sleep 2
done

echo "Timed out waiting for ninfer-serve; log:"
tail -n 60 "$WORKDIR/logs/ninfer-serve.log"
exit 1
