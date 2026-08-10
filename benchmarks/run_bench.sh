#!/usr/bin/env bash
# End-to-end benchmark for the InstantMesh ggml port.
#
#   usage: bash benchmarks/run_bench.sh
#
# For each {backend x precision x image} it runs the full image->mesh pipeline
# once on the target build and records wall time + dumps the raw SDF grid
# (a deterministic geometry fingerprint) for precision comparison.
#
# Backends are selected by dedicated build dirs so that each build exposes
# exactly one GPU/CPU backend (avoiding registry-order ambiguity):
#   cpu    -> build       (CPU-only)
#   cuda   -> build-cuda  (CUDA-only)
#   vulkan -> build-vk    (Vulkan-only)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=benchmarks/results
mkdir -p "$OUT"
LOG="$OUT/bench.log"
: > "$LOG"

declare -A BIN=(
  [cpu]="build/instantmesh"
  [cuda]="build-cuda/instantmesh"
  [vulkan]="build-vk/instantmesh"
)
declare -A DEV=(
  [cpu]="cpu"
  [cuda]="gpu"
  [vulkan]="gpu"
)
declare -A PREFIX=( [f32]="" [f16]="" [q8]="" )

BACKENDS="cpu cuda vulkan"
PRECISIONS="f32 f16 q8"
# image dirs under benchmarks/mv/<name> containing image.bin + camera.bin
IMAGES="$(ls benchmarks/mv 2>/dev/null)"

if [ -z "$IMAGES" ]; then
  echo "no multiview inputs under benchmarks/mv — run gen first" | tee -a "$LOG"
  exit 1
fi

run_one() { # backend precision image
  local b="$1" p="$2" img="$3"
  local bin="${BIN[$b]}" dev="${DEV[$b]}"
  local base="dino_${p}.gguf"
  local sdf="$OUT/${img}__${b}__${p}.sdf.bin"
  local obj="$OUT/${img}__${b}__${p}.obj"
  local t0 t1 dt
  t0=$(date +%s.%N)
  "$bin" --dino "models/gguf/dino_${p}.gguf" \
         --transformer "models/gguf/lrm_transformer_${p}.gguf" \
         --synthesizer "models/gguf/synthesizer_${p}.gguf" \
         --image "benchmarks/mv/$img/image.bin" \
         --camera "benchmarks/mv/$img/camera.bin" \
         --grid-res 88 --out "$obj" --dump-sdf "$sdf" \
         --device "$dev" >> "$LOG" 2>&1
  local rc=$?
  t1=$(date +%s.%N)
  dt=$(echo "$t1 $t0" | awk '{printf "%.3f", $1-$2}')
  echo "$img,$b,$p,$dt,$rc" | tee -a "$OUT/times.csv"
}

echo "image,backend,precision,wall_s,rc" > "$OUT/times.csv"
for img in $IMAGES; do
  for b in $BACKENDS; do
    for p in $PRECISIONS; do
      run_one "$b" "$p" "$img"
    done
  done
done
echo "DONE"
