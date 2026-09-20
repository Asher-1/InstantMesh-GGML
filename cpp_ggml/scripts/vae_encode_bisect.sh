#!/usr/bin/env bash
# VAE encode parity bisection: dump per-stage tensors via IM_VAE_DUMP on the
# CPU (reference) and a GPU backend, then report the first stage where the GPU
# path diverges beyond the fp32 rounding level (~1e-4 through the deep stack).
#
#   usage: bash scripts/vae_encode_bisect.sh [device]      # default: cuda
#          bash scripts/vae_encode_bisect.sh vulkan
#
# Requires build-gpu (dual-backend build). Requires `./build-gpu/test_vae` to
# pass on the CPU so the fixtures themselves are sane (SKIP(77) = fixtures
# missing; regenerate with convert/dump_vae_stages.py in the torch venv).
#
# Context (2026-09-19): after the TF32 fix (patches/ggml-cuda-f32-matmul-
# exact.patch), test_vae encode on CUDA still shows max 1.45e-2 vs the torch
# fp32 reference (CPU 2.4e-4 / Vulkan 1.3e-4) — a second, unlocated error
# source. This script reproduces the tap workflow for bisecting it
# (docs/ALIGNMENT.md "End-to-end vs PyTorch full-matrix acceptance", issue 3).
set -euo pipefail

DEV="${1:-cuda}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # cpp_ggml/
cd "$ROOT"

FIX="$PWD/benchmarks/fixtures/vae"
MODEL="$PWD/models/gguf/zero123pp_vae_f32.gguf"
OUT=/tmp/vaedbg_bisect
rm -rf "$OUT"
mkdir -p "$OUT/cpu" "$OUT/$DEV"

run() { # <device> <outdir>
  rm -f /tmp/vae_*.bin
  local rc=0
  IM_VAE_DUMP=1 VAE_FIXTURE_DIR="$FIX" VAE_MODEL="$MODEL" \
    ./build-gpu/test_vae "$1" > "$OUT/log_$1.txt" 2>&1 || rc=$?
  # A FAIL verdict (non-zero exit) does not invalidate the dumps — the bisect
  # is precisely how we localize a known-failing stage, so keep going.
  if [ "$rc" -ne 0 ]; then
    echo "note: test_vae $1 exited $rc (parity verdict FAIL — see $OUT/log_$1.txt)"
  fi
  if ! mv /tmp/vae_*.bin "$2/" 2>/dev/null; then
    echo "error: no dumps produced for $1 — see $OUT/log_$1.txt"; exit 1
  fi
}

echo "== dumping CPU reference =="
run cpu "$OUT/cpu"
echo "== dumping $DEV =="
run "$DEV" "$OUT/$DEV"

python3 - "$OUT" "$DEV" <<'EOF'
import os, sys
import numpy as np
out, dev = sys.argv[1], sys.argv[2]
stages = sorted(os.listdir(f'{out}/cpu'))
if not stages:
    sys.exit('no dumps produced — check /tmp/vaedbg_bisect/log_cpu.txt')
first = None
for f in stages:
    p = f'{out}/{dev}/{f}'
    if not os.path.exists(p):
        print(f'{f:<36} MISSING on {dev}')
        continue
    a = np.fromfile(f'{out}/cpu/{f}', np.float32)
    b = np.fromfile(p, np.float32)
    if a.size != b.size:
        print(f'{f:<36} SIZE MISMATCH ({a.size} vs {b.size})')
        continue
    d = np.abs(a - b)
    mark = ''
    if d.max() > 1e-4 and first is None:
        first = f
        mark = '   <-- FIRST DIVERGENCE'
    print(f'{f:<36} max {d.max():.3e}  mean {d.mean():.3e}{mark}')
print()
print('first divergent stage:', first or '(none — all within fp32 rounding level)')
print('next step: set the resnet dbg filter in src/models/vae.cpp to the block')
print('before that stage and re-run to narrow to a single op.')
EOF
