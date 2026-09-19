#!/usr/bin/env bash
# Remove stale baselines superseded by the refreshed baseline set
# (benchmarks/results/{img}__{backend}__{prec}.{sdf.bin,obj} @ .rev sidecars,
# see docs/ALIGNMENT.md "回归基线管理规范" 基线台账).
#
#   usage: bash benchmarks/clean_stale_baselines.sh [--dry-run]
#
# --dry-run lists what would be removed without deleting anything.
set -u
cd "$(dirname "$0")/.."
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# 1. old bench drafts (superseded by the refreshed {img}__{dev}__{p} set)
# 2. legacy SDF baseline collection in results/sdf/ (pre-sidecar, old code)
STALE=()
for f in benchmarks/results/_bench_*.obj benchmarks/results/sdf/*.sdf.bin; do
    [ -f "$f" ] && STALE+=("$f")
done

if [ ${#STALE[@]} -eq 0 ]; then
    echo "nothing stale to remove"
    exit 0
fi

if [ "$DRY" = 1 ]; then
    printf '%s\n' "${STALE[@]}"
    echo "-- dry run: ${#STALE[@]} file(s) would be removed"
    exit 0
fi

rm -v "${STALE[@]}"
rmdir benchmarks/results/sdf 2>/dev/null && echo "removed empty dir benchmarks/results/sdf"
echo "removed ${#STALE[@]} stale file(s)"
