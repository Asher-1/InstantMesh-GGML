#!/usr/bin/env python3
"""Compare the zero123pp end-to-end output against the official fixture set.

Fixture dir (from cpp_ggml/convert/dump_e2e.py):
  benchmarks/fixtures/e2e/<name>/ref_latents.bin   final raw latents [1*4*120*80]
  benchmarks/fixtures/e2e/<name>/latents_%03d.bin   raw latents after each scheduler step
  benchmarks/fixtures/e2e/<name>/ref_grid.png       official 3x2 output grid (960x640)

C++ side:
  zero123pp --fixture-dir <dir> --dump-final-latents <out>.bin --dump-steps <dir2> --out <outdir>
  → grid.png + view_%d.png under <outdir>

Run (any numpy python, repo root):
  python3 cpp_ggml/convert/compare_e2e.py <fixture_dir> \
      [--cpp-latents <out>.bin] [--cpp-out <outdir>] [--steps 75]
"""
import argparse
import glob
import os

import numpy as np

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


def load(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def psnr(a: np.ndarray, b: np.ndarray, vmax: float = 1.0) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(vmax * vmax / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture_dir")
    ap.add_argument("--cpp-latents", default="", help="C++ --dump-final-latents output")
    ap.add_argument("--cpp-out", default="", help="C++ --out dir (grid.png/view_*.png)")
    ap.add_argument("--steps", type=int, default=75)
    args = ap.parse_args()

    nbad = 0
    lat_n = 4 * 120 * 80

    # ── final raw latents ────────────────────────────────────────────────
    ref = os.path.join(args.fixture_dir, "ref_latents.bin")
    if args.cpp_latents and os.path.exists(ref) and os.path.exists(args.cpp_latents):
        a, b = load(ref), load(args.cpp_latents)
        if a.size != b.size:
            print(f"latents SIZE mismatch ref={a.size} cpp={b.size}"); nbad += 1
        else:
            d = np.abs(a - b)
            m = float(d.max())
            rng = float(np.abs(a).max()) + 1e-9
            ok = "PASS" if m < 2e-2 else "FAIL"   # raw latents ~ O(1); f32 ref
            if m >= 2e-2:
                nbad += 1
            print(f"{ok} final latents : max_abs={m:.3e} mean_abs={d.mean():.3e} "
                  f"PSNR={psnr(a, b, rng):.2f}dB (range {rng:.2f})")
    else:
        print("skip latents: give --cpp-latents and a fixture with ref_latents.bin")

    # ── per-step latents curve (optional) ────────────────────────────────
    if args.cpp_out:
        cpp_steps = sorted(glob.glob(os.path.join(args.cpp_out, "latents_*.bin")))
        worst = []
        for i in range(args.steps):
            rf = os.path.join(args.fixture_dir, f"latents_{i:03d}.bin")
            cf = os.path.join(args.cpp_out, f"latents_{i:03d}.bin")
            if not (os.path.exists(rf) and os.path.exists(cf)):
                continue
            a, b = load(rf), load(cf)
            if a.size == b.size == lat_n:
                worst.append((float(np.abs(a - b).max()), i))
        if worst:
            worst.sort(reverse=True)
            print(f"per-step latents: {len(worst)} steps compared, "
                  f"worst step {worst[0][1]}: max_abs={worst[0][0]:.3e}")

    # ── output grid PSNR ─────────────────────────────────────────────────
    if HAVE_PIL and args.cpp_out:
        refg = os.path.join(args.fixture_dir, "ref_grid.png")
        cppg = os.path.join(args.cpp_out, "grid.png")
        if os.path.exists(refg) and os.path.exists(cppg):
            ra = np.asarray(Image.open(refg).convert("RGB"), dtype=np.float32) / 255.0
            ca = np.asarray(Image.open(cppg).convert("RGB"), dtype=np.float32) / 255.0
            if ra.shape != ca.shape:
                print(f"grid SIZE mismatch ref={ra.shape} cpp={ca.shape}"); nbad += 1
            else:
                p = psnr(ra, ca)
                m = float(np.abs(ra - ca).max())
                ok = "PASS" if p >= 25.0 else "FAIL"   # ~25dB is clearly visible-grade match
                if p < 25.0:
                    nbad += 1
                print(f"{ok} grid.png    : PSNR={p:.2f}dB max_abs={m:.3f}")
                for i, fn in enumerate(sorted(glob.glob(os.path.join(args.cpp_out, "view_*.png")))):
                    rv = np.asarray(Image.open(os.path.join(
                        args.fixture_dir, f"ref_view_{i}.png")).convert("RGB"),
                        dtype=np.float32) / 255.0
                    cv = np.asarray(Image.open(fn).convert("RGB"), dtype=np.float32) / 255.0
                    print(f"    view_{i}: PSNR={psnr(rv, cv):.2f}dB")
        else:
            print("skip grid: give --cpp-out and a fixture with ref_grid.png")

    print(f"\n{'ALL PASS' if nbad == 0 else str(nbad) + ' FAIL'}")
    return 1 if nbad else 0


if __name__ == "__main__":
    raise SystemExit(main())
