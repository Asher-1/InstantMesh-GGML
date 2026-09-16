#!/usr/bin/env python3
"""Compare staged VAE activations: C++ dumps (/tmp/vae_<stage>.bin) vs torch
references (/tmp/ref_<stage>.bin). Both sides are torch [C,H,W] ordered (the
C++ dumps are captured via dump_torch()).

Run with any numpy python from the repo root:
  python3 cpp_ggml/convert/compare_vae_stages.py
"""
import glob
import os
import re

import numpy as np

TMP = "/tmp"


def load(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def main() -> int:
    refs = sorted(glob.glob(os.path.join(TMP, "ref_*.bin")))
    if not refs:
        print("no /tmp/ref_*.bin — run cpp_ggml/convert/dump_vae_stages.py first")
        return 1
    worst = []
    nbad = 0
    for rp in refs:
        name = os.path.basename(rp)[len("ref_"):-len(".bin")]
        cp = os.path.join(TMP, f"vae_{name}.bin")
        if not os.path.exists(cp):
            print(f"{name:60s} C++ dump missing")
            continue
        a = load(rp)
        b = load(cp)
        if a.size != b.size:
            print(f"{name:60s} SIZE mismatch ref={a.size} cpp={b.size}")
            nbad += 1
            continue
        d = np.abs(a - b)
        m = d.max()
        worst.append((m, name, a.size))
        ok = "PASS" if m < 3e-3 else "FAIL"
        if m >= 3e-3:
            nbad += 1
        print(f"{ok} {name:56s} max_abs={m:.3e} mean_abs={d.mean():.3e} n={a.size}")
    if worst:
        worst.sort(reverse=True)
        print("\nworst stages:")
        for m, name, n in worst[:8]:
            print(f"  {m:.3e}  {name}")
    print(f"\n{len(worst)} compared, {nbad} over threshold")
    return 1 if nbad else 0


if __name__ == "__main__":
    raise SystemExit(main())
