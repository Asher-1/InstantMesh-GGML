"""Batch-convert all InstantMesh models to GGUF in all three precisions.

Every model is emitted as f32 / f16 / q8 so the C++ runtime can pick the best
trade-off on any backend (CPU / CUDA / Vulkan) while keeping the SAME weights
for parity and cross-backend comparison.

Usage:
    python -m convert.convert_all \
        --lrm ckpts/instant_mesh_large.ckpt \
        --out-dir models/gguf
"""

import argparse
import logging
import os
import subprocess
import sys

PRECISIONS = ["f32", "f16", "q8"]


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Convert all InstantMesh models to GGUF (f32/f16/q8)")
    ap.add_argument("--lrm", type=str, required=True,
                    help="Path to the LRM reconstruction checkpoint (.ckpt)")
    ap.add_argument("--out-dir", type=str, default="models/gguf")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for precision in PRECISIONS:
        logging.info("== conversion precision: %s ==", precision)
        subprocess.check_call([
            sys.executable, "-m", "convert.convert_lrm",
            args.lrm,
            "--out-dir", args.out_dir,
            "--precision", precision,
            "--prefix", "lrm_generator.",
        ])

    logging.info("done. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()