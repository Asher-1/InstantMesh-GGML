#!/usr/bin/env python3
"""Generate CLIP vision encoder parity fixtures for tests/test_clip_vision.cpp.

Dumps a deterministic pseudo-image (CLIP-normalized space, as produced by
feature_extractor_clip) plus the fp32 transformers reference image_embeds:

  fixtures/clip_vision/input.bin    [1*3*224*224] float32
  fixtures/clip_vision/embeds.bin   [1*1024]      float32

Usage: python3 convert/parity_clip_vision.py [--snapshot DIR] [--out benchmarks/fixtures]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch


def locate_snapshot(arg: str) -> str:
    if arg and os.path.isdir(arg):
        return arg
    for pat in ("~/.cache/huggingface/hub/models--sudo-ai--zero123plus-v1.2/snapshots/*",):
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            if os.path.isdir(os.path.join(d, "vision_encoder")):
                return d
    raise FileNotFoundError("zero123plus-v1.2 snapshot not found; pass --snapshot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=str, default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "benchmarks", "fixtures"))
    args = ap.parse_args()

    snap = locate_snapshot(args.snapshot)
    out_dir = os.path.join(args.out, "clip_vision")
    os.makedirs(out_dir, exist_ok=True)

    from transformers import CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained(
        os.path.join(snap, "vision_encoder"), torch_dtype=torch.float32).eval()

    rng = np.random.default_rng(42)
    img = rng.standard_normal(1 * 3 * 224 * 224, dtype=np.float32) * 0.5
    img.tofile(os.path.join(out_dir, "input.bin"))

    x = torch.from_numpy(img).reshape(1, 3, 224, 224)
    with torch.no_grad():
        embeds = model(x).image_embeds[0]  # [1024]
    embeds.numpy().tofile(os.path.join(out_dir, "embeds.bin"))

    meta = {"dim": int(embeds.numel()), "abs_mean": float(embeds.abs().mean())}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
