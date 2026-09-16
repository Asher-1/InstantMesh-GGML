#!/usr/bin/env python3
"""Generate VAE parity fixtures for tests/test_vae.cpp (diffusers reference).

  fixtures/vae/enc_in.bin    [1*3*512*512]  encoder input  (fx_vae normalized space)
  fixtures/vae/enc_lat.bin   [1*4*64*64]    encoder posterior mode
  fixtures/vae/dec_lat.bin   [1*4*80*120]   decoder input  (pipeline-scale latents)
  fixtures/vae/dec_img.bin   [1*3*640*960]  decoder output (still scaled: /0.5*0.8 pending)

Run with the reference venv (diffusers needs a newer transformers than the
pinned main env):  /tmp/vref/bin/python convert/parity_vae.py
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
            if os.path.isdir(os.path.join(d, "vae")):
                return d
    raise FileNotFoundError("zero123plus-v1.2 snapshot not found; pass --snapshot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=str, default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "benchmarks", "fixtures"))
    args = ap.parse_args()
    snap = locate_snapshot(args.snapshot)
    out_dir = os.path.join(args.out, "vae")
    os.makedirs(out_dir, exist_ok=True)

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(os.path.join(snap, "vae"), torch_dtype=torch.float32).eval()

    rng = np.random.default_rng(42)
    # encoder: 512x512 in the fx_vae normalized domain (mean .5 / std .8).
    enc_in = (rng.standard_normal(1 * 3 * 512 * 512, dtype=np.float32) * 0.15).astype(np.float32)
    enc_in.tofile(os.path.join(out_dir, "enc_in.bin"))
    with torch.no_grad():
        post = vae.encode(torch.from_numpy(enc_in).reshape(1, 3, 512, 512)).latent_dist
        lat = post.mode()[0]
    lat.numpy().tofile(os.path.join(out_dir, "enc_lat.bin"))

    # decoder: 80x120 pipeline-scale latents (typical denoising magnitudes).
    dec_lat = (rng.standard_normal(1 * 4 * 80 * 120, dtype=np.float32) * 1.2).astype(np.float32)
    dec_lat.tofile(os.path.join(out_dir, "dec_lat.bin"))
    with torch.no_grad():
        img = vae.decode(torch.from_numpy(dec_lat).reshape(1, 4, 80, 120) / vae.config.scaling_factor,
                         return_dict=False)[0][0]
    img.numpy().tofile(os.path.join(out_dir, "dec_img.bin"))

    meta = {
        "enc_lat_abs_mean": float(lat.abs().mean()),
        "dec_img_abs_mean": float(img.abs().mean()),
        "dec_img_range": [float(img.min()), float(img.max())],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
