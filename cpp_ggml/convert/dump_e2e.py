#!/usr/bin/env python3
"""Generate deterministic end-to-end fixtures for the zero123pp CLI.

Runs the OFFICIAL Zero123PlusPipeline (fp16 weights + InstantMesh white-bg
unet override, trailing EulerAncestral scheduler, 75 steps by default) on a
given input image, recording every torch.randn draw so the C++ CLI can replay
the exact same noise via --fixture-dir.

Fixtures (benchmarks/fixtures/e2e/<name>/):
  img_vae.bin          [1*3*512*512]  fx_vae preprocessed input (fp32)
  img_clip.bin         [1*3*224*224]  fx_clip preprocessed input (fp32)
  cond_pos.bin         [1*4*64*64]    VAE posterior mode of img_vae
  cond_neg.bin         [1*4*64*64]    VAE posterior mode of zeros
  latents0.bin         [1*4*120*80]   initial raw randn
  cond_noise_%03d.bin  [2*4*64*64]    w-pass cond noise, per step
  step_noise_%03d.bin  [1*4*120*80]   ancestral randn, per step
  latents_%03d.bin     [1*4*120*80]   raw latents after each step
  ref_latents.bin      [1*4*120*80]   final raw latents (before unscale)
  r_input_%03d.bin     [2*4*120*80]   r-pass unet sample (scaled model input)
  w_input_%03d.bin     [2*4*64*64]    w-pass unet sample (scaled noisy cond)
  context_%03d.bin     [2*77*1024]    unet encoder_hidden_states, per step
  eps_%03d.bin         [2*4*120*80]   raw unet noise prediction (pre-CFG)
  ref_grid.png                       official 3x2 output grid
  ref_view_%d.png                    official per-view crops

Usage (from repo root, reference venv):
  /tmp/vref/bin/python -m cpp_ggml.convert.dump_e2e --image examples/cute_horse.jpg \
      --name cute_horse [--steps 75] [--seed 42] [--dtype fp16]
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def locate_snapshot(arg: str) -> str:
    if arg and os.path.isdir(arg):
        return arg
    for pat in ("~/.cache/huggingface/hub/models--sudo-ai--zero123plus-v1.2/snapshots/*",):
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            if os.path.isdir(os.path.join(d, "vae")):
                return d
    raise FileNotFoundError("zero123plus-v1.2 snapshot not found; pass --snapshot")


def locate_unet_override(arg: str) -> str:
    if arg and os.path.isfile(arg):
        return arg
    pats = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--TencentARC--InstantMesh/snapshots/*/"
        "diffusion_pytorch_model.bin"))
    if pats:
        return pats[0]
    raise FileNotFoundError("InstantMesh white-bg unet override not found; pass --unet-override")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--snapshot", type=str, default="")
    ap.add_argument("--unet-override", type=str, default="")
    ap.add_argument("--steps", type=int, default=75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "benchmarks", "fixtures"))
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join(args.out, "e2e", name)
    os.makedirs(out_dir, exist_ok=True)
    dt = torch.float16 if args.dtype == "fp16" else torch.float32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from diffusers import AutoencoderKL, UNet2DConditionModel, EulerAncestralDiscreteScheduler
    from transformers import (
        CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection,
    )
    from zero123plus.pipeline import Zero123PlusPipeline, to_rgb_image

    snap = locate_snapshot(args.snapshot)
    ramp = json.load(open(os.path.join(snap, "model_index.json")))["ramping_coefficients"]

    vae = AutoencoderKL.from_pretrained(os.path.join(snap, "vae"), torch_dtype=dt).eval()
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(snap, "text_encoder"), torch_dtype=dt).eval()
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    unet = UNet2DConditionModel.from_pretrained(os.path.join(snap, "unet"), torch_dtype=dt).eval()
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(os.path.join(snap, "scheduler"))
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        os.path.join(snap, "vision_encoder"), torch_dtype=dt).eval()
    fex_clip = CLIPImageProcessor.from_pretrained(os.path.join(snap, "feature_extractor_clip"))
    fex_vae = CLIPImageProcessor.from_pretrained(os.path.join(snap, "feature_extractor_vae"))

    # InstantMesh white-background finetune overrides every unet key.
    sd = torch.load(locate_unet_override(args.unet_override), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    unet.load_state_dict(sd, strict=True)
    print("unet: white-bg override applied")

    pipe = Zero123PlusPipeline(vae, text_encoder, tokenizer, unet, scheduler,
                               vision_encoder, fex_clip, fex_vae,
                               ramping_coefficients=ramp)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    pipe = pipe.to("cpu")

    # C++ uses the posterior MODE (vae_encode_mode); drop the .sample() draws
    # so the recorded randn stream lines up 1:1 with the C++ fixture contract.
    pipe.encode_condition_image = lambda image: pipe.vae.encode(image).latent_dist.mode()

    # ── record every global-RNG randn draw ────────────────────────────────
    draws = []
    orig_randn = torch.randn
    orig_randn_like = torch.randn_like

    def rec_randn(*a, **k):
        t = orig_randn(*a, **k)
        draws.append(t.detach().float().cpu().numpy())
        return t

    def rec_randn_like(x, *a, **k):
        t = orig_randn_like(x, *a, **k)
        draws.append(t.detach().float().cpu().numpy())
        return t

    torch.randn = rec_randn
    torch.randn_like = rec_randn_like

    # ── capture the raw latents after every scheduler step ────────────────
    latents_trace = []
    orig_step = pipe.scheduler.step

    def step_wrap(model_output, sample, *a, **k):
        res = orig_step(model_output, sample, *a, **k)
        ps = res[0] if isinstance(res, tuple) else res.prev_sample
        latents_trace.append(ps.detach().float().cpu().numpy())
        return res

    pipe.prepare()  # wrap unet into RefOnlyNoisedUNet (idempotent after first)
    pipe.scheduler.step = step_wrap

    # ── capture per-step unet inputs (r-pass sample, w-pass sample, context) ─
    r_inputs, w_inputs, ctx_inputs, eps_inputs = [], [], [], []
    orig_fwd = pipe.unet.forward
    orig_fc = pipe.unet.forward_cond

    def fwd_hook(sample, timestep, encoder_hidden_states, *a, **k):
        r_inputs.append(sample.detach().float().cpu().numpy())
        ctx_inputs.append(encoder_hidden_states.detach().float().cpu().numpy())
        res = orig_fwd(sample, timestep, encoder_hidden_states, *a, **k)
        pred = res[0] if isinstance(res, tuple) else res
        eps_inputs.append(pred.detach().float().cpu().numpy())
        return res

    def fc_hook(noisy_cond_lat, timestep, encoder_hidden_states, *a, **k):
        w_inputs.append(noisy_cond_lat.detach().float().cpu().numpy())
        return orig_fc(noisy_cond_lat, timestep, encoder_hidden_states, *a, **k)

    pipe.unet.forward = fwd_hook
    pipe.unet.forward_cond = fc_hook

    # ── preprocessed inputs (identical to what the C++ feeds its encoders) ──
    from PIL import Image
    pil = to_rgb_image(Image.open(args.image))
    img_vae = fex_vae(images=pil, return_tensors="pt").pixel_values.float()
    img_clip = fex_clip(images=pil, return_tensors="pt").pixel_values.float()

    # ── run the official pipeline ─────────────────────────────────────────
    with torch.no_grad():
        out = pipe(pil, num_inference_steps=args.steps)
    grid = out.images[0]

    torch.randn = orig_randn
    torch.randn_like = orig_randn_like
    pipe.scheduler.step = orig_step
    pipe.unet.forward = orig_fwd
    pipe.unet.forward_cond = orig_fc

    # ── classify draws ────────────────────────────────────────────────────
    big = [d for d in draws if d.shape == (1, 4, 120, 80)]
    cond = [d for d in draws if d.shape == (2, 4, 64, 64)]
    print(f"draws: total={len(draws)} latents0+step_noise={len(big)} cond_noise={len(cond)}")
    assert len(big) == args.steps + 1, "unexpected [1,4,120,80] draw count"
    assert len(cond) == args.steps, "unexpected [2,4,64,64] draw count"
    assert len(latents_trace) == args.steps, "scheduler step count mismatch"
    assert len(r_inputs) == len(w_inputs) == len(ctx_inputs) == args.steps, "unet input capture mismatch"
    assert len(eps_inputs) == args.steps, "unet output capture mismatch"
    latents0, step_noises = big[0], big[1:]

    def put(fn, arr):
        np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(out_dir, fn))

    put("img_vae.bin", img_vae.numpy())
    put("img_clip.bin", img_clip.numpy())
    # cond latents from the SAME pipeline objects (posterior mode).
    with torch.no_grad():
        pos = pipe.vae.encode(img_vae.to(dt)).latent_dist.mode().float().cpu().numpy()
        neg = pipe.vae.encode(torch.zeros_like(img_vae).to(dt)).latent_dist.mode().float().cpu().numpy()
    put("cond_pos.bin", pos)
    put("cond_neg.bin", neg)
    put("latents0.bin", latents0)
    for i in range(args.steps):
        put(f"cond_noise_{i:03d}.bin", cond[i])
        put(f"step_noise_{i:03d}.bin", step_noises[i])
        put(f"latents_{i:03d}.bin", latents_trace[i])
        put(f"r_input_{i:03d}.bin", r_inputs[i])
        put(f"w_input_{i:03d}.bin", w_inputs[i])
        put(f"context_{i:03d}.bin", ctx_inputs[i])
        put(f"eps_{i:03d}.bin", eps_inputs[i])
    put("ref_latents.bin", latents_trace[-1])

    # reference images: grid + per-view crops (n=3 rows, m=2 cols)
    grid.save(os.path.join(out_dir, "ref_grid.png"))
    arr = np.asarray(grid)  # (960, 640, 3)
    for r in range(3):
        for c in range(2):
            view = arr[r * 320:(r + 1) * 320, c * 320:(c + 1) * 320, :]
            Image.fromarray(view).save(os.path.join(out_dir, f"ref_view_{r * 2 + c}.png"))
    print("wrote fixtures to", out_dir)


if __name__ == "__main__":
    main()
