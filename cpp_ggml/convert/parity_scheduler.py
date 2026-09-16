#!/usr/bin/env python3
"""Generate scheduler parity fixtures for tests/test_scheduler.cpp.

Dumps the exact float32 numeric trace of diffusers'
EulerAncestralDiscreteScheduler configured like the official InstantMesh
run.py (trailing timestep spacing) for num_inference_steps=75:

  fixtures/scheduler/timesteps.bin   [75]  float32
  fixtures/scheduler/sigmas.bin      [76]  float32 (incl. trailing 0)
  fixtures/scheduler/x0.bin          [4*80*120]
  fixtures/scheduler/noise_init.bin  [4*80*120]   (initial latent noise)
  fixtures/scheduler/eps.bin         [75*4*80*120] (fake unet outputs, fixed)
  fixtures/scheduler/anc_noise.bin   [75*4*80*120] (ancestral noise, fixed)
  fixtures/scheduler/trace.bin       [75] records of (sigma, sigma_up, sigma_down)
  fixtures/scheduler/x_final.bin     [4*80*120]

Usage: python3 convert/parity_scheduler.py [--out benchmarks/fixtures]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from diffusers import EulerAncestralDiscreteScheduler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "benchmarks", "fixtures"))
    ap.add_argument("--steps", type=int, default=75)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, "scheduler")
    os.makedirs(out_dir, exist_ok=True)

    sched = EulerAncestralDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",          # zero123plus-v1.2 scheduler_config.json
        prediction_type="v_prediction",  # ditto; run.py only overrides spacing
        steps_offset=1,
        timestep_spacing="trailing",     # run.py override
    )
    # torch.cumprod's float32 rounding is not reproducible bit-exactly in
    # portable C++, so the curve travels with the model (GGUF KV at conversion
    # time) and is injected into the C++ scheduler. Dump it here for tests.
    sched.alphas_cumprod.numpy().tofile(os.path.join(out_dir, "alphas_cumprod.bin"))
    sched.set_timesteps(args.steps)

    ts = sched.timesteps.numpy().astype(np.float32)
    sg = sched.sigmas.numpy().astype(np.float32)
    ts.tofile(os.path.join(out_dir, "timesteps.bin"))
    sg.tofile(os.path.join(out_dir, "sigmas.bin"))

    n = 4 * 80 * 120
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(n, dtype=np.float32)
    noise_init = rng.standard_normal(n, dtype=np.float32)
    eps = rng.standard_normal((args.steps, n), dtype=np.float32)
    anc = rng.standard_normal((args.steps, n), dtype=np.float32)
    x0.tofile(os.path.join(out_dir, "x0.bin"))
    noise_init.tofile(os.path.join(out_dir, "noise_init.bin"))
    eps.tofile(os.path.join(out_dir, "eps.bin"))
    anc.tofile(os.path.join(out_dir, "anc_noise.bin"))

    x0_t = torch.from_numpy(x0).reshape(1, 4, 80, 120)
    noise_init_t = torch.from_numpy(noise_init).reshape(1, 4, 80, 120)

    trace = np.zeros((args.steps, 3), dtype=np.float32)

    # initial latent: x = x0 + noise*sigma(timesteps[0])   (RefOnly cond path uses add_noise; the
    # diffusion loop starts from randn*sigma_max — here we mirror our C++ runner:
    # x_init = add_noise(cond_lat-ish x0) then walk the schedule).
    x = sched.add_noise(x0_t, noise_init_t, torch.tensor([ts[0]])).float()
    x_np = x.reshape(-1).numpy().copy()
    x_np.tofile(os.path.join(out_dir, "x_init.bin"))

    xt = x
    for i in range(args.steps):
        sigma = float(sched.sigmas[i])
        sigma_to = float(sched.sigmas[i + 1])
        model_out = torch.from_numpy(eps[i]).reshape(1, 4, 80, 120)
        scaled = sched.scale_model_input(xt, torch.tensor(ts[i]))
        if i == 0:
            scaled.reshape(-1).numpy().tofile(os.path.join(out_dir, "x_scaled0.bin"))
        # Manual ancestral Euler step (verbatim math from diffusers step(),
        # with our fixed noise instead of its internal randn) so that the
        # C++ port — which takes external noise — is compared 1:1.
        sigma_up = float((sigma_to ** 2 * (sigma ** 2 - sigma_to ** 2) / sigma ** 2) ** 0.5)
        sigma_down = float((sigma_to ** 2 - sigma_up ** 2) ** 0.5)
        denom = sigma ** 2 + 1.0
        pred_x0 = model_out * (-sigma / (denom ** 0.5)) + xt / denom   # v_prediction
        derivative = (xt - pred_x0) / sigma
        xt = (xt + derivative * (sigma_down - sigma)
              + torch.from_numpy(anc[i]).reshape(1, 4, 80, 120) * sigma_up).float()
        trace[i, 0] = sigma
        trace[i, 1] = sigma_up
        trace[i, 2] = sigma_down

    xt.reshape(-1).numpy().tofile(os.path.join(out_dir, "x_final.bin"))
    trace.tofile(os.path.join(out_dir, "trace.bin"))

    meta = {
        "steps": args.steps,
        "n": n,
        "timesteps_head": ts[:4].tolist(),
        "sigmas_head": sg[:4].tolist(),
        "x_final_abs_mean": float(np.abs(xt.reshape(-1).numpy()).mean()),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
