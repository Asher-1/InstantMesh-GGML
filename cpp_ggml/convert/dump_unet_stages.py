#!/usr/bin/env python3
"""Generate UNet RefOnly parity fixtures using the OFFICIAL zero123plus
pipeline code (ReferenceOnlyAttnProc + a faithful re-run of RefOnlyNoisedUNet).

Run under /tmp/vref from the repo root:
  /tmp/vref/bin/python -m cpp_ggml.convert.dump_unet_stages

Fixtures (benchmarks/fixtures/unet/):
  unet_in.bin      [2,4,80,120]   r-pass sample   (batch = [uncond, cond])
  unet_ref_in.bin  [2,4,80,120]   w-pass noisy cond lat
  unet_ctx.bin     [2,77,1024]    cross-attention context
  unet_t.bin       [1]            integer timestep (e.g. 999)
  unet_eps_out.bin [2,4,80,120]   official r-pass output (fp32)
Plus staged activations under /tmp/ref_unet_*.bin for staged comparison.
"""
import glob
import json
import os

import numpy as np
import torch

from .convert_zero123pp import locate_snapshot


def main():
    torch.manual_seed(7)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "fixtures", "unet")
    os.makedirs(out_dir, exist_ok=True)

    snap = locate_snapshot("")
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(os.path.join(snap, "unet"),
                                                torch_dtype=torch.float32).eval()
    # InstantMesh white-background finetune override
    ckpt = os.environ.get(
        "UNET_OVERRIDE",
        glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--TencentARC--InstantMesh/snapshots/*/"
            "diffusion_pytorch_model.bin"))[0])
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    print("override missing:", len(missing), "unexpected:", len(unexpected))

    # Official RefOnly processor wrapping the fp32 default processor.
    # ReferenceOnlyAttnProc signature: (chained_proc, enabled, name).
    from zero123plus.pipeline import ReferenceOnlyAttnProc
    from diffusers.models.attention_processor import AttnProcessor
    procs = {name: ReferenceOnlyAttnProc(AttnProcessor(),
                                         enabled=name.endswith("attn1.processor"),
                                         name=name)
             for name in unet.attn_processors.keys()}
    unet.set_attn_processor(procs)

    B, H, W = 2, 80, 120
    sample = torch.randn(B, 4, H, W)
    ref_in = torch.randn(B, 4, H, W)
    ctx = torch.randn(B, 77, 1024)
    t = 999.0

    # ── w pass: collect refs exactly like forward_cond ──────────────────
    ref_dict = {}
    with torch.no_grad():
        unet(ref_in, torch.full((B,), t),
             encoder_hidden_states=ctx,
             cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict))
        # ── r pass ────────────────────────────────────────────────────────
        eps = unet(sample, torch.full((B,), t),
                   encoder_hidden_states=ctx,
                   cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict)).sample
    print("r output:", tuple(eps.shape), "abs_mean", eps.abs().mean().item())

    def put(path, arr):
        arr.detach().cpu().float().numpy().tofile(path)

    put(os.path.join(out_dir, "unet_in.bin"), sample)
    put(os.path.join(out_dir, "unet_ref_in.bin"), ref_in)
    put(os.path.join(out_dir, "unet_ctx.bin"), ctx)
    np.array([t], dtype=np.float32).tofile(os.path.join(out_dir, "unet_t.bin"))
    put(os.path.join(out_dir, "unet_eps_out.bin"), eps)

    # staged refs under /tmp (names aligned with the C++ IM_VAE_DUMP points)
    def dump_stage(name, arr):
        arr.detach().cpu().float().numpy().tofile(f"/tmp/ref_unet_{name}.bin")

    def hook_of(name):
        def f(m, i, o):
            dump_stage(name, o if not isinstance(o, tuple) else o[0])
        return f

    unet.conv_in.register_forward_hook(hook_of("w.conv_in"))
    for lvl in range(4):
        for j in range(2):
            unet.down_blocks[lvl].resnets[j].register_forward_hook(
                hook_of(f"w.d{lvl}.r{j}"))
        if lvl < 3:
            unet.down_blocks[lvl].attentions[0].register_forward_hook(
                hook_of(f"w.d{lvl}.attn"))
    unet.mid_block.resnets[0].register_forward_hook(hook_of("w.mid"))
    # resnet internals for the first down resnet + time embedding
    unet.time_embedding.linear_2.register_forward_hook(hook_of("temb"))
    t0 = unet.down_blocks[0].attentions[0]
    t0.norm.register_forward_hook(hook_of("down_blocks.0.attentions.0.tgn"))
    t0.proj_in.register_forward_hook(hook_of("down_blocks.0.attentions.0.tproj"))
    t0.proj_in.register_forward_pre_hook(
        lambda m, a: dump_stage("down_blocks.0.attentions.0.tgather", a[0]))
    r0 = unet.down_blocks[0].resnets[0]
    r0.norm1.register_forward_hook(hook_of("down_blocks.0.resnets.0.norm1"))
    r0.conv1.register_forward_hook(hook_of("down_blocks.0.resnets.0.conv1"))
    r0.time_emb_proj.register_forward_hook(hook_of("down_blocks.0.resnets.0.te"))
    r0.norm2.register_forward_hook(hook_of("down_blocks.0.resnets.0.norm2"))
    r0.conv2.register_forward_hook(hook_of("down_blocks.0.resnets.0.conv2"))
    with torch.no_grad():
        unet(ref_in, torch.full((B,), t),
             encoder_hidden_states=ctx,
             cross_attention_kwargs=dict(mode="w", ref_dict={}))

    # the official ref_dict values themselves (attn1 inputs, torch order)
    # re-collect in a clean pass (pop-consumed above)
    ref_dict2 = {}
    with torch.no_grad():
        unet(ref_in, torch.full((B,), t),
             encoder_hidden_states=ctx,
             cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict2))
    for i, (name, v) in enumerate(sorted(ref_dict2.items())):
        print(f"ref[{i}] {name} {tuple(v.shape)}")
        v.float().cpu().numpy().tofile(f"/tmp/ref_unet_refstore_{i}.bin")

    meta = {"t": t, "B": B, "H": H, "W": W, "L": 77,
            "n_ref": len(ref_dict2)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
