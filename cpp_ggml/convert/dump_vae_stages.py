#!/usr/bin/env python3
"""Dump staged encoder activations (torch references) for VAE bring-up.

Mirrors the C++ dump points in src/models/vae.cpp (IM_VAE_DUMP) one-to-one:
same stage names, same torch [C,H,W] memory order, same input fixture
(benchmarks/fixtures/vae/enc_in.bin) so the two sides are directly comparable
via convert/compare_vae_stages.py.

Run (reference venv, from the repo root):
  /tmp/vref/bin/python -m cpp_ggml.convert.dump_vae_stages

Writes /tmp/ref_<stage>.bin.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from .convert_zero123pp import load_safetensors, locate_snapshot

OUT = "/tmp"
DUMP = True  # flip False to just print stage stats


def put(name: str, t: torch.Tensor) -> None:
    t = t.detach().to("cpu").float()
    if DUMP:
        t.numpy().tofile(os.path.join(OUT, f"ref_{name}.bin"))
    print(f"{name:52s} {tuple(t.shape)}  abs_mean={t.abs().mean():.4f}")


def resnet(sd, x, i, j):
    p = f"encoder.down_blocks.{i}.resnets.{j}."
    # C++ dumps the bare group_norm output (affine is a separate op there)
    h = F.group_norm(x, 32, None, None, 1e-6)
    put(p + "rs_norm1", h)
    h = F.silu(h * sd[p + "norm1.weight"].reshape(1, -1, 1, 1) +
               sd[p + "norm1.bias"].reshape(1, -1, 1, 1))
    put(p + "rs_silu1", h)
    h = F.conv2d(h, sd[p + "conv1.weight"], sd[p + "conv1.bias"], padding=1)
    put(p + "rs_conv1", h)
    h = F.group_norm(h, 32, None, None, 1e-6)
    h = F.silu(h * sd[p + "norm2.weight"].reshape(1, -1, 1, 1) +
               sd[p + "norm2.bias"].reshape(1, -1, 1, 1))
    put(p + "rs_silu2", h)
    h = F.conv2d(h, sd[p + "conv2.weight"], sd[p + "conv2.bias"], padding=1)
    put(p + "rs_conv2", h)
    if p + "conv_shortcut.weight" in sd:
        x = F.conv2d(x, sd[p + "conv_shortcut.weight"], sd[p + "conv_shortcut.bias"])
    out = h + x
    put(p + "rs_out", out)
    return out


def main():
    snap = locate_snapshot("")
    sd = load_safetensors(os.path.join(snap, "vae", "diffusion_pytorch_model.safetensors"))
    sd = {k: v.float() for k, v in sd.items()}  # snapshot stores fp16
    root = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "fixtures", "vae")
    enc_in = np.fromfile(os.path.join(root, "enc_in.bin"), dtype=np.float32)
    x = torch.from_numpy(enc_in.copy()).reshape(1, 3, 512, 512)

    with torch.no_grad():
        h = F.conv2d(x, sd["encoder.conv_in.weight"], None, padding=1)
        put("encoder.conv_in_raw", h)
        h = h + sd["encoder.conv_in.bias"].reshape(1, -1, 1, 1)
        put("encoder.conv_in", h)
        for i in range(4):
            h = resnet(sd, h, i, 0)
            h = resnet(sd, h, i, 1)
            if i < 3:
                if i == 0:
                    put("encoder.down0_pre_ds", h)
                # diffusers Downsample2D: asymmetric (right, bottom) zero-pad
                # then stride-2 padding-0 conv (SD VAE behavior)
                h = F.pad(h, (0, 1, 0, 1))
                h = F.conv2d(h, sd[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"],
                             sd[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"],
                             stride=2, padding=0)
            put(f"encoder.down{i}", h)
        # mid block + head (no C++ dump points yet; for later staging)
        p = "encoder.mid_block.resnets.0."
        m0 = F.silu(F.group_norm(h, 32, sd[p + "norm1.weight"], sd[p + "norm1.bias"], 1e-6))
        m0 = F.conv2d(m0, sd[p + "conv1.weight"], sd[p + "conv1.bias"], padding=1)
        m0 = F.silu(F.group_norm(m0, 32, sd[p + "norm2.weight"], sd[p + "norm2.bias"], 1e-6))
        m0 = F.conv2d(m0, sd[p + "conv2.weight"], sd[p + "conv2.bias"], padding=1) + h
        put("encoder.mid0", m0)
        pa = "encoder.mid_block.attentions.0."
        a = F.group_norm(m0, 32, sd[pa + "group_norm.weight"], sd[pa + "group_norm.bias"], 1e-6)
        B, C, H, W = a.shape
        a = a.reshape(B, C, H * W).transpose(1, 2)
        q = F.linear(a, sd[pa + "to_q.weight"], sd[pa + "to_q.bias"])
        k = F.linear(a, sd[pa + "to_k.weight"], sd[pa + "to_k.bias"])
        v = F.linear(a, sd[pa + "to_v.weight"], sd[pa + "to_v.bias"])
        att = torch.softmax(q @ (k.transpose(1, 2) * C ** -0.5), dim=-1)
        a = F.linear(att @ v, sd[pa + "to_out.0.weight"], sd[pa + "to_out.0.bias"])
        attn_out = m0 + a.transpose(1, 2).reshape(B, C, H, W)
        put("encoder.attn", attn_out)
        p = "encoder.mid_block.resnets.1."
        m1 = F.silu(F.group_norm(attn_out, 32, sd[p + "norm1.weight"], sd[p + "norm1.bias"], 1e-6))
        m1 = F.conv2d(m1, sd[p + "conv1.weight"], sd[p + "conv1.bias"], padding=1)
        m1 = F.silu(F.group_norm(m1, 32, sd[p + "norm2.weight"], sd[p + "norm2.bias"], 1e-6))
        m1 = F.conv2d(m1, sd[p + "conv2.weight"], sd[p + "conv2.bias"], padding=1) + attn_out
        put("encoder.mid1", m1)
        h = F.silu(F.group_norm(m1, 32, sd["encoder.conv_norm_out.weight"],
                                sd["encoder.conv_norm_out.bias"], 1e-6))
        h = F.conv2d(h, sd["encoder.conv_out.weight"], sd["encoder.conv_out.bias"], padding=1)
        put("encoder.convout", h)
        h = F.conv2d(h, sd["quant_conv.weight"], sd["quant_conv.bias"])
        put("encoder.quant", h)
        put("encoder.mode", h[:, :4])


if __name__ == "__main__":
    main()
