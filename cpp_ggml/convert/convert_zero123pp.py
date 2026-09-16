#!/usr/bin/env python3
"""Convert Zero123++ (sudo-ai/zero123plus-v1.2 + TencentARC white-bg UNet) to GGUF.

Three outputs per precision (see docs/ALIGNMENT.md):

  zero123pp_unet_<p>.gguf   SD2.1 UNet2DConditionModel weights (white-bg finetune
                            from TencentARC/InstantMesh overrides every key) +
                            UNet hparams KV.
  zero123pp_vae_<p>.gguf    AutoencoderKL encoder+decoder weights + hparams KV.
  zero123pp_cond_<p>.gguf   CLIPVisionModelWithProjection weights + pipeline
                            constants (always F32):
                              cond.text_emb      [77, 1024]  empty-prompt CLIP text
                                                             embedding (constant)
                              cond.negative_lat  [4, 64, 64] VAE mode-latent of a
                                                             zero image (cfg negative)
                              cond.ramp          [77]        ramping_coefficients
                              scheduler.alphas_cumprod [1000] (torch float32 curve)
                            + preprocessing KV (feature_extractor settings).

Usage:
  python3 -m cpp_ggml.convert.convert_zero123pp \
      --snapshot <hf snapshot dir> --unet-override <diffusion_pytorch_model.bin> \
      --out-dir cpp_ggml/models/gguf -p f16
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

from gguf import GGMLQuantizationType as QType
from gguf import GGUFWriter

from .convert_common import (
    add_kv, begin_gguf_writer, finalize_gguf, ggml_type_for, write_tensor,
)

logger = logging.getLogger("convert-zero123pp")

ARCH_UNET = "zero123pp-unet"
ARCH_VAE = "zero123pp-vae"
ARCH_COND = "zero123pp-cond"


def load_safetensors(path: str) -> dict:
    from safetensors.torch import load_file
    return load_file(path)


# --- torch-op reimplementations (import-diffusers-free) ---------------------
# The installed diffusers 0.39 requires a newer transformers than the official
# run.py pins, so we avoid importing it: the scheduler init math and the VAE
# encoder forward below use the exact same torch ops / dtypes / order as
# diffusers internals, giving identical constants.

def scheduler_alphas_cumprod(sched_cfg: dict) -> torch.Tensor:
    """Verbatim EulerAncestralDiscreteScheduler.__init__ math (linear betas)."""
    T = sched_cfg["num_train_timesteps"]
    if sched_cfg["beta_schedule"] == "linear":
        betas = torch.linspace(sched_cfg["beta_start"], sched_cfg["beta_end"], T,
                               dtype=torch.float32)
    else:  # scaled_linear
        betas = torch.linspace(sched_cfg["beta_start"] ** 0.5, sched_cfg["beta_end"] ** 0.5,
                               T, dtype=torch.float32) ** 2
    return torch.cumprod(1.0 - betas, dim=0)


def _vae_resnet(sd, x, i, j, ch_in, ch_out):
    import torch.nn.functional as F
    p = f"encoder.down_blocks.{i}.resnets.{j}."
    h = F.silu(F.group_norm(x, 32, sd[p+"norm1.weight"], sd[p+"norm1.bias"], 1e-6))
    h = F.conv2d(h, sd[p+"conv1.weight"], sd[p+"conv1.bias"], padding=1)
    h = F.silu(F.group_norm(h, 32, sd[p+"norm2.weight"], sd[p+"norm2.bias"], 1e-6))
    h = F.conv2d(h, sd[p+"conv2.weight"], sd[p+"conv2.bias"], padding=1)
    if p + "conv_shortcut.weight" in sd:
        x = F.conv2d(x, sd[p+"conv_shortcut.weight"], sd[p+"conv_shortcut.bias"])
    return h + x


def _vae_mid_resnet(sd, x, j):
    import torch.nn.functional as F
    p = f"encoder.mid_block.resnets.{j}."
    h = F.silu(F.group_norm(x, 32, sd[p+"norm1.weight"], sd[p+"norm1.bias"], 1e-6))
    h = F.conv2d(h, sd[p+"conv1.weight"], sd[p+"conv1.bias"], padding=1)
    h = F.silu(F.group_norm(h, 32, sd[p+"norm2.weight"], sd[p+"norm2.bias"], 1e-6))
    h = F.conv2d(h, sd[p+"conv2.weight"], sd[p+"conv2.bias"], padding=1)
    return h + x  # mid resnets keep channels (no shortcut conv)


def _vae_attn(sd, x):
    """SD-VAE vanilla spatial attention (heads=1, to_q/k/v Linears)."""
    import torch.nn.functional as F
    p = "encoder.mid_block.attentions.0."
    residual = x
    h = F.group_norm(x, 32, sd[p+"group_norm.weight"], sd[p+"group_norm.bias"], 1e-6)
    B, C, H, W = h.shape
    h = h.reshape(B, C, H * W).transpose(1, 2)                        # [B, HW, C]
    q = F.linear(h, sd[p+"to_q.weight"], sd[p+"to_q.bias"])
    k = F.linear(h, sd[p+"to_k.weight"], sd[p+"to_k.bias"])
    v = F.linear(h, sd[p+"to_v.weight"], sd[p+"to_v.bias"])
    scale = C ** -0.5
    att = torch.softmax(q @ (k.transpose(1, 2) * scale), dim=-1)
    out = F.linear(att @ v, sd[p+"to_out.0.weight"], sd[p+"to_out.0.bias"])
    return residual + out.transpose(1, 2).reshape(B, C, H, W)


def vae_encode_mode(sd, x):
    """AutoencoderKL.encode -> latent_dist.mode() (deterministic).

    Flow: conv_in -> 4 down levels (2 resnets each, stride-2 downsample on
    levels 0-2) -> mid (resnet, attn, resnet) -> norm/silu/conv_out ->
    quant_conv; mode = first latent_channels of the 2*latent_channels head.
    """
    import torch.nn.functional as F
    h = F.conv2d(x, sd["encoder.conv_in.weight"], sd["encoder.conv_in.bias"], padding=1)
    for i in range(4):
        h = _vae_resnet(sd, h, i, 0, 0, 0)   # (ch args unused; shortcut via keys)
        h = _vae_resnet(sd, h, i, 1, 0, 0)
        if i < 3:
            # diffusers Downsample2D(padding=0): right/bottom zero-pad first
            h = F.pad(h, (0, 1, 0, 1))
            h = F.conv2d(h, sd[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"],
                         sd[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"], stride=2, padding=0)
    h = _vae_mid_resnet(sd, h, 0)
    h = _vae_attn(sd, h)
    h = _vae_mid_resnet(sd, h, 1)
    h = F.silu(F.group_norm(h, 32, sd["encoder.conv_norm_out.weight"],
                            sd["encoder.conv_norm_out.bias"], 1e-6))
    h = F.conv2d(h, sd["encoder.conv_out.weight"], sd["encoder.conv_out.bias"], padding=1)
    h = F.conv2d(h, sd["quant_conv.weight"], sd["quant_conv.bias"])
    return h[:, :4]


def locate_snapshot(arg: str) -> str:
    if arg and os.path.isdir(arg):
        return arg
    import glob
    for pat in ("~/.cache/huggingface/hub/models--sudo-ai--zero123plus-v1.2/snapshots/*",
                os.environ.get("HF_HOME", "") + "/hub/models--sudo-ai--zero123plus-v1.2/snapshots/*"):
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            if os.path.isfile(os.path.join(d, "model_index.json")):
                return d
    raise FileNotFoundError("zero123plus-v1.2 snapshot not found; pass --snapshot")


def write_gguf(arch: str, fname: str, hparams: dict, tensors: dict, precision: str,
               f32_tensors: set) -> None:
    writer = begin_gguf_writer(arch, fname)
    for k, v in hparams.items():
        add_kv(writer, k, v)
    total = 0
    for name, t in tensors.items():
        qtype = QType.F32 if name in f32_tensors else ggml_type_for(name, precision)
        total += write_tensor(writer, name, t.detach().to("cpu").float().contiguous(), qtype)
    finalize_gguf(writer, fname)
    logger.info("  %s: %d tensors, %.1f MB", fname, len(tensors), total / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=str, default="")
    ap.add_argument("--unet-override", type=str, default="",
                    help="TencentARC/InstantMesh diffusion_pytorch_model.bin (white-bg UNet)")
    ap.add_argument("--out-dir", type=str, default="models/gguf")
    ap.add_argument("-p", "--precision", type=str, default="all",
                    choices=["f32", "f16", "q8", "all"])
    ap.add_argument("--no-consts", action="store_true",
                    help="skip the torch-side constant precomputation (text_emb/negative_lat)")
    ap.add_argument("--cond-only", action="store_true",
                    help="write only the cond gguf (skip unet/vae rewrite)")
    ap.add_argument("--vae-only", action="store_true",
                    help="write only the vae gguf (skip unet/cond rewrite)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    snap = locate_snapshot(args.snapshot)
    logger.info("snapshot: %s", snap)

    model_index = json.load(open(os.path.join(snap, "model_index.json")))
    unet_cfg = json.load(open(os.path.join(snap, "unet", "config.json")))
    vae_cfg = json.load(open(os.path.join(snap, "vae", "config.json")))
    vis_cfg = json.load(open(os.path.join(snap, "vision_encoder", "config.json")))
    sched_cfg = json.load(open(os.path.join(snap, "scheduler", "scheduler_config.json")))
    fx_vae = json.load(open(os.path.join(snap, "feature_extractor_vae", "preprocessor_config.json")))
    fx_clip = json.load(open(os.path.join(snap, "feature_extractor_clip", "preprocessor_config.json")))
    ramp = model_index["ramping_coefficients"]

    # ---- UNet: official repo weights, overridden by the white-bg finetune ----
    unet_sd = load_safetensors(os.path.join(snap, "unet", "diffusion_pytorch_model.safetensors"))
    if args.unet_override:
        ov = torch.load(args.unet_override, map_location="cpu", weights_only=False)
        if isinstance(ov, dict) and "state_dict" in ov:
            ov = ov["state_dict"]
        missing = [k for k in unet_sd if k not in ov]
        extra = [k for k in ov if k not in unet_sd]
        if missing or extra:
            raise RuntimeError(f"unet override key mismatch: missing={missing[:4]} extra={extra[:4]}")
        unet_sd = {k: ov[k] for k in unet_sd}
        logger.info("unet: applied white-bg override (%d tensors)", len(unet_sd))

    vae_sd = load_safetensors(os.path.join(snap, "vae", "diffusion_pytorch_model.safetensors"))
    vis_sd = load_safetensors(os.path.join(snap, "vision_encoder", "model.safetensors"))

    precisions = ["f32", "f16", "q8"] if args.precision == "all" else [args.precision]
    os.makedirs(args.out_dir, exist_ok=True)

    # UNet hparams (diffusers UNet2DConditionModel config -> KV)
    unet_hp = {
        "unet.in_channels": unet_cfg["in_channels"],
        "unet.out_channels": unet_cfg["out_channels"],
        "unet.block_out_channels": ",".join(map(str, unet_cfg["block_out_channels"])),
        "unet.attention_head_dim": ",".join(map(str, unet_cfg["attention_head_dim"])),
        "unet.down_block_types": ",".join(unet_cfg["down_block_types"]),
        "unet.up_block_types": ",".join(unet_cfg["up_block_types"]),
        "unet.layers_per_block": unet_cfg["layers_per_block"],
        "unet.cross_attention_dim": unet_cfg["cross_attention_dim"],
        "unet.norm_num_groups": unet_cfg["norm_num_groups"],
        "unet.norm_eps": unet_cfg["norm_eps"],
        "unet.use_linear_projection": 1 if unet_cfg["use_linear_projection"] else 0,
        "unet.flip_sin_to_cos": 1 if unet_cfg["flip_sin_to_cos"] else 0,
        "unet.freq_shift": unet_cfg["freq_shift"],
    }
    vae_hp = {
        "vae.block_out_channels": ",".join(map(str, vae_cfg["block_out_channels"])),
        "vae.latent_channels": vae_cfg["latent_channels"],
        "vae.scaling_factor": vae_cfg["scaling_factor"],
        "vae.norm_num_groups": vae_cfg.get("norm_num_groups", 32),
        "vae.norm_eps": vae_cfg.get("norm_eps", 1e-6),
        "vae.sample_size": vae_cfg["sample_size"],
    }
    vis_hp = {
        "vis.hidden_size": vis_cfg["hidden_size"],
        "vis.num_hidden_layers": vis_cfg["num_hidden_layers"],
        "vis.num_attention_heads": vis_cfg["num_attention_heads"],
        "vis.intermediate_size": vis_cfg["intermediate_size"],
        "vis.image_size": vis_cfg["image_size"],
        "vis.patch_size": vis_cfg["patch_size"],
        "vis.projection_dim": vis_cfg["projection_dim"],
        "vis.layer_norm_eps": vis_cfg["layer_norm_eps"],
    }

    for p in precisions:
        skip_unet = args.cond_only or args.vae_only
        logger.info("== precision %s ==", p)
        if not skip_unet:
            write_gguf(ARCH_UNET, os.path.join(args.out_dir, f"zero123pp_unet_{p}.gguf"),
                       unet_hp, unet_sd, p, set())
        if not args.cond_only:
            write_gguf(ARCH_VAE, os.path.join(args.out_dir, f"zero123pp_vae_{p}.gguf"),
                       vae_hp, vae_sd, p, set())

    # ---- cond file: vision weights + pipeline constants (always F32) ----
    tensors = {f"vis.{k}": v for k, v in vis_sd.items()}
    f32_names = set(tensors.keys())

    if not args.no_consts:
        logger.info("precomputing pipeline constants (torch, diffusers-free)...")
        from transformers import CLIPTextModel, CLIPTokenizer

        tok = CLIPTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
        txt = CLIPTextModel.from_pretrained(os.path.join(snap, "text_encoder"),
                                            torch_dtype=torch.float32).eval()
        with torch.no_grad():
            ids = tok("", padding="max_length", max_length=txt.config.max_position_embeddings,
                      truncation=True, return_tensors="pt").input_ids
            text_emb = txt(ids).last_hidden_state[0]  # [77, 1024]
        tensors["cond.text_emb"] = text_emb
        f32_names.add("cond.text_emb")

        # negative cfg branch: VAE mode-latent of a zero image (512x512 crop ->
        # 64x64 latent). Deterministic mode instead of the pipeline's .sample().
        vae_sd_f32 = {k: v.float() for k, v in vae_sd.items()}
        with torch.no_grad():
            zeros = torch.zeros(1, 3, fx_vae["crop_size"]["height"],
                                fx_vae["crop_size"]["width"])
            negative_lat = vae_encode_mode(vae_sd_f32, zeros)
        tensors["cond.negative_lat"] = negative_lat[0]
        f32_names.add("cond.negative_lat")

        tensors["scheduler.alphas_cumprod"] = scheduler_alphas_cumprod(sched_cfg)
        f32_names.add("scheduler.alphas_cumprod")

    cond_hp = {
        "cond.ramp": ",".join(f"{r:.9g}" for r in ramp),
        "cond.num_ramp": len(ramp),
        "cond.text_len": 77,
        "sched.num_train_timesteps": sched_cfg["num_train_timesteps"],
        "sched.beta_start": sched_cfg["beta_start"],
        "sched.beta_end": sched_cfg["beta_end"],
        "sched.beta_schedule": sched_cfg["beta_schedule"],
        "sched.prediction_type": sched_cfg["prediction_type"],
        "sched.timestep_spacing": "trailing",
        "sched.steps_offset": sched_cfg["steps_offset"],
        "fx_vae.size": fx_vae["size"]["shortest_edge"],
        "fx_vae.crop": fx_vae["crop_size"]["height"],
        "fx_vae.mean": fx_vae["image_mean"] if isinstance(fx_vae["image_mean"], float) else -1.0,
        "fx_vae.std": fx_vae["image_std"] if isinstance(fx_vae["image_std"], float) else -1.0,
        "fx_clip.size": fx_clip["size"]["shortest_edge"],
        "fx_clip.crop": fx_clip["crop_size"]["height"],
        "fx_clip.mean": ",".join(map(str, fx_clip["image_mean"])),
        "fx_clip.std": ",".join(map(str, fx_clip["image_std"])),
    }
    write_gguf(ARCH_COND, os.path.join(args.out_dir, "zero123pp_cond_f32.gguf"),
               cond_hp, tensors, "f32", f32_names)
    logger.info("done.")


if __name__ == "__main__":
    sys.exit(main())
