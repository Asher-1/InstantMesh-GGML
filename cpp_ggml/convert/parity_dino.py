"""First-round parity check for the C++ DINO ggml port against PyTorch.

Reference forward is exactly InstantMesh's DinoWrapper (src/models/encoder/):
  camera_embedder(camera) -> adaln_input [B, 768]
  patch_embeddings(image)[+cls] + pos_enc -> [B, 1+N, 768]
  12x ViTLayer(adaln) -> last_hidden_state -> final LayerNorm

Usage:
 1. Generate the shared input and the PyTorch reference output:
      python3 -m convert.parity_dino --ckpt ckpts/instant_mesh_large.ckpt \
          --image examples/robot.jpg --camera camera.bin \
          --in /tmp/dino_inputs.bin --ref /tmp/dino_ref.bin
 2. Run the C++ graph on the same input:
      ./build/dino --device cpu models/gguf/dino_f16.gguf \
          --in /tmp/dino_inputs.bin --out /tmp/dino_out.bin
 3. Compare:
      python3 -m convert.parity_dino --compare /tmp/dino_ref.bin /tmp/dino_out.bin
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class RefDino(nn.Module):
    """Self-contained DINO ViT-B/16 + per-layer adaLN, mirroring src dino.py.

    Weights are keyed by the GGUF tensor names (after stripping `encoder.`):
      model.embeddings.*, model.encoder.layer.N.*, model.layernorm.*
    """

    def __init__(self):
        super().__init__()
        H = 768
        self.cls = nn.Parameter(torch.zeros(1, 1, H))
        self.posemb = nn.Parameter(torch.zeros(1, 197, H))
        self.patch_proj = nn.Conv2d(3, H, kernel_size=16, stride=16)
        self.layernorm = nn.LayerNorm(H, eps=1e-12)
        self.layers = nn.ModuleList([RefDinoLayer() for _ in range(12)])

    def load_from(self, ckpt, strip="lrm_generator.encoder."):
        sd = torch.load(ckpt, map_location="cpu")
        ck = sd["state_dict"] if "state_dict" in sd else sd
        remap = {}
        for k, v in ck.items():
            if not k.startswith(strip):
                continue
            name = k[len(strip):]  # e.g. model.encoder.layer.3.attention.output.dense.weight
            if name.startswith("model.embeddings.cls_token"):
                dst = "cls"
            elif name.startswith("model.embeddings.position_embeddings"):
                dst = "posemb"
            elif name.startswith("model.layernorm."):
                dst = "layernorm." + name.split(".")[-1]
            elif ".patch_embeddings.projection." in name:
                dst = "patch_proj." + name.split(".")[-1]
            elif ".adaLN_modulation.1." in name:
                n = int(name.split("layer.")[1].split(".")[0])
                dst = f"layers.{n}.adaLN_modulation." + name.split(".")[-1]
            elif ".attention.attention.query." in name:
                n = int(name.split("layer.")[1].split(".")[0])
                dst = f"layers.{n}.q." + name.split(".")[-1]
            elif ".attention.attention.key." in name:
                n = int(name.split("layer.")[1].split(".")[0])
                dst = f"layers.{n}.k." + name.split(".")[-1]
            elif ".attention.attention.value." in name:
                n = int(name.split("layer.")[1].split(".")[0])
                dst = f"layers.{n}.v." + name.split(".")[-1]
            elif ".attention.output.dense." in name:
                n = int(name.split("layer.")[1].split(".")[0])
                dst = f"layers.{n}.attn_out." + name.split(".")[-1]
            elif ".layernorm_before." in name:
                n = int(name.split("layer.")[1].split(".")[0])
                dst = f"layers.{n}.layernorm_before." + name.split(".")[-1]
            elif ".layernorm_after." in name:
                n = int(name.split("layer.")[1].split(".")[0])
                dst = f"layers.{n}.layernorm_after." + name.split(".")[-1]
            elif ".intermediate.dense." in name:
                n = int(name.split("layer.")[1].split(".")[0])
                dst = f"layers.{n}.intermediate." + name.split(".")[-1]
            elif ".output.dense." in name:
                n = int(name.split("layer.")[1].split(".")[0])
                dst = f"layers.{n}.output." + name.split(".")[-1]
            else:
                continue
            remap[dst] = v
        self.load_state_dict(remap, strict=True)

    def forward(self, image, adaln, collect=False):
        # adaln: [B', 768], image: [B', 3, 224, 224]
        patch = self.patch_proj(image).flatten(2).transpose(1, 2)  # [B,196,768]
        x = torch.cat([self.cls.expand(patch.shape[0], -1, -1), patch], dim=1)
        x = x + self.posemb
        outs = [x] if collect else None
        for layer in self.layers:
            x = layer(x, adaln)
            if collect:
                outs.append(x)
        out = self.layernorm(x)
        return (out, outs) if collect else out


class RefDinoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        H = 768
        self.layernorm_before = nn.LayerNorm(H, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(H, eps=1e-12)
        self.adaLN_modulation = nn.Linear(H, 4 * H)
        self.q = nn.Linear(H, H); self.k = nn.Linear(H, H); self.v = nn.Linear(H, H)
        self.attn_out = nn.Linear(H, H)
        self.intermediate = nn.Linear(H, 4 * H)
        self.output = nn.Linear(4 * H, H)

    def forward(self, x, adaln):
        # adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(H, 4H)) — SiLU first.
        shift_msa, scale_msa, shift_mlp, scale_mlp = torch.chunk(
            self.adaLN_modulation(F.silu(adaln)), 4, dim=1)
        # attention
        h = modulate(self.layernorm_before(x), shift_msa, scale_msa)
        q = self.q(h); k = self.k(h); vv = self.v(h)
        B, N, C = q.shape
        q = q.view(B, N, 12, -1).transpose(1, 2)
        k = k.view(B, N, 12, -1).transpose(1, 2)
        vv = vv.view(B, N, 12, -1).transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) / (C // 12) ** 0.5, dim=-1)
        o = (att @ vv).transpose(1, 2).reshape(B, N, C)
        o = self.attn_out(o)
        x = o + x
        # mlp
        y = modulate(self.layernorm_after(x), shift_mlp, scale_mlp)
        y = self.output(F.gelu(self.intermediate(y)))
        return y + x


def build_reference(ckpt_path, strip="lrm_generator.encoder."):
    model = RefDino()
    model.load_from(ckpt_path, strip)
    model.eval()
    # camera embedder: Linear(16->768) -> SiLU -> Linear(768->768)
    cam1 = nn.Linear(16, 768)
    cam2 = nn.Linear(768, 768)
    sd = torch.load(ckpt_path, map_location="cpu")
    ck = sd["state_dict"] if "state_dict" in sd else sd
    for k, v in ck.items():
        if not k.startswith(strip + "camera_embedder."):
            continue
        short = k[len(strip + "camera_embedder."):]  # "0.weight"/"2.weight"
        if short.startswith("0."):
            cam1.load_state_dict({short[2:]: v}, strict=False)
        elif short.startswith("2."):
            cam2.load_state_dict({short[2:]: v}, strict=False)
    cam1.eval(); cam2.eval()
    return model, cam1, cam2


def load_image(path, size=224):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)  # [3,H,W]
    arr = (arr - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return arr


def generate(args):
    model, cam1, cam2 = build_reference(args.ckpt, args.prefix)
    model.eval()

    image_np = load_image(args.image, size=224)
    image = torch.from_numpy(image_np)[None]  # [1,3,224,224]
    if args.camera:
        camera = np.fromfile(args.camera, dtype=np.float32).reshape(1, 16)
    else:
        camera = np.zeros((1, 16), dtype=np.float32)
        for i in range(16):
            camera[0, i] = (i % 4) * 0.1
    camera_t = torch.from_numpy(camera).float()

    with torch.no_grad():
        adaln = F.silu(cam1(camera_t))
        adaln = cam2(adaln)  # [1,768]
        if os.environ.get("IM_DUMP"):
            np.asarray(adaln[0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_adaln.bin")
            # layer0 internals for isolation
            l0 = model.layers[0]
            x0 = model(image.float(), adaln, collect=False)
            # recompute layer0 pieces on the embedding input
            with torch.no_grad():
                emb = model.patch_proj(image.float()).flatten(2).transpose(1, 2)
                emb = torch.cat([model.cls.expand(emb.shape[0], -1, -1), emb], dim=1) + model.posemb
                mod0 = l0.adaLN_modulation(F.silu(adaln))
                np.asarray(mod0[0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_mod.bin")
                sm, sc, sm2, sc2 = torch.chunk(mod0, 4, dim=1)
                pa = modulate(l0.layernorm_before(emb), sm, sc)
                np.asarray(pa[0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_preattn.bin")
                q = l0.q(pa); k = l0.k(pa); vv = l0.v(pa)
                np.asarray(q[0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_q.bin")
                np.asarray(k[0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_k.bin")
                np.asarray(vv[0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_v.bin")
                B, N, C = q.shape
                q = q.view(B, N, 12, -1).transpose(1, 2)
                k = k.view(B, N, 12, -1).transpose(1, 2)
                vv = vv.view(B, N, 12, -1).transpose(1, 2)
                att = F.softmax((q @ k.transpose(-2, -1)) / (C // 12) ** 0.5, dim=-1)
                np.asarray(att[0].permute(2, 0, 1).contiguous().reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_att.bin")
                o = (att @ vv).transpose(1, 2).reshape(B, N, C)
                np.asarray(o[0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_attn.bin")
            feats, layer_outs = model(image.float(), adaln, collect=True)
            np.asarray(layer_outs[0][0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_emb.bin")
            for i, lo in enumerate(layer_outs[1:]):
                np.asarray(lo[0].reshape(-1).numpy(), dtype=np.float32).tofile(f"/tmp/ref_layer{i}.bin")
            # conv output before cls concat: patch [B,196,768] (flattened from [B,768,14,14])
            with torch.no_grad():
                patch = model.patch_proj(image.float())  # [1,768,14,14]
                patch = patch.flatten(2).transpose(1, 2).contiguous()  # [1,196,768]
            np.asarray(patch[0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_conv.bin")
            # posemb + cls
            np.asarray(model.posemb.detach().reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_posemb.bin")
            np.asarray(model.cls.detach().reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_cls.bin")
            print("dumped ref embedding + 12 layer outputs + conv/posemb/cls")
        else:
            feats = model(image.float(), adaln)  # [1,197,768]

    # write shared input: image [1,3,224,224] f32 then camera [1,16] f32
    blob = np.concatenate([
        image_np.reshape(-1).astype(np.float32),
        camera.reshape(-1).astype(np.float32),
    ])
    blob.tofile(args.in_blob)
    # write reference output [1,197,768] f32
    feats.detach().cpu().numpy().reshape(-1).astype(np.float32).tofile(args.ref_blob)
    print(f"wrote {args.in_blob} ({blob.size} floats) and {args.ref_blob} ({feats.numel()} floats)")
    ref = feats.detach().cpu().numpy().reshape(-1)
    print(f"ref stats: mean={ref.mean():.5f} std={ref.std():.5f} first={ref[0]:.5f}")


def compare(args):
    ref = np.fromfile(args.ref, dtype=np.float32)
    out = np.fromfile(args.out, dtype=np.float32)
    return _report(ref, out, "final")


def compare_layers(args):
    r = 0
    for name in ["conv", "emb"] + [f"layer{i}" for i in range(12)]:
        rf = np.fromfile(f"/tmp/ref_{name}.bin", dtype=np.float32)
        of = np.fromfile(f"/tmp/dino_{name}.bin", dtype=np.float32)
        if rf.size == 0 or of.size == 0:
            print(f"{name}: missing file")
            continue
        r = max(r, _report(rf, of, name))
    return r


def _report(ref, out, tag):
    if ref.size != out.size:
        print(f"[{tag}] size mismatch: ref={ref.size} out={out.size}")
        return 1
    err = np.abs(ref - out)
    print(f"[{tag}] max_abs={err.max():.3e} mean_abs={err.mean():.3e} "
          f"(ref std={ref.std():.4f})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--ckpt", default="ckpts/instant_mesh_large.ckpt")
    g.add_argument("--prefix", default="lrm_generator.encoder.")
    g.add_argument("--image", default="examples/robot.jpg")
    g.add_argument("--camera", default=None)
    g.add_argument("--in", dest="in_blob", default="/tmp/dino_inputs.bin")
    g.add_argument("--ref", dest="ref_blob", default="/tmp/dino_ref.bin")
    g.set_defaults(fn=generate)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("out")
    c.set_defaults(fn=compare)
    cl = sub.add_parser("compare-layers")
    cl.set_defaults(fn=compare_layers)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()