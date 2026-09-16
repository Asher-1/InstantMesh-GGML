"""Parity check for the C++ TriplaneTransformer ggml port against PyTorch.

Reference forward mirrors InstantMesh's TriplaneTransformer
(src/models/decoder/transformer.py):
  x = pos_embed.repeat(N,1,1)                       -> [N, 3072, 1024]
  16x BasicTransformerBlock(x, cond)                -> [N, 3072, 1024]
  norm(x) -> view(N,3,32,32,d) -> einsum('nihwd->indhw')
         -> view(3N,d,32,32) -> deconv(2,2)        -> [3N, 80, 64, 64]
         -> view(3,N,80,64,64) -> einsum('indhw->nidhw') -> [N,3,80,64,64]

Usage:
 1. Generate the shared cond input and the PyTorch reference output:
      python3 -m convert.parity_lrm --ckpt ckpts/instant_mesh_large.ckpt \
          --cond /tmp/cond.bin --ref /tmp/lrm_ref.bin
 2. Run the C++ graph on the same input:
      ./build/lrm_transformer --device cpu models/gguf/lrm_transformer_f32.gguf \
          --cond /tmp/cond.bin --cond-len 197 --out /tmp/lrm_out.bin
 3. Compare:
      python3 -m convert.parity_lrm --compare /tmp/lrm_ref.bin /tmp/lrm_out.bin
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RefTriplaneTransformer(nn.Module):
    """Self-contained TriplaneTransformer, keyed by the GGUF tensor names."""

    def __init__(self, inner_dim=1024, cond_dim=768, low=32, high=64, dim=80,
                 num_layers=16, num_heads=16, eps=1e-6):
        super().__init__()
        self.inner_dim = inner_dim
        self.low = low
        self.high = high
        self.dim = dim
        self.pos_embed = nn.Parameter(torch.randn(1, 3 * low * low, inner_dim) * (1. / inner_dim) ** 0.5)
        self.layers = nn.ModuleList([
            RefBasicTransformerBlock(inner_dim, cond_dim, num_heads, eps)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.deconv = nn.ConvTranspose2d(inner_dim, dim, kernel_size=2, stride=2, padding=0)

    def load_from(self, ckpt, strip="lrm_generator.transformer."):
        sd = torch.load(ckpt, map_location="cpu")
        ck = sd["state_dict"] if "state_dict" in sd else sd
        # GGUF names are the state_dict names with the transformer. prefix removed.
        remap = {}
        for k, v in ck.items():
            if not k.startswith(strip):
                continue
            name = k[len(strip):]
            remap[name] = v
        self.load_state_dict(remap, strict=True)

    def forward(self, cond):
        # cond: [N, L_cond, D_cond]
        N = cond.shape[0]
        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        for layer in self.layers:
            x = layer(x, cond)
        x = self.norm(x)
        x = x.view(N, 3, self.low, self.low, -1)
        x = torch.einsum('nihwd->indhw', x)
        x = x.contiguous().view(3 * N, -1, self.low, self.low)
        x = self.deconv(x)
        x = x.view(3, N, *x.shape[-3:])
        x = torch.einsum('indhw->nidhw', x)
        return x.contiguous()


class RefBasicTransformerBlock(nn.Module):
    def __init__(self, inner_dim, cond_dim, num_heads, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=0., bias=False, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, dropout=0., bias=False,
            batch_first=True)
        self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, 4 * inner_dim),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(4 * inner_dim, inner_dim),
            nn.Dropout(0.),
        )

    def forward(self, x, cond):
        x = x + self.cross_attn(self.norm1(x), cond, cond)[0]
        if getattr(self, "_dump0", False):
            np.asarray(x.reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_x_ca.bin")
        s = self.norm2(x)
        qkv = s @ self.self_attn.in_proj_weight.T  # [N, seq, 3D]
        if getattr(self, "_dump0", False):
            np.asarray(qkv.reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_sa_out.bin")
            # self-attn kq within heads: qh @ kh^T / sqrt(hd), layout [nh,q,kv]
            D = qkv.shape[-1] // 3
            hd = D // self.self_attn.num_heads
            qh = qkv[..., :D].view(-1, self.self_attn.num_heads, hd).transpose(0, 1)
            kh = qkv[..., D:2*D].view(-1, self.self_attn.num_heads, hd).transpose(0, 1)
            kq = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(hd)
            np.asarray(kq[0].reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_sa_kq.bin")
        sa = self.self_attn(s, s, s)[0]
        if getattr(self, "_dump0", False):
            np.asarray(sa.reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_sa_attn.bin")
        x = x + sa
        if getattr(self, "_dump0", False):
            np.asarray(x.reshape(-1).numpy(), dtype=np.float32).tofile("/tmp/ref_l0_x_sa.bin")
        x = x + self.mlp(self.norm3(x))
        return x


def build_reference(ckpt_path, strip="lrm_generator.transformer."):
    model = RefTriplaneTransformer()
    model.load_from(ckpt_path, strip)
    model.eval()
    return model


def generate(args):
    model = build_reference(args.ckpt, args.prefix)
    # condition: DINO output [N, 197, 768]; use N=1, cond-len from args.
    L_cond = args.cond_len
    N = args.batch
    if args.reuse_dino:
        # feed the previously dumped DINO output [1,197,768] as the condition.
        feats = np.fromfile(args.reuse_dino, dtype=np.float32).reshape(1, 197, 768)
        cond_np = feats[:N].copy()
    else:
        rng = np.random.default_rng(0)
        cond_np = rng.standard_normal((N, L_cond, 768)).astype(np.float32)
    cond = torch.from_numpy(cond_np).float()

    with torch.no_grad():
        x = model.pos_embed.repeat(N, 1, 1)
        for li, layer in enumerate(model.layers):
            x = layer(x, cond)
        x_norm = model.norm(x)
        x = x_norm.view(N, 3, model.low, model.low, -1)
        x = torch.einsum('nihwd->indhw', x)
        x = x.contiguous().view(3 * N, -1, model.low, model.low)
        out = model.deconv(x)
        out = out.view(3, N, *out.shape[-3:])
        out = torch.einsum('indhw->nidhw', out).contiguous()

    cond_np.reshape(-1).astype(np.float32).tofile(args.cond_blob)
    out.detach().cpu().numpy().reshape(-1).astype(np.float32).tofile(args.ref_blob)
    print(f"wrote {args.cond_blob} ({cond_np.size} floats) and "
          f"{args.ref_blob} ({out.numel()} floats)")
    ref = out.detach().cpu().numpy().reshape(-1)
    print(f"ref stats: mean={ref.mean():.5f} std={ref.std():.5f} first={ref[0]:.5f}")


def compare(args):
    a = np.fromfile(args.ref, dtype=np.float32)
    b = np.fromfile(args.out, dtype=np.float32)
    if a.size != b.size:
        print(f"size mismatch: ref {a.size} vs out {b.size}")
        return
    d = np.abs(a - b)
    denom = 1e-6 + np.abs(a)
    rel = d / denom
    print(f"[final] max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
          f"max_rel={rel.max():.3e} mean_rel={rel.mean():.3e} "
          f"(ref std={a.std():.4f})")
    # per-plane breakdown (final layout [N,3,80,64,64] -> plane stride = 80*64*64)
    if a.size % (3 * 80 * 64 * 64) == 0:
        a3 = a.reshape(-1, 3, 80, 64, 64)
        b3 = b.reshape(-1, 3, 80, 64, 64)
        for p in range(3):
            da = np.abs(a3[:, p] - b3[:, p])
            print(f"  plane[{p}]: max_abs={da.max():.3e} mean_abs={da.mean():.3e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--prefix", default="lrm_generator.transformer.")
    g.add_argument("--cond", dest="cond_blob", required=True)
    g.add_argument("--ref", dest="ref_blob", required=True)
    g.add_argument("--cond-len", type=int, default=197)
    g.add_argument("--batch", type=int, default=1)
    g.add_argument("--reuse-dino", default=None,
                   help="use the DINO example output as the cond input")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()