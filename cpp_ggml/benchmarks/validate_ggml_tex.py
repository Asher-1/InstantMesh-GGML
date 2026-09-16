#!/usr/bin/env python3
"""Direct color-source consistency check: bake a texture with the PyTorch
model's net_rgb using the GGML UV layout, and compare pixel-for-pixel against
the actual ggml texture map (same UV -> same pixels -> pure color-source test).

If PSNR is high, the ggml texture colors equal the official net_rgb exactly.
"""
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "benchmarks", "results")
from pytorch_reference import load_model  # noqa: E402
from official_texmap import rasterize_worldpos  # noqa: E402

IMG = "cute_horse"
RESOLUTION = 512


def main():
    model = load_model()
    from pytorch_reference import MV
    images = np.fromfile(os.path.join(MV, IMG, "image.bin"), dtype=np.float32)[:6*3*224*224].reshape(1, 6, 3, 224, 224)
    cameras = np.fromfile(os.path.join(MV, IMG, "camera.bin"), dtype=np.float32)[:6*16].reshape(1, 6, 16)
    with torch.no_grad():
        planes = model.forward_planes(torch.from_numpy(images).cuda(),
                                      torch.from_numpy(cameras).cuda())

    m = trimesh.load(os.path.join(RES, f"{IMG}_tex.obj"), process=False)
    V = np.asarray(m.vertices, np.float32)
    F = np.asarray(m.faces, np.int64)
    uv = np.asarray(m.visual.uv, np.float32)

    # bake with PyTorch net_rgb using ggml UV. ggml C++ writes the PNG with
    # v=0 at image row 0 (no flip), so match that convention (flip_v=False).
    gb, mask = rasterize_worldpos(V, F, uv, RESOLUTION, flip_v=False)
    gb_t = torch.from_numpy(gb).unsqueeze(0).cuda()
    hm = torch.from_numpy(mask[..., None].astype(np.float32)).unsqueeze(0).cuda()
    with torch.no_grad():
        tex_pt = model.get_texture_prediction(planes, [gb_t], hm).clamp(0, 1).squeeze(0).cpu().numpy()
    tex_pt = np.where(mask[..., None], tex_pt, 0.0)

    # actual ggml texture downscaled to same resolution
    tex_gg = np.asarray(Image.open(os.path.join(RES, f"{IMG}_tex.png")).convert("RGB").resize((RESOLUTION, RESOLUTION)), np.float32) / 255.0

    m = mask
    diff = (tex_pt - tex_gg) ** 2
    mse = diff[m].mean()
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    mae = np.abs(tex_pt - tex_gg)[m].mean()
    per = np.abs(tex_pt - tex_gg)[m].mean(axis=0)
    print(f"bake-with-PyTorch-net_rgb (ggml UV) vs ggml texture, over {m.sum()} covered pixels:")
    print(f"  PSNR = {psnr:.2f} dB   MAE = {mae:.4f}   per-channel MAE = {per.round(4)}")
    # correlation of covered colors
    a = tex_pt[m].reshape(-1, 3); b = tex_gg[m].reshape(-1, 3)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    print(f"  pearson r = {corr:.4f}")
    print(f"  ggml covered mean={tex_gg[m].mean(axis=0).round(2)}  py mean={tex_pt[m].mean(axis=0).round(2)}")

    # ---- Decisive: render both textures on the SAME geometry/UV and compare
    # ggml texture vs a PyTorch-net_rgb-baked reference on the ggml UV. Same
    # world positions -> pure texture-content test.
    import tempfile
    ref_dir = tempfile.mkdtemp(prefix="texref_")
    ref_obj = os.path.join(ref_dir, "ref.obj")
    ref_png = os.path.join(ref_dir, "ref.png")
    Image.fromarray((tex_pt * 255).astype(np.uint8)).save(ref_png)
    with open(ref_obj, "w") as f:
        f.write("mtllib ref.mtl\n")
        for v in V: f.write("v %g %g %g\n" % (v[0], v[1], v[2]))
        for u in uv: f.write("vt %g %g\n" % (u[0], u[1]))
        for fc in F: f.write("f %d/%d %d/%d %d/%d\n" % (fc[0]+1, fc[0]+1, fc[1]+1, fc[1]+1, fc[2]+1, fc[2]+1))
    with open(os.path.join(ref_dir, "ref.mtl"), "w") as f:
        f.write("newmtl mat\nKa 1 1 1\nKd 1 1 1\nmap_Kd ref.png\n")
    from render_textured import render_textured
    r_ggml = render_textured(os.path.join(RES, f"{IMG}_tex.obj"))
    r_ref = render_textured(ref_obj)
    rmask = r_ggml.sum(axis=2) > 0.01
    rmse = np.sqrt(((r_ggml - r_ref) ** 2)[rmask].mean())
    rpsnr = 20 * np.log10(1.0 / rmse)
    rmae = np.abs(r_ggml - r_ref)[rmask].mean()
    print(f"RENDER (same UV): ggml vs PyTorch-baked reference, over {rmask.sum()} px:")
    print(f"  PSNR = {rpsnr:.2f} dB   MAE = {rmae:.4f}")


if __name__ == "__main__":
    main()
