#!/usr/bin/env python3
"""Quantitative texture-consistency check: ggml texture map vs a faithful
PyTorch-official texture map, on the SAME mesh geometry and SAME input.

Official path (src/utils/mesh_util.xatlas_uvmap + lrm_mesh.get_texture_prediction):
  xatlas.parametrize -> UV atlas -> rasterize world positions -> net_rgb.
nvdiffrast GPU rasterization is replaced by an equivalent CPU barycentric
rasterizer (same interpolation semantics); the COLOR source is the official
PyTorch model's net_rgb. The ggml texture is baked by the ggml port's
synthesizer_texture_forward, which was validated to match PyTorch net_rgb.

Both textured meshes are then rendered from an identical camera and compared
with PSNR over the object mask -> a direct measure of "texture effect parity".
"""
import os
import sys

import numpy as np
import torch
import trimesh
import xatlas
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "benchmarks", "results")

from pytorch_reference import load_model  # noqa: E402  (also sets up shims)
from render_textured import render_textured, rot, VIEW  # noqa: E402

RESOLUTION = 512
IMG = "cute_horse"


def rasterize_worldpos(V, F, uv, res, flip_v=True):
    """CPU orthographic-style rasterization in UV space -> per-pixel world pos.
    flip_v=True  -> image row 0 = v=1 (standard OBJ texture, nvdiffrast-style)
    flip_v=False -> image row 0 = v=0 (matches ggml C++ export convention)."""
    H = W = res
    gb = np.zeros((H, W, 3), np.float32)
    mask = np.zeros((H, W), bool)
    for fi in range(len(F)):
        f = F[fi]
        # pixel coords (v=0 at top like image, uv v=0 bottom -> flip)
        u0, u1, u2 = uv[f[0]], uv[f[1]], uv[f[2]]
        p = np.array([u0[0], u1[0], u2[0]]) * res
        vv = np.array([u0[1], u1[1], u2[1]])
        q = (1.0 - vv if flip_v else vv) * res
        w0, w1, w2 = V[f[0]], V[f[1]], V[f[2]]
        minx = max(0, int(np.floor(p.min()))); maxx = min(res-1, int(np.ceil(p.max())))
        miny = max(0, int(np.floor(q.min()))); maxy = min(res-1, int(np.ceil(q.max())))
        if minx > maxx or miny > maxy: continue
        gx = np.arange(minx, maxx+1, dtype=np.float32) + 0.5
        gy = np.arange(miny, maxy+1, dtype=np.float32) + 0.5
        GX, GY = np.meshgrid(gx, gy)
        den = (p[1]-p[2])*(q[0]-q[2]) + (p[2]-p[0])*(q[1]-q[2])
        if abs(den) < 1e-9: continue
        l0 = ((p[1]-p[2])*(GX-p[2]) + (p[2]-p[0])*(GY-q[2])) / den
        l1 = ((p[2]-p[0])*(GX-p[0]) + (p[0]-p[2])*(GY-q[0])) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any(): continue
        wp = (l0[..., None]*w0 + l1[..., None]*w1 + l2[..., None]*w2)
        reg = gb[miny:maxy+1, minx:maxx+1]
        regm = mask[miny:maxy+1, minx:maxx+1]
        # first writer wins (approx z-buffer free; official uses depth, same)
        sel = inside & ~regm
        reg[sel] = wp[sel]
        regm[sel] = True
    return gb, mask


def official_texture(model, planes, V, F, res=RESOLUTION):
    vmapping, indices, uvs = xatlas.parametrize(V, F)
    # per-original-vertex UV via atlas mapping
    uv_verts = uvs[vmapping]
    gb, mask = rasterize_worldpos(V, F, uv_verts, res)
    gb_t = torch.from_numpy(gb).unsqueeze(0).to("cuda")          # (1,res,res,3)
    hm = torch.from_numpy(mask[..., None].astype(np.float32)).unsqueeze(0).to("cuda")  # (1,res,res,1)
    with torch.no_grad():
        tex = model.get_texture_prediction(planes, [gb_t], hm)   # (1,res,res,3)
        tex = tex.clamp(0, 1).squeeze(0).cpu().numpy()
    tex = np.where(mask[..., None], tex, 0.0)
    return tex, uv_verts, indices


def psnr(a, b, mask):
    a = a.astype(np.float64); b = b.astype(np.float64)
    m = mask[..., None]
    diff = ((a - b) ** 2) * m
    n = m.sum()
    mse = diff.sum() / n if n > 0 else float("inf")
    if mse == 0: return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


def main():
    model = load_model()
    # forward planes for the chosen image (reuse pytorch_reference inputs)
    from pytorch_reference import MV
    images = np.fromfile(os.path.join(MV, IMG, "image.bin"), dtype=np.float32)[:6*3*224*224].reshape(1, 6, 3, 224, 224)
    cameras = np.fromfile(os.path.join(MV, IMG, "camera.bin"), dtype=np.float32)[:6*16].reshape(1, 6, 16)
    with torch.no_grad():
        planes = model.forward_planes(torch.from_numpy(images).cuda(),
                                      torch.from_numpy(cameras).cuda())

    # same mesh geometry the ggml texture was baked on
    m = trimesh.load(os.path.join(RES, f"{IMG}_tex.obj"), process=False)
    V = np.asarray(m.vertices, np.float32)
    F = np.asarray(m.faces, np.int64)

    # official texture (xatlas UV + official net_rgb)
    tex_off, uv_off, idx_off = official_texture(model, planes, V, F)
    off_path = os.path.join(RES, f"{IMG}_official_tex.png")
    Image.fromarray((tex_off * 255).astype(np.uint8)).save(off_path)
    # write official textured obj
    obj_off = os.path.join(RES, f"{IMG}_official_tex.obj")
    with open(obj_off, "w") as f:
        f.write("mtllib %s_off.mtl\n" % IMG)
        for v in V: f.write("v %g %g %g\n" % (v[0], v[1], v[2]))
        for uv in uv_off: f.write("vt %g %g\n" % (uv[0], uv[1]))
        for fc in F: f.write("f %d/%d %d/%d %d/%d\n" % (fc[0]+1, fc[0]+1, fc[1]+1, fc[1]+1, fc[2]+1, fc[2]+1))
    with open(os.path.join(RES, f"{IMG}_off.mtl"), "w") as f:
        f.write("newmtl mat\nKa 1 1 1\nKd 1 1 1\nmap_Kd %s_official_tex.png\n" % IMG)

    # render both textured meshes from the same camera
    r_ggml = render_textured(os.path.join(RES, f"{IMG}_tex.obj"))
    r_off = render_textured(obj_off)
    mask = (r_ggml.sum(axis=2) > 0.01)  # object silhouette (same geometry)
    psnr_val = psnr(r_ggml, r_off, mask)
    mae = np.abs(r_ggml - r_off)[mask].mean()
    print(f"render comparison over {mask.sum()} object pixels:")
    print(f"  PSNR = {psnr_val:.2f} dB   MAE = {mae:.4f}")

    # save side-by-side
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(r_ggml); ax[0].set_title("ggml texture"); ax[0].axis("off")
    ax[1].imshow(r_off); ax[1].set_title("PyTorch-official texture (xatlas+net_rgb)"); ax[1].axis("off")
    ax[2].imshow(np.abs(r_ggml - r_off)); ax[2].set_title(f"|diff|  PSNR={psnr_val:.1f}dB"); ax[2].axis("off")
    fig.tight_layout()
    out = os.path.join(RES, f"texmap_ggml_vs_official_{IMG}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "| official tex:", off_path)


if __name__ == "__main__":
    main()
