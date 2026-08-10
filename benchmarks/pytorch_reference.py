#!/usr/bin/env python3
"""PyTorch-CUDA reference reconstruction, fed the EXACT same multiview inputs
as the ggml benchmark (benchmarks/mv/<img>/image.bin + camera.bin).

Uses the upstream InstantMesh model (src/models/lrm_mesh.LRMGenerator):
  planes = forward_planes(images, cameras)
  verts, faces, colors = extract_mesh(planes, use_texture_map=False)

extract_mesh queries per-vertex RGB via the synthesizer's net_rgb — the same
branch the ggml port now implements — so this is an aligned colored reference.
Only needs CUDA + torch (no nvdiffrast; that's only for the render/texmap path).

Usage: python3 benchmarks/pytorch_reference.py [image ...]
Writes benchmarks/pytorch_ref/<img>__pytorch.obj and a times CSV.
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- Self-contained dependency shims ------------------------------
# The upstream run.py imports pytorch_lightning (only for seed_everything)
# and src.models imports nvdiffrast (only for the render/texmap path). We only
# need vertex-color extraction, so create minimal in-memory shims rather than
# requiring those heavy GPU packages. Written to a temp dir and prepended to
# sys.path so `import` succeeds; any actual nvdiffrast call raises.
import tempfile

_SHIM = tempfile.mkdtemp(prefix="instmesh_shims_")
sys.path.insert(0, _SHIM)

import os as _os
_os.makedirs(_os.path.join(_SHIM, "pytorch_lightning"), exist_ok=True)
with open(_os.path.join(_SHIM, "pytorch_lightning", "__init__.py"), "w") as _f:
    _f.write("import random\nimport numpy as np\nimport torch\n"
             "def seed_everything(seed):\n"
             "    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed);"
             " torch.cuda.manual_seed_all(seed); return seed\n")

_os.makedirs(_os.path.join(_SHIM, "nvdiffrast"), exist_ok=True)
with open(_os.path.join(_SHIM, "nvdiffrast", "__init__.py"), "w") as _f:
    _f.write("from . import torch\n")
with open(_os.path.join(_SHIM, "nvdiffrast", "torch.py"), "w") as _f:
    _f.write("""
class _Unavailable:
    def __init__(self, name): self._name = name
    def __call__(self, *a, **k):
        raise NotImplementedError(f"nvdiffrast.torch.{self._name} is a stub "
                                  "(render/texmap path unavailable)")
    def __getattr__(self, name): return _Unavailable(f"{self._name}.{name}")
class RasterizeCudaContext(_Unavailable):
    def __init__(self, *a, **k): super().__init__("RasterizeCudaContext")
rasterize = _Unavailable("rasterize"); interpolate = _Unavailable("interpolate")
texture = _Unavailable("texture"); antialias = _Unavailable("antialias")
mipmap = _Unavailable("mipmap")
""")

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything  # shim above

# Newer `transformers` removed prune_heads helpers; dino.py imports them at
# module load but only ever calls them from ViT.prune_heads (never in forward).
# Provide compatible no-op shims so the upstream encoder imports cleanly.
import transformers.pytorch_utils as _tpu
if not hasattr(_tpu, "find_pruneable_heads_and_indices"):
    _tpu.find_pruneable_heads_and_indices = lambda heads, n_heads, head_size, already_pruned_heads: (heads, set())
    _tpu.prune_linear_layer = lambda layer, index, dim=0: layer

from src.utils.train_util import instantiate_from_config

# The custom DINO ViT calls self.get_head_mask(), removed from newer
# `transformers`. head_mask is always None during inference -> return as-is.
from src.models.encoder.dino import ViTModel
if not hasattr(ViTModel, "get_head_mask"):
    ViTModel.get_head_mask = lambda self, head_mask, num_hidden_layers, *a, **k: head_mask

MV = os.path.join(ROOT, "benchmarks", "mv")
OUT = os.path.join(ROOT, "benchmarks", "pytorch_ref")
CONFIG = os.path.join(ROOT, "configs", "instant-mesh-large.yaml")


def load_model():
    seed_everything(42)
    config = OmegaConf.load(CONFIG)
    model = instantiate_from_config(config.model_config)
    ckpt = torch.load(config.infer_config.model_path, map_location="cpu")["state_dict"]
    state_dict = {k[14:]: v for k, v in ckpt.items() if k.startswith("lrm_generator.")}
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda")
    # Full-res flexicubes grid (grid_res=128) needs ~15GB VRAM; this GPU has
    # ~11.7GB. Lower the geometry grid so the same large-model weights fit.
    # Color/appearance comparison stays valid (net_rgb is grid-independent);
    # only the extracted mesh is slightly coarser than the ggml 128-grid.
    model.grid_res = 88
    model.init_flexicubes_geometry("cuda", fovy=30.0)
    model = model.eval()

    # --- CRITICAL input normalization fix ----------------------------------
    # benchmarks/mv/<img>/image.bin is already ImageNet-normalized (the exact
    # tensor the ggml pipeline feeds into its DINO port). The upstream
    # DinoWrapper applies ViTImageProcessor normalisation on top of a *raw*
    # [0,1] image; feeding our pre-normalized tensor through it would
    # double-normalize and corrupt the reference (this was the source of the
    # large ggml-vs-pytorch difference). Bypass the processor so the reference
    # sees the same normalized input ggml does.
    from einops import rearrange as _r  # noqa: E402

    def _no_norm_forward(image, camera):
        if image.ndim == 5:
            image = _r(image, "b n c h w -> (b n) c h w")
        cam_emb = model.encoder.camera_embedder(camera)
        cam_emb = _r(cam_emb, "b n d -> (b n) d")
        out = model.encoder.model(image, adaln_input=cam_emb,
                                  interpolate_pos_encoding=True)
        return out.last_hidden_state

    model.encoder.forward = _no_norm_forward
    return model


def reference(model, img):
    images = np.fromfile(os.path.join(MV, img, "image.bin"), dtype=np.float32)
    cameras = np.fromfile(os.path.join(MV, img, "camera.bin"), dtype=np.float32)
    V = 6
    images = images[: V * 3 * 224 * 224].reshape(1, V, 3, 224, 224)
    cameras = cameras[: V * 16].reshape(1, V, 16)
    images = torch.from_numpy(images).to("cuda")
    cameras = torch.from_numpy(cameras).to("cuda")

    t0 = time.time()
    with torch.no_grad():
        planes = model.forward_planes(images, cameras)
        vertices, faces, colors = model.extract_mesh(planes, use_texture_map=False)
    dt = time.time() - t0

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{img}__pytorch.obj")
    with open(path, "w") as f:
        for v, c in zip(vertices, colors):
            # colors come from extract_mesh as uint8 [0,255]; write as [0,1]
            # floats to match the ggml OBJ format (v x y z r g b)
            f.write("v %g %g %g %g %g %g\n" % (v[0], v[1], v[2],
                                               c[0] / 255.0, c[1] / 255.0, c[2] / 255.0))
        for fc in faces:
            f.write("f %d %d %d\n" % (fc[0] + 1, fc[1] + 1, fc[2] + 1))
    with open(os.path.join(OUT, "times.csv"), "a", newline="") as f:
        csv.writer(f).writerow([img, "pytorch", "cuda", f"{dt:.3f}", 0])
    print(f"{img}: pytorch mesh {len(vertices)} verts in {dt:.2f}s -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", default=["blue_cat", "cute_horse", "fox", "robot"])
    args = ap.parse_args()
    model = load_model()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(OUT, "times.csv")):
        with open(os.path.join(OUT, "times.csv"), "w", newline="") as f:
            csv.writer(f).writerow(["image", "backend", "precision", "wall_s", "rc"])
    for img in args.images:
        reference(model, img)


if __name__ == "__main__":
    main()
