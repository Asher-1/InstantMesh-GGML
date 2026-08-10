"""Parity check for the C++ geometry-prediction (OSGDecoder) ggml port.

Ports the InstantMesh geometry branch (FlexiCubes). This script produces the
reference outputs for the *neural* part that carries weights -- the triplane
sampling + OSGDecoder that turns planes into (sdf, deformation, weight):

  planes [N,3,C,H,W] -> sample_from_planes -> sampled [N,3,M,C]
  sampled.permute(0,2,1,3).reshape(N,M,3C)
    -> net_sdf(sampled)            -> [N,M,1]
    -> net_deformation(sampled)    -> [N,M,3]
  net_weight(grid_features)        -> [N,n_cubes,21]   (grid_features gathered
    at the 8 cube corners per cube, x0.1)

It also dumps the fixed voxel grid (verts + cube indices) so the C++ side can
reuse the exact same geometry without re-deriving torch's unique ordering.

Usage:
 1. Generate the reference:
      python3 -m convert.parity_synth --ckpt ckpts/instant_mesh_large.ckpt \
          --planes /tmp/planes.bin --grid-res 8 --scale 2.1 \
          --out-dir /tmp/synth_ref
 2. Run the C++ graph on the same planes:
      ./build/synthesizer --device cpu models/gguf/synthesizer_f32.gguf \
          --planes /tmp/planes.bin --grid-res 8 --scale 2.1 \
          --in-dir /tmp/synth_ref --out-dir /tmp/synth_cpp
 3. Compare:  python3 -m convert.parity_synth --compare /tmp/synth_ref /tmp/synth_cpp
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_planes():
    return torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                         [[0, 0, 1], [0, 1, 0], [1, 0, 0]]], dtype=torch.float32)


def project_onto_planes(planes, coordinates):
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N * n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N * n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


def sample_from_planes(plane_axes, plane_features, coordinates, box_warp=2.0):
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N * n_planes, C, H, W)
    coordinates = (2 / box_warp) * coordinates
    projected = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    out = F.grid_sample(plane_features, projected, mode='bilinear',
                        padding_mode='zeros', align_corners=False)
    return out.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)


def construct_voxel_grid(res, device='cpu'):
    cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                                 [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                                dtype=torch.float, device=device)
    base_cube_f = torch.arange(8, device=device)
    if isinstance(res, int):
        res = (res, res, res)
    voxel_grid_template = torch.ones(res, device=device)
    res_t = torch.tensor([res], dtype=torch.float, device=device)
    coords = torch.nonzero(voxel_grid_template).float() / res_t
    verts = (cube_corners.unsqueeze(0) / res_t + coords.unsqueeze(1)).reshape(-1, 3)
    cubes = (base_cube_f.unsqueeze(0) +
             torch.arange(coords.shape[0], device=device).unsqueeze(1) * 8).reshape(-1)
    verts_rounded = torch.round(verts * 1e5) / 1e5
    verts_unique, inverse_indices = torch.unique(verts_rounded, dim=0, return_inverse=True)
    cubes = inverse_indices[cubes.reshape(-1)].reshape(-1, 8)
    return verts_unique - 0.5, cubes


class RefOSGDecoder(nn.Module):
    """Self-contained OSGDecoder keyed by the GGUF tensor names (decoder.*)."""

    def __init__(self, n_features=80, hidden_dim=64, num_layers=4):
        super().__init__()
        self.net_sdf = self._mlp(3 * n_features, hidden_dim, 1, num_layers)
        self.net_rgb = self._mlp(3 * n_features, hidden_dim, 3, num_layers)
        self.net_deformation = self._mlp(3 * n_features, hidden_dim, 3, num_layers)
        self.net_weight = self._mlp(8 * 3 * n_features, hidden_dim, 21, num_layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _mlp(self, fin, hidden, fout, num_layers):
        layers = [nn.Linear(fin, hidden), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, fout)]
        return nn.Sequential(*layers)

    def load_from(self, ckpt, strip="lrm_generator.synthesizer.decoder."):
        sd = torch.load(ckpt, map_location="cpu")
        ck = sd["state_dict"] if "state_dict" in sd else sd
        remap = {}
        for k, v in ck.items():
            if not k.startswith(strip):
                continue
            remap[k[len(strip):]] = v
        self.load_state_dict(remap, strict=True)

    def get_geometry_prediction(self, sampled_features, flexicubes_indices):
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes * _C)
        sdf = self.net_sdf(sampled_features)
        deformation = self.net_deformation(sampled_features)
        grid_features = sampled_features[:, flexicubes_indices.reshape(-1), :]
        grid_features = grid_features.reshape(
            sampled_features.shape[0], flexicubes_indices.shape[0],
            flexicubes_indices.shape[1] * sampled_features.shape[-1])
        weight = self.net_weight(grid_features) * 0.1
        return sdf, deformation, weight


def generate(args):
    device = "cpu"
    model = RefOSGDecoder()
    model.load_from(args.ckpt)
    model.eval()

    verts, cubes = construct_voxel_grid(args.grid_res, device)
    verts = verts * args.scale
    M = verts.shape[0]
    n_cubes = cubes.shape[0]

    planes_np = np.fromfile(args.planes, dtype=np.float32)
    C, H, W = args.plane_dim, args.grid_res, args.grid_res
    planes = torch.from_numpy(planes_np.reshape(1, 3, C, H, W)).float()

    plane_axes = generate_planes()
    sampled = sample_from_planes(plane_axes, planes, verts.unsqueeze(0), box_warp=2.0)
    sdf, deformation, weight = model.get_geometry_prediction(sampled, cubes)

    # net_rgb color reference (raw sigmoid, no MipNeRF affine clamp) on the
    # same sampled features — matches the ggml synthesizer_texture_forward.
    _, n_planes, M, C = sampled.shape
    feats = sampled.permute(0, 2, 1, 3).reshape(1, M, n_planes * C)
    rgb = torch.sigmoid(model.net_rgb(feats))

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        np.asarray(verts.reshape(-1).numpy(), dtype=np.float32).tofile(f"{args.out_dir}/verts.bin")
        np.asarray(cubes.reshape(-1).numpy(), dtype=np.int32).tofile(f"{args.out_dir}/cubes.bin")
        np.asarray(sdf.reshape(-1).numpy(), dtype=np.float32).tofile(f"{args.out_dir}/sdf.bin")
        np.asarray(deformation.reshape(-1).numpy(), dtype=np.float32).tofile(f"{args.out_dir}/deformation.bin")
        np.asarray(weight.reshape(-1).numpy(), dtype=np.float32).tofile(f"{args.out_dir}/weight.bin")
        np.asarray(rgb.reshape(-1).numpy(), dtype=np.float32).tofile(f"{args.out_dir}/rgb.bin")
    print(f"generated: M={M} cubes={n_cubes} sdf[{sdf.numel()}] deform[{deformation.numel()}] weight[{weight.numel()}] rgb[{rgb.numel()}]")


def compare(args):
    def load(name, dt):
        return np.fromfile(os.path.join(args.ref, name), dtype=dt)
    ok = True
    for name, dt in [("sdf", np.float32), ("deformation", np.float32), ("weight", np.float32),
                     ("rgb", np.float32)]:
        a = load(f"ref_{name}.bin" if os.path.exists(os.path.join(args.ref, f"ref_{name}.bin")) else f"{name}.bin", dt)
        b = np.fromfile(os.path.join(args.cpp, name + ".bin"), dtype=dt)
        if a.size != b.size:
            print(f"{name}: SIZE MISMATCH ref={a.size} cpp={b.size}")
            ok = False
            continue
        d = np.abs(a - b)
        std = a.std() if a.size else 1.0
        print(f"{name}: max_abs={d.max():.3e} mean_abs={d.mean():.3e} rel={d.mean()/std:.2e}")
        ok &= d.mean() / std < 1e-3
    print("PARITY " + ("PASS" if ok else "FAIL"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("generate")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--planes", required=True)
    p.add_argument("--grid-res", type=int, default=8)
    p.add_argument("--scale", type=float, default=2.1)
    p.add_argument("--plane-dim", type=int, default=80)
    p.add_argument("--out-dir", default="/tmp/synth_ref")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("cpp")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args)
    elif args.cmd == "compare":
        compare(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()