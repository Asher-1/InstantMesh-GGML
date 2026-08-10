"""Parity check for the C++ FlexiCubes mesh-extraction port.

Reads the synthesizer outputs (verts/cubes/sdf/deformation/weight) and runs the
real InstantMesh FlexiCubes `get_mesh` to produce reference vertices+faces, then
compares against the C++ `flexicubes` example output.

Usage:
 1. (reference) generate synth refs first:
      python3 -m convert.parity_synth generate --ckpt ckpts/instant_mesh_large.ckpt \
          --planes /tmp/planes.bin --grid-res 8 --scale 2.1 --out-dir /tmp/synth_ref
 2. Generate reference mesh (this script):
      python3 -m convert.parity_flexicubes --in-dir /tmp/synth_ref --grid-res 8 \
          --out-dir /tmp/mesh_ref
 3. Run C++:
      ./build/flexicubes --in-dir /tmp/synth_ref --grid-res 8 --out-dir /tmp/mesh_cpp
 4. Compare:
      python3 -m convert.parity_flexicubes --compare /tmp/mesh_ref /tmp/mesh_cpp
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.geometry.rep_3d.flexicubes_geometry import FlexiCubesGeometry


def generate(args):
    grid_res = args.grid_res
    fc = FlexiCubesGeometry(grid_res=grid_res, scale=1.0, device="cpu")
    verts = np.fromfile(os.path.join(args.in_dir, "verts.bin"), dtype=np.float32)
    sdf = torch.from_numpy(np.fromfile(os.path.join(args.in_dir, "sdf.bin"), dtype=np.float32))
    deform = torch.from_numpy(np.fromfile(os.path.join(args.in_dir, "deformation.bin"), dtype=np.float32))
    weight = torch.from_numpy(np.fromfile(os.path.join(args.in_dir, "weight.bin"), dtype=np.float32))

    M = verts.size // 3
    v_deformed = torch.from_numpy(verts.reshape(M, 3)) + deform.reshape(M, 3)
    n_cubes = weight.size(0) // 21
    weight = weight.reshape(n_cubes, 21)
    indices = fc.indices

    mesh_v, mesh_f, _reg = fc.get_mesh(
        v_deformed, sdf.squeeze(-1), with_uv=False, indices=indices,
        weight_n=weight, is_training=False)
    mesh_v = mesh_v.detach().numpy()
    mesh_f = mesh_f.detach().numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    mesh_v.reshape(-1).astype(np.float32).tofile(os.path.join(args.out_dir, "vertices.bin"))
    mesh_f.reshape(-1).astype(np.int32).tofile(os.path.join(args.out_dir, "faces.bin"))
    print(f"ref mesh: V={mesh_v.shape[0]} F={mesh_f.shape[0]}")


def compare(args):
    rv = np.fromfile(os.path.join(args.ref, "vertices.bin"), dtype=np.float32).reshape(-1, 3)
    rf = np.fromfile(os.path.join(args.ref, "faces.bin"), dtype=np.int32).reshape(-1, 3)
    cv = np.fromfile(os.path.join(args.cpp, "vertices.bin"), dtype=np.float32).reshape(-1, 3)
    cf = np.fromfile(os.path.join(args.cpp, "faces.bin"), dtype=np.int32).reshape(-1, 3)
    ok = True
    if cv.shape != rv.shape:
        print(f"VERTEX COUNT mismatch: ref={rv.shape} cpp={cv.shape}"); ok = False
    elif cf.shape != rf.shape:
        print(f"FACE COUNT mismatch: ref={rf.shape} cpp={cf.shape}"); ok = False
    else:
        d = np.abs(rv - cv)
        print(f"vertices: max_abs={d.max():.3e} mean_abs={d.mean():.3e} (scale={rv.std():.2f})")
        ok &= d.mean() / (rv.std() + 1e-9) < 1e-4
        if np.array_equal(rf, cf):
            print("faces: identical")
        else:
            # faces may differ only if vertex indexing differs; check sorted match
            print("faces: INDEX MISMATCH (may be vertex-permutation)")
            ok = False
    print("PARITY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("generate")
    p.add_argument("--in-dir", default="/tmp/synth_ref")
    p.add_argument("--grid-res", type=int, default=8)
    p.add_argument("--out-dir", default="/tmp/mesh_ref")
    c = sub.add_parser("compare")
    c.add_argument("--ref", default="/tmp/mesh_ref")
    c.add_argument("--cpp", default="/tmp/mesh_cpp")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args)
    elif args.cmd == "compare":
        sys.exit(compare(args))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()