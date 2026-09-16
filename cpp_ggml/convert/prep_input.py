"""Prepare the multi-view input binaries for the end-to-end instantmesh CLI.

Turns a single input image into the [V,3,224,224] RGB + [V,16] camera inputs
that examples/instantmesh.cpp consumes. The reconstruction model (DINO ->
TriplaneTransformer -> OSGDecoder -> FlexiCubes) is trained on multi-view
conditioning; each view is passed through the same image here (a stand-in for
the Zero123++ multiview generator), which is enough to run the full C++ chain
and obtain a mesh.

  python3 -m cpp_ggml.convert.prep_input --image examples/robot.jpg \
      --views 6 --out-dir /tmp/input

writes /tmp/input/image.bin and /tmp/input/camera.bin.
"""
import argparse
import os

import numpy as np
from PIL import Image

# ImageNet normalization (matches convert/parity_dino.py).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path, size=224):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)  # [3,H,W]
    arr = (arr - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return arr


def default_cameras(radius=4.0, fov=30.0):
    """Mirror get_zero123plus_input_cameras: azimuth=[30,90,150,210,270,330],
    elevation=[20,-10,20,-10,20,-10], intrinsics [fx,fy,cx,cy]."""
    az = np.array([30, 90, 150, 210, 270, 330], dtype=np.float64)
    el = np.array([20, -10, 20, -10, 20, -10], dtype=np.float64)
    az = np.deg2rad(az)
    el = np.deg2rad(el)
    pos = np.stack([
        radius * np.cos(el) * np.cos(az),
        radius * np.cos(el) * np.sin(az),
        radius * np.sin(el),
    ], axis=-1)  # [6,3]

    up = np.array([0, 0, 1], dtype=np.float64)
    cams = []
    focal = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    for p in pos:
        z = p / np.linalg.norm(p)
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        cams.append([x[0], y[0], z[0], p[0],
                     x[1], y[1], z[1], p[1],
                     x[2], y[2], z[2], p[2],
                     focal, focal, 0.5, 0.5])
    return np.asarray(cams, dtype=np.float32)  # [6,16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image (jpg/png)")
    ap.add_argument("--views", type=int, default=6, help="number of views to emit")
    ap.add_argument("--out-dir", default="/tmp/input")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    img = load_image(args.image, args.size)  # [3, H, W]
    images = np.repeat(img[None], args.views, axis=0)  # [V,3,H,W]
    cameras = default_cameras()
    if cameras.shape[0] < args.views:
        cameras = np.repeat(cameras, np.ceil(args.views / cameras.shape[0]).astype(int), axis=0)[:args.views]
    cameras = cameras[:args.views]

    os.makedirs(args.out_dir, exist_ok=True)
    images.tofile(os.path.join(args.out_dir, "image.bin"))
    cameras.tofile(os.path.join(args.out_dir, "camera.bin"))
    print(f"wrote {args.out_dir}/image.bin [{images.shape}], camera.bin [{cameras.shape}]")


if __name__ == "__main__":
    main()