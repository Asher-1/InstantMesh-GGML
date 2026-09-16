"""Run Zero123++ (diffusion) to generate real multi-view images from a single image,
then emit the [V,3,224,224] normalized + [V,16] camera binaries for examples/instantmesh.

Mirrors run.py Stage 1 (background removal -> Zero123++ -> 6-view grid -> split),
reusing the white-background UNet from TencentARC/InstantMesh.

  python3 -m cpp_ggml.convert.gen_multiview --image examples/robot.jpg \
      --out-dir /tmp/mv --device cuda --steps 75

writes /tmp/mv/image.bin and /tmp/mv/camera.bin.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download

from .prep_input import default_cameras
from src.utils.infer_util import remove_background, resize_foreground

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", default="/tmp/mv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=75)
    ap.add_argument("--no_rembg", action="store_true")
    ap.add_argument("--white-bg-unet", default=None,
                    help="optional local path to diffusion_pytorch_model.bin")
    args = ap.parse_args()

    device = args.device
    print("loading Zero123++ pipeline ...")
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", custom_pipeline="zero123plus",
        torch_dtype=torch.float16, trust_remote_code=True,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing")
    if args.white_bg_unet:
        unet_path = args.white_bg_unet
    else:
        unet_path = hf_hub_download(repo_id="TencentARC/InstantMesh",
                                    filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_path, map_location="cpu")
    pipeline.unet.load_state_dict(state_dict, strict=True)
    pipeline = pipeline.to(device)
    print("pipeline ready")

    rembg_session = None if args.no_rembg else __import__("rembg").new_session()
    input_image = Image.open(args.image).convert("RGB")
    if not args.no_rembg:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    output_image = pipeline(input_image, num_inference_steps=args.steps).images[0]
    images = np.asarray(output_image, dtype=np.float32) / 255.0           # (3,960,640)
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)  # (6,3,320,320)
    images = torch.nn.functional.interpolate(images, size=224, mode="bilinear",
                                             align_corners=False).clamp(0, 1)
    norm = (images - torch.from_numpy(IMAGENET_MEAN)[:, None, None]) / torch.from_numpy(IMAGENET_STD)[:, None, None]
    norm = norm.numpy().astype(np.float32)  # [V,3,224,224]

    cameras = default_cameras()[: images.shape[0]]

    os.makedirs(args.out_dir, exist_ok=True)
    norm.tofile(os.path.join(args.out_dir, "image.bin"))
    cameras.tofile(os.path.join(args.out_dir, "camera.bin"))
    print(f"wrote {args.out_dir}/image.bin [{norm.shape}], camera.bin [{cameras.shape}]")


if __name__ == "__main__":
    main()