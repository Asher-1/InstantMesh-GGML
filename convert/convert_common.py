"""Shared GGUF conversion helpers for the InstantMesh models.

Conversion-time only (pure Python + torch); the C++ runtime is PyTorch-free.
Every model is exported as GGUF with a `general.architecture` tag plus its own
hparams KV block, and every weight tensor is written in one of three precisions:

  f32  GGML_TYPE_F32   precision baseline
  f16  GGML_TYPE_F16   default inference
  q8   GGML_TYPE_Q8_0  low VRAM / bandwidth

The same callback is used by all per-model converters so the three precisions
stay consistent across every model in the pipeline.
"""

import argparse
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from gguf import GGMLQuantizationType as QType
from gguf import GGUFValueType as VType
from gguf import GGUFWriter
from gguf import quantize

logger = logging.getLogger("instantmesh-convert")

# Precision -> ggml tensor type used for NON-quantized weight tensors.
PRECISION_TYPES = {
    "f32": QType.F32,
    "f16": QType.F16,
    "q8": QType.Q8_0,
}


def load_state_dict(ckpt_path: str, prefix: str) -> Dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint and strip an optional key prefix.

    As in the native run.py, the LRM checkpoint keys are namespaced under
    `lrm_generator.`; we strip it so C++ can use short, stable tensor names.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if prefix and k.startswith(prefix):
            k = k[len(prefix):]
        out[k] = v.detach().to("cpu")
    return out


def ggml_type_for(name: str, precision: str) -> QType:
    """Choose the ggml tensor type for a weight named `name`.

    Small buffers (gains, biases, norms, embeddings) are always kept as F32 —
    quantizing them gains nothing and costs accuracy. Everything else follows
    the requested precision. This mirrors stable-diffusion.cpp's per-tensor
    policy and keeps numeric parity with PyTorch.
    """
    if precision == "f32":
        return QType.F32
    low = "gain" in name or "bias" in name or "beta" in name or "gamma" in name \
        or "norm" in name or "mean" in name or "var" in name \
        or name.endswith(".weight") is False and name.endswith(".embeddings") \
        or "position_embeddings" in name or "cls_token" in name
    if low:
        return QType.F32
    return PRECISION_TYPES[precision]


def write_tensor(writer: GGUFWriter, name: str, t: torch.Tensor, qtype: QType) -> int:
    """Write one tensor to the GGUF writer, quantizing to `qtype` when needed.

    `add_tensor` writes raw bytes only, so non-F32 types are quantized here with
    `gguf.quantize` first. Quantized buffers change byte-shape, so the original
    torch shape is passed back in for the GGUF tensor info.
    """
    arr = t.detach().to("cpu").numpy()
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    if qtype in (QType.F32,):
        writer.add_tensor(name, arr)
    elif qtype == QType.F16 or arr.ndim != 2 or arr.shape[-1] % 32 != 0:
        # Q8_0 blocks along the last (fastest) dim; only 2D weight matrices whose
        # last dim is a multiple of 32 can be quantized. Conv kernels and small
        # (in < 32) matrices stay F16 even in the q8 variant.
        writer.add_tensor(name, arr.astype(np.float16))
    else:
        # add_tensor infers the GGUF logical shape from the quantized byte
        # shape, so we must NOT pass a raw_shape here.
        q = quantize(np.ascontiguousarray(arr, dtype=np.float32), qtype)
        writer.add_tensor(name, q, raw_dtype=qtype)
    return int(arr.nbytes)


def add_kv(writer: GGUFWriter, key: str, value) -> None:
    """Write a metadata KV with the correct GGUF type for the Python value."""
    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, int):
        writer.add_int64(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, str):
        writer.add_string(key, value)
    else:
        raise TypeError(f"unsupported KV value type for {key}: {type(value)}")


def begin_gguf_writer(arch: str, fname: str) -> GGUFWriter:
    """Create a GGUFWriter. The architecture tag is written by the constructor."""
    return GGUFWriter(fname, arch)


def finalize_gguf(writer: GGUFWriter, fname: str) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    logger.info("wrote %s", fname)


def parse_common_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=str, help="PyTorch checkpoint (.ckpt/.bin)")
    p.add_argument("-o", "--out-dir", type=str, default="models/gguf")
    p.add_argument("-p", "--precision", type=str, default="f16",
                   choices=["f32", "f16", "q8"],
                   help="weight precision (default: f16)")
    p.add_argument("--prefix", type=str, default="",
                   help="state_dict key prefix to strip (e.g. lrm_generator.)")
    return p