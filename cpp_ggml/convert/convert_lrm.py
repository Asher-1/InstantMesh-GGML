"""Convert the InstantMesh LRM reconstruction checkpoint to GGUF.

Splits the single `.ckpt` into per-component GGUF files so the C++ runtime can
load each stage independently (and pin the 3D geometry/render stages to CPU):

  dino_<p>.gguf          encoder.*        (DINO ViT-B/16)
  lrm_transformer_<p>.gguf  transformer.*  (TriplaneTransformer)
  synthesizer_<p>.gguf   synthesizer.*    (NeuralRender; CPU-pinned)
  geometry_<p>.gguf      geometry.*       (FlexiCubes; CPU-pinned)

Run once per precision: `-p f32|f16|q8`.
"""

import logging
import os

from .convert_common import (
    add_kv, begin_gguf_writer, finalize_gguf, ggml_type_for, load_state_dict,
    parse_common_args, write_tensor,
)

logger = logging.getLogger("convert-lrm")

# TriplaneTransformer hparams (instant-mesh-large.yaml).
LRM_HARAMS = {
    "transformer.inner_dim": 1024,
    "transformer.num_layers": 16,
    "transformer.num_heads": 16,
    "transformer.cond_dim": 768,
    "transformer.triplane_low_res": 32,
    "transformer.triplane_high_res": 64,
    "transformer.triplane_dim": 80,
}
# DINO ViT-B/16 hparams (facebook/dino-vitb16).
DINO_HARAMS = {
    "dino.hidden_size": 768,
    "dino.num_hidden_layers": 12,
    "dino.num_attention_heads": 12,
    "dino.image_size": 224,
    "dino.patch_size": 16,
}


def _write_component(ckpt, component_prefix, arch, hparams, out_dir, precision):
    tensors = {k: v for k, v in ckpt.items() if k.startswith(component_prefix)}
    if not tensors:
        logger.warning("no tensors for %s", component_prefix)
        return
    fname = os.path.join(out_dir, f"{arch}_{precision}.gguf")
    writer = begin_gguf_writer(arch, fname)
    for k, v in hparams.items():
        add_kv(writer, k, v)
    for k, v in tensors.items():
        short = k[len(component_prefix):]
        # NOTE: no manual permute for conv kernels. GGUFWriter.add_tensor stores
        # the numpy shape reversed (ggml ne[0] = numpy last dim), so a torch
        # [OC, IC, KH, KW] weight is written as ggml [KW, KH, IC, OC] — exactly
        # the layout ggml_conv_2d expects. permuting here would corrupt it.
        write_tensor(writer, short, v, ggml_type_for(short, precision))
    finalize_gguf(writer, fname)


def main():
    args = parse_common_args().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = load_state_dict(args.ckpt, args.prefix or "lrm_generator.")

    # DINO encoder.
    _write_component(ckpt, "encoder.", "dino", DINO_HARAMS, args.out_dir, args.precision)
    # TriplaneTransformer.
    _write_component(ckpt, "transformer.", "lrm_transformer", LRM_HARAMS, args.out_dir, args.precision)
    # Geometry + renderer (passthrough, CPU-pinned for now).
    for comp, arch in (("geometry.", "geometry"), ("synthesizer.", "synthesizer")):
        _write_component(ckpt, comp, arch, {}, args.out_dir, args.precision)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()