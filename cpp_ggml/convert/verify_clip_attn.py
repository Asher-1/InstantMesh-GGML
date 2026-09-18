#!/usr/bin/env python3
"""Verify whether CLIP's first error-amplification point (softmax attention
reduction) comes from fp32 reduction rounding, by recomputing the attention
path at fp64 / fp32-Kahan from backend-dumped q/k/v.

Layout (matches clip_vision.cpp dumps): ggml dump files are contiguous
[seq=257, C=1280] memory (C fastest, C = nh*hd head-major: nh outer, hd inner),
so load() transposes to [C, seq]. Attention = softmax(k^T q / sqrt(hd)) over
kv, then o = att^T v; att must be laid out [nh, q, kv] before contracting with
v [nh, kv, hd] (see attn()).

Usage:
  IM_CV_L0=1 ./build-gpu/test_clip_vision vulkan ; mv /tmp/cv_*.bin /tmp/l0_vk/
  IM_CV_L0=1 ./build-gpu/test_clip_vision cuda   ; mv /tmp/cv_*.bin /tmp/l0_cu/
  python3 cpp_ggml/convert/verify_clip_attn.py /tmp/l0_cu /tmp/l0_vk
"""
import sys

import numpy as np

C, SEQ, NH, HD = 1280, 257, 16, 80
SCALE = 1.0 / np.sqrt(HD)


def load(p):
    # ggml dumps are contiguous [seq, C] memory (C fastest, C = nh*hd head-major);
    # transpose to the [C, seq] shape the analysis below expects.
    return np.fromfile(p, dtype=np.float32).reshape(SEQ, C).T


def head(x):
    return x.reshape(NH, HD, SEQ)


def softmax_fp64(kq):
    kq = kq - kq.max(axis=1, keepdims=True)
    e = np.exp(kq.astype(np.float64))
    return e / e.sum(axis=1, keepdims=True)


def softmax_fp32(kq, kahan=False):
    kq32 = (kq - kq.max(axis=1, keepdims=True)).astype(np.float32)
    e = np.exp(kq32)
    if not kahan:
        s = e.sum(axis=1, dtype=np.float32)  # [nh, seq_q]
    else:
        s = np.zeros((NH, SEQ), np.float32)
        c = np.zeros((NH, SEQ), np.float32)
        for i in range(SEQ):  # Kahan along kv (axis 1), vectorized over (nh, q)
            y = e[:, i, :] - c
            t = s + y
            c = (t - s) - y
            s = t
    return (e / s[:, None, :]).astype(np.float64)


def attn(q, k, v, softmax_fn, att_dtype=np.float64):
    Q = head(q.astype(np.float64))                     # [nh, hd, seq_q]
    K = head(k.astype(np.float64))                     # [nh, hd, seq_kv]
    V = head(v.astype(np.float64)).transpose(0, 2, 1)  # [nh, seq_kv, hd]
    kq = np.einsum('hds,hdt->hst', K, Q, dtype=np.float64) * SCALE  # [nh, kv, q]
    att = softmax_fn(kq)                               # [nh, kv, q]
    # contract over kv: att must be [nh, q, kv] against V [nh, kv, hd]
    att_q = att.transpose(0, 2, 1)
    if att_dtype == np.float32:
        o = np.einsum('hst,htd->hsd', att_q.astype(np.float32),
                      V.astype(np.float32), dtype=np.float32).astype(np.float64)
    else:
        o = np.einsum('hst,htd->hsd', att_q, V)        # [nh, seq_q, hd]
    # head-major C: c = nh*hd + hd_inner  (matches clip_vision.cpp cont/reshape)
    return o.transpose(0, 2, 1).reshape(C, SEQ)


def stats(name, a, b):
    d = np.abs(a - b)
    print(f"{name:44s} max_abs={d.max():.3e} mean_abs={d.mean():.3e}")


def main() -> int:
    cu_dir, vk_dir = sys.argv[1], sys.argv[2]
    cu = {n: load(f"{cu_dir}/cv_{n}.bin") for n in ("q", "k", "v", "attn_raw")}
    vk = {n: load(f"{vk_dir}/cv_{n}.bin") for n in ("q", "k", "v", "attn_raw")}

    # fp64 ground truth recomputed from EACH backend's own q/k/v.
    ref_cu = attn(cu["q"], cu["k"], cu["v"], softmax_fp64)
    ref_vk = attn(vk["q"], vk["k"], vk["v"], softmax_fp64)

    print("=== per-backend reduction error (own attn_raw vs own-input fp64) ===")
    stats("CUDA   fp32-path vs fp64", cu["attn_raw"], ref_cu)
    stats("Vulkan fp32-path vs fp64", vk["attn_raw"], ref_vk)
    stats("CUDA vs Vulkan attn_raw", cu["attn_raw"], vk["attn_raw"])
    stats("input-only contribution (fp64 both)", ref_cu, ref_vk)

    for tag, q, k, v, raw in (("CUDA", cu["q"], cu["k"], cu["v"], cu["attn_raw"]),
                              ("Vulkan", vk["q"], vk["k"], vk["v"], vk["attn_raw"])):
        ref = attn(q, k, v, softmax_fp64)
        print(f"\n=== {tag}: ablation vs own-input fp64 ===")
        stats("fp32 softmax(plain) + fp64 attv", attn(q, k, v, lambda kq: softmax_fp32(kq, False)), ref)
        stats("fp32 softmax(kahan)  + fp64 attv", attn(q, k, v, lambda kq: softmax_fp32(kq, True)), ref)
        stats("fp64 softmax + fp32 attv", attn(q, k, v, softmax_fp64, att_dtype=np.float32), ref)
        stats("actual backend output", raw, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
