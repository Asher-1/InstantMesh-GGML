# IM_LRM_PROBE: NaN Bisection Probe Guide

`IM_LRM_PROBE=1` is the numeric diagnostic probe built into
`models/lrm_transformer.cpp`. It captures NaN / numeric anomalies at every
node of the compute graph in a single forward pass, quickly pinpointing the
exact layer and op that trigger "output is all NaN" problems. This document
covers usage, the localization methodology, and how to port the pattern to
other models.

## 1. Quick start

```bash
cd cpp_ggml
IM_LRM_PROBE=1 ./build-gpu/lrm_transformer --device cpu \
    models/gguf/lrm_transformer_f16.gguf
```

Without `IM_LRM_PROBE` the probe is fully off (zero overhead: no taps
registered, no printing, `ggml_set_output` not called — memory-reuse
behavior matches the normal path).

## 2. Reading the output

Each run prints two kinds of information:

```text
probe pre-compute cond: amax=0.25 size=151296    # input write self-check
probe init_x      : nan=0/3145728 amax=0.302002  # initial token stream (pos_embed)
probe L0.norm1    : nan=0/3145728 amax=4.09583   # L0 cross-attn norm1 output
probe L0.q        : nan=0/3145728 amax=10.4975   # q/k/v projections
probe L0.k        : nan=201728/201728 amax=0     # ← all NaN, problem is here
probe L0.xattn.qh : ...                          # q/k/v after to_head reorder
probe L0.xattn.fa : ...                          # raw flash_attn_ext output
probe L0.outproj  : ...                          # out_proj output
probe L0_cross    : ...                          # after the cross-attn residual
probe L0_self     : ...                          # after the self-attn residual
probe layer 0     : ...                          # after the MLP residual (full layer output)
...
probe layer 15    : ...                          # last layer
```

Each tap reports three things:

| Field | Meaning | Interpretation |
|---|---|---|
| `nan=N/total` | NaN elements / total elements | the first tap with `nan>0` is either the pollution source or its direct downstream |
| `amax` | max abs of non-NaN elements | `amax=0` with all-NaN means the whole block is polluted; an abnormally huge `amax` (e.g. >1e10) means the input is garbage data that has not overflowed yet |
| tap name | label registered at build time | see the tap list in §3 |

**Key interpretation rule**: NaN propagates along the residual connections,
so the *first node that produces NaN* is the source — every later layer will
show all-NaN. Do not be alarmed by a wall of NaN; only the earliest one
matters.

## 3. Tap list & bisection workflow

### Round 1: full-layer scan

Look only at `layer N` / `init_x` first. Determine which layer starts
producing NaN (e.g. `layer 0` all-NaN while `init_x` is fine → the problem is
inside L0).

### Round 2: L0 sub-stage refinement

Round 1 already includes fine-grained L0 taps; read them directly:

1. `L0.norm1` — LayerNorm output. NaN → the input stream is already polluted (look further back).
2. `L0.q` / `L0.k` / `L0.v` — the three projection `mul_mat` outputs.
   - Only k/v NaN while q is fine → the problem is in k/v's input (the external cond) or its weights.
3. `L0.xattn.qh/kh/vh` — after the head reorder (reshape/permute/cont). Projections fine but these NaN → a problem in the reordering ops.
4. `L0.xattn.fa` — raw `ggml_flash_attn_ext` output. NaN while q/k/v are fine → the attention computation itself (scale, mask, type support).
5. `L0.outproj` — after out_proj.

### Round 3: input / weight self-check

When round 2 points at an input or a weight, check these special taps (they
read the loader-allocated buffers directly, no compute involved):

- `cond` — the external input itself. **If amax is abnormally huge, suspect
  the host-side data first.**
- `L0.cq_w` / `L0.ck_w` / `L0.cv_w` — cross-attention weights (f16 weights
  are converted to f32 before statistics).

### Case study: size_t underflow → all-NaN in f16

A real problem localized in 2026-09, which walked the full workflow above:

1. Round 1: `init_x` fine (amax 0.302), all-NaN from `layer 0` on → inside L0.
2. Round 2: `L0.q` fine (f16 weights × f32 activations), `L0.k`/`L0.v` all-NaN →
   rules out op/backend issues; points at k/v's shared input `cond` or its weights.
3. Round 3: all weights fine; `cond` read back amax=9.2e17 (should be 0.25) →
   the host-side `tensor_set` data itself was garbage.
4. Root cause: the tool's synthetic input `((i % 10) - 5) * 0.05f` used a
   `size_t` `i`; when `i%10 < 5` the unsigned underflow produced ~2^64; the
   f16 matmul converts f32 inputs to f16, so 9.2e17 → inf → NaN. The f32 model
   does not convert to f16, and the huge values were masked by softmax
   saturation + LayerNorm normalization, so the final stats looked "normal".

**Lesson**: when judging "which precision/backend is broken", do not look only
at the final output's mean/std — softmax saturation and LayerNorm can mask
garbage in the middle. An f32 model showing "normal" does not mean it did not
eat the same bad input.

## 4. Implementation notes (porting to other models)

The probe lives in `src/models/lrm_transformer.cpp`, ~40 lines at its core.
Porting checklist:

1. **Single global sink**: `ProbeSink g_probe` (two vectors, taps + tags;
   tags must be stored as `std::string` — do not store pointers to stack
   char buffers in the vector, loop reuse overwrites the labels).
2. **Register = mark output**: `ggml_set_output(t)`. ggml-alloc never
   recycles output-node storage; this is the precondition for every tap to
   still be readable after one full compute.
3. **Single full-graph compute**: never "compute up to layer N" in multiple
   truncated computes — leftover allocated pointers after gallocr teardown
   cause use-after-free segfaults (the source of `tensor buffer not set` /
   memmove crashes).
4. **Read back after compute**: `ggml_backend_tensor_get` per tap,
   interpreted by `t->type` (F32 straight memcpy; F16 via
   `ggml_fp16_to_fp32`).
5. **Weight taps read the loader buffer directly**: weight nodes already
   have a buffer, gallocr skips their reallocation, so a plain get works.
6. **Overhead**: output nodes block memory reuse and grow peak memory; use
   the probe for diagnosis only — never enable it in perf tests or
   production paths.

### Environment variable conventions

| Variable | Effect |
|---|---|
| `IM_LRM_PROBE=1` | enable the lrm_transformer NaN probe |
| `IM_UNET_PROBE=1` | UNet prints a one-line summary per forward (w/r eps nan/amax) |
| `IM_UNET_PROBE=2` | UNet additionally registers all stage taps and prints per tap |
| `IM_UNET_PROBE_T=<int>` | UNet stage taps print only on the matching timestep call |
| `IM_DUMP_STAGES=<substr>` | substring filter on UNet stage tap names (combine with probe=2) |

## 5. zero123pp UNet probe (IM_UNET_PROBE)

Diffusion models have a different call pattern from LRM: the UNet runs one
forward per denoising step (w/r passes in the same graph), so 75 steps =
150 passes / 75 computes. The probe therefore has a two-level design
(implemented in `src/models/unet.cpp`, reusing the `IM_VAE_DUMP` tap
infrastructure):

### Usage 1: full-run summary scan (`=1`)

```bash
IM_UNET_PROBE=1 ./build-gpu/zero123pp --image ../examples/blue_cat.png \
    --device cpu --steps 75 --seed 42 2>&1 | grep unet_probe
```

Each call prints one line per pass:

```text
unet_probe w.eps          nan=0/32768 amax=4.77203    # w-pass eps (cond scale)
unet_probe r.eps          nan=0/76800 amax=0.842033   # r-pass eps (denoise scale)
```

75 lines show at a glance from which timestep NaN starts. An `amax` that
grows abnormally step over step is also worth watching (numeric divergence
usually precedes NaN).

### Usage 2: single-step stage bisection (`=2`)

Once the summary locates the starting timestep, lock that step with
`IM_UNET_PROBE_T` and look at the internal taps:

```bash
IM_UNET_PROBE=2 IM_UNET_PROBE_T=979 IM_DUMP_STAGES=d0 \
    ./build-gpu/zero123pp --device cpu --steps 75 --seed 42
```

The taps cover two kinds of points (names consistent with `IM_VAE_DUMP`):

- **dmark backbone**: `w.conv_in` / `w.d0.r0` / `w.d0.attn0` / ... / `w.mid` /
  `wup0.upc` / `w.conv_out` (r-pass uses the same names with the `r.`
  prefix) — resblock/attention outputs and the upsample chain; locates the
  first anomalous block.
- **u_mark internals**: `*.norm1` / `*.silu1` / `*.conv1` / `*.te` /
  `*.gn2_bare` / `*.norm2` / `*.conv2` (inside a resnet) and `*.tgn` /
  `*.tgather` / `*.tproj` (inside a transformer) — locates the exact op
  within a block.

The `IM_DUMP_STAGES` substring filter shrinks the print volume (e.g.
`IM_DUMP_STAGES=mid` shows only mid_block-related taps).

### Differences from the LRM probe (important)

| Aspect | LRM (single call) | UNet (150 calls) |
|---|---|---|
| Detailed-tap gate | not needed | `IM_UNET_PROBE_T` (detailed only on the 1st call by default) |
| Tap implementation | `ggml_set_output` on the original nodes | reuses `IM_VAE_DUMP`'s `ggml_cont` copies, expanded immediately |
| Memory overhead | outputs block reuse, one-shot | copies add peak memory, accumulating over the 75-step loop |

Two UNet-specific caveats:

1. **Only the first call prints in detail by default** (when
   `IM_UNET_PROBE_T` is unset). If NaN appears only in later steps, lock it
   with `IM_UNET_PROBE_T=<that timestep>`.
2. **Run stage taps on CPU preferentially**: each tap is a `ggml_cont` copy;
   ~50 taps add hundreds of MB of peak memory, and the 12GB Vulkan path's
   VAE decode is already close to OOM — full taps can blow VRAM outright.
   Summary mode (`=1`) only `set_output`s the graph tail, negligible
   overhead, safe on GPU.

### Verification record (2026-09-18)

- `IM_UNET_PROBE=1/2` 2-step short runs print correctly; all taps `nan=0`,
  amax reasonable.
- probe1/probe2 final latents are **bit-exact** with the non-probe run —
  the probe has zero numerical impact.
