# IQ1_M routed experts on gfx1151

Why this branch exists: the ISTA-DASLab GSQ-RCO 1017 release is the only
DeepSeek-V4.1-Flash-1017 quantisation small enough to be interesting on a
122.2 GiB unified pool — about 112.44 GiB for the IQ1_M main weights, against
246 GiB for the smallest alternative (mrfakename Q2_K) and 311.80 GiB for
unsloth UD-Q2_K_XL. Its routed experts are IQ1_M, which ds4 refuses to load.
This is upstream issue #1268.

## What already works, and what does not

The gap is narrower than it looks from the kernel symbol counts. Verified in
this tree:

| piece | state |
|---|---|
| `block_iq1_m` layout | present, `cuda/mmq/ggml-common.h:423` |
| `vec_dot_iq1_m_q8_1` | present and `__device__`, `cuda/mmq/vecdotq.cuh:1223` |
| `VDR_IQ1_M_Q8_1_MMVQ` / `_MMQ` | both defined, `cuda/mmq/vecdotq.cuh:1220` |
| generic ggml mat-vec | instantiated, `cuda/mmq/mmvq.cu:1018` |
| `ds4_mmq_vdr_mmvq_value<IQ1_M>` | **added on this branch** |
| `ds4_mmq_vec_dot_q8_1` IQ1_M arm | **added on this branch** |
| `load_tiles_iq1_m` for tiled MMQ | **missing, and hard — see below** |
| `DS4_TENSOR_IQ1_M` in `ds4.c` | **missing** |

So GPU *decode* for IQ1_M is reachable with small changes. GPU *prefill*
through the tiled MMQ path is not.

## Why there is no `load_tiles_iq1_m`

`load_tiles_iq1_s` exists (`cuda/mmq/mmq.cuh:3308`) and shares the
`iq1s_grid_gpu` codebook, so the grid-lookup half of an IQ1_M loader is nearly
a copy. The scales are the problem. Compare the blocks:

```c
block_iq1_s { ggml_half d; uint8_t qs[32]; uint16_t qh[8];  }
block_iq1_m {              uint8_t qs[32]; uint8_t  qh[16]; uint8_t scales[8]; }
```

IQ1_M has **no `d` field**. Per `vec_dot_iq1_m_q8_1`, the superblock scale is
reassembled from one nibble in each of four `uint16`s:

```c
scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
          | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
```

and the sub-scales are two 3-bit fields per 32-element group, i.e. **one scale
per 16 values**:

```c
const int tmp = sc[iqs/2] >> (6*(iqs%2));
sc0 = 2*((tmp >> 0) & 0x07) + 1;   // first 16
sc1 = 2*((tmp >> 3) & 0x07) + 1;   // second 16
```

while the delta sign lives in bit 3 of each `qh` nibble, i.e. **one delta per 8
values**:

```c
delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
```

The MMQ tile carries one `half2` per 32-element group — `(d, d*delta)` for
IQ1_S, where delta is uniform across the 32. IQ1_M needs scale granularity of
16 and delta granularity of 8, so one `half2` per 32 cannot represent it.

The obvious dodge does not work either: the sub-scales are odd integers
`2k+1` up to 15, and the grid values occupy the full 0..15 nibble range, so
folding the sub-scale into the stored values overflows the `int8` lanes that
`dp4a` consumes. A correct loader therefore needs a **new tile layout** with
finer scale storage, following the precedent of the K-quant loaders rather
than `load_tiles_iq1_s`. That is the one genuinely novel piece of work here,
and it is why upstream llama.cpp has an IQ1_S MMQ loader but not an IQ1_M one.

## Plan

Staged so that each step is independently verifiable, cheapest first.

1. **Vector path in the mmq TU** — done on this branch. Four lines:
   `ds4_mmq_vdr_mmvq_value<GGML_TYPE_IQ1_M>` plus an `if constexpr` arm in
   `ds4_mmq_vec_dot_q8_1`. Six `ds4_mmq_moe_*_vec_impl` sites read the former,
   so this is what makes an IQ1_M expert instantiation possible at all. Note
   that template specialisations are only instantiated on use, so compiling
   this TU proves syntax and symbol resolution, not behaviour.

2. **`DS4_TENSOR_IQ1_M` plumbing in `ds4.c`** — add `= 29` to the enum at
   `ds4.c:2360`, extend `tensor_is_routed_expert_type()` (`ds4.c:5085`) and
   `routed_expert_block_bytes()` (`ds4.c:5096`, 56 bytes). The type table at
   `ds4.c:2351` already carries `{"iq1_m", 256, 56}`.

   Scope warning: **58 sites in `ds4.c` hardcode `DS4_TENSOR_IQ2_XXS`**, and
   they are not one switch — they run through the deepseek41 routed-expert
   path (`8358`, `11710`, `13607`, `16693`, `20183`, `22665`, …). This branch
   deliberately touches only the sites the streaming path reaches, and leaves
   the rest, so the diff stays reviewable. Full genericity is a later project.

3. **Parity test before any tile work.** `cuda/mmq/test/test_mmq_parity.cu`
   and `test_mmq_soa_tiles.cu` already exist as harnesses, so IQ1_M
   correctness is checkable against the CPU reference in
   `gguf-tools/quants.c` without downloading 112 GiB.

4. **`load_tiles_iq1_m` plus a finer-grained tile layout** — the hard step
   above. Only worth starting once 1-3 hold.

5. **gfx1151 fast paths are out of scope.** The SoA repack, D2R fusion and the
   `cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151` special case are gated on
   `type == GGML_TYPE_IQ2_XXS` (`cuda/mmq/ds4_mmq.cu:1402, 1512, 1520, 1727`).
   IQ1_M will take the generic path, so expect it to be slower per byte than
   IQ2_XXS despite moving ~13% fewer bytes. Matching those is a second project.

## What this does and does not buy

It makes GSQ-RCO loadable on the SSD-streaming path: roughly 232 GiB on disk
(112.44 main + 103.00 Engram-IQ4_NL) against the 341 GiB
`DeepSeek-V4.1-Flash-Q2.gguf` in use today, at better claimed quality.

It does **not** obviously unlock resident DSpark, and the arithmetic is worth
stating because it is closer than it first appears. `docs/STRIX_HALO.md:113`
gives 151.76 GiB of resident target spans plus 7.42 GiB of support model for
V4.1 Q2. Substituting the GSQ-RCO main weights gives 112.44 + 7.42 =
**119.86 GiB against a 122.2 GiB pool** — about 2.3 GiB left, and that figure
is explicitly "before context and runtime buffers". The DSpark example in that
doc uses `--ctx 8192`, which is small for a reason. Two things are unverified
and either would sink it: whether runtime buffers fit in what remains, and
whether the pinned MXFP4 support GGUF is a valid draft for a **1017** target
when it was released against V4.1. Treat resident DSpark as one measurement to
make, not as the goal.
