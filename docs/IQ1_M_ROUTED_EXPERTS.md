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
| `ds4_mmq_iq1_m_moe_vec` entry point | **added on this branch** |
| `DS4_TENSOR_IQ1_M`, `block_iq1_m` in `ds4.c` | **added on this branch** |
| CPU `ds4_dequantize_iq1_m_block` | **added, tested** |
| CPU `ds4_vec_dot_iq1_m_q8_K` | **added, tested** |
| CPU `matvec_iq1_m_experts_{mid,accum}_prequant` | **added, wired into both dispatchers** |
| `load_tiles_iq1_m` for tiled MMQ | **missing, and hard — see below** |
| `tensor_is_routed_expert_type` admits IQ1_M | **deliberately not yet** |
| `ds41_stream_table` per-role typing | **not yet** |
| GPU graph dispatch to `ds4_mmq_iq1_m_moe_vec` | **not yet** |

So GPU *decode* for IQ1_M is reachable with small changes, and the CPU
reference path is complete. GPU *prefill* through the tiled MMQ path is not.

Nothing loads yet: `tensor_is_routed_expert_type` is the gate and it stays shut
until the paths behind it are all in place, so the tree is never in a state
where it accepts a model it cannot run.

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

## What GSQ-RCO actually contains

Read from the GGUF header of shard 1 (a 48 MiB range request, no full
download). Expert dimensions match what ds4 expects for deepseek41 —
gate/up `[5120, 2304, 384]`, down `[2304, 5120, 384]`, i.e. `DS4_N_EMBD`
5120, `DS4_N_FF_EXP` 2304, `DS4_N_EXPERT` 384.

The important part is that **precision is mixed within a layer**:

| layer | gate | up | down |
|---|---|---|---|
| blk.0 - blk.6 | IQ1_M | IQ1_M | IQ1_M |
| blk.7, blk.8 | IQ1_M | IQ1_M | **IQ2_XXS** |
| blk.9 | IQ1_M | IQ1_M | IQ1_M |
| blk.10 | IQ2_XXS | IQ2_XXS | IQ2_XXS |
| blk.11 | IQ2_XXS | IQ2_XXS | **IQ1_M** |

So gate and up agree with each other in every layer seen, but `down` varies
independently in both directions. `ds41_stream_table` currently demands one
exact combination:

```c
if (t[i]->type != (i == 2 ? DS4_TENSOR_Q2_K : DS4_TENSOR_IQ2_XXS) || ...
```

which is wrong for GSQ-RCO twice over — `down` is never Q2_K in it, and
gate/up may be IQ1_M. The fix is to type the three roles independently against
a permitted set rather than against one expected pair. The surrounding
machinery already tolerates `down` differing from gate/up, since that is
exactly the V4.1 Q2 shape (IQ2_XXS gate/up, Q2_K down), so this is a widening
of an existing distinction rather than a new one.

## Trap: two grid conventions

The codebook exists in two forms and they are off by one, so transcribing the
GPU arithmetic onto the CPU table silently biases every value by -1.0.

| table | value alphabet | paired delta |
|---|---|---|
| `iq1s_grid_gpu` (uint32, nibbles) | `{0, 1, 2}` | `-1 +/- IQ1M_DELTA` |
| `iq1s_grid` (uint64, bytes) | `{-1, 0, +1}` | `+/- IQ1M_DELTA` |

The `-1` in the GPU delta exists only to cancel the `+1` offset baked into
`iq1s_grid_gpu`. Entry 1 pins the relation: it is `0x00000002` in the GPU table
and `0xffffffffffffff01` in the CPU one, so `g_gpu = g_cpu + 1`.

This was caught by `tests/test_iq1_m_dequant.c` rather than by reading, and it
is worth noting that one of that test's cases initially agreed with the buggy
code because the error happened to cancel for `g_cpu = 0`. Any future IQ1_M
work should keep a case with `g_cpu = +1`, where nothing cancels.

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
   they are not one switch. `scripts/map_iq2_sites.py` maps each to its
   enclosing function: 58 sites in 33 functions, grouping as follows. (It
   attributes the enum definition itself to `align_up`, which is a harmless
   artefact of scanning backwards for the nearest definition.)

   **Other architectures, out of scope** (10 functions): everything named
   `glm_*` (`8377`, `8479`, `16712`, `16720`, `16729`, `47669`, `47753`) and
   `qwen4_*` (`59254`, `59260`, `68736`).

   **On the deepseek41 ROCm path, in scope** (21 functions). Note that the
   `metal_graph_*` prefix is misleading and does **not** mean Metal-only: the
   family includes `metal_graph_cuda_tp_*` helpers that read `DS4_CUDA_TP_*`
   environment variables, and `metal_graph_decode_cuda_selected_slots_expected`
   is called from the CUDA/ROCm decode path at `23455`, `23528`, `26818` and
   `26852`. Treat it as the shared graph scheduler. The group covers
   `metal_graph_*` (`20202`, `20265`, `20301`, `22684`, `22759`, `22805`,
   `22853`, `24299`, `29391`, `32951`), the CPU `matvec_*_prequant` dispatchers
   (`10854`, `11729`, `11754`, `11776`, `11831`), `layer_routed_moe_batch`
   (`13626`, `13734`), `ds41_stream_table` (`41134`),
   `ds41_cuda_row_batch_supported` (`43268`) and `ds4_engine_open_internal`
   (`72574`, `72618`).

   **Deferred**: `ds4_session_dspark_seed_batch_enabled` (`61957`, `61967`).

   Most of the in-scope entries are `*_supported` / `*_expected` / `*_eligible`
   predicates that gate a fast path, so returning false for IQ1_M is the
   correct behaviour and they need no change — the generic path takes over.
   The ones that must be extended are the handful that otherwise reach a
   `ds4_die`: the `matvec_*_prequant` dispatchers, `layer_routed_moe_batch`,
   `ds41_stream_table` and `ds4_engine_open_internal`.

3. **Correctness before any tile work** — done for the CPU reference.
   `tests/test_iq1_m_dequant.c` checks the layout against hand-derived values
   and is wired into `test` and `test-rocm`.

   Note that `cuda/mmq/test/test_mmq_parity.cu` and `test_mmq_soa_tiles.cu`
   have **no Makefile targets**, so they are source files rather than a
   harness that can simply be pointed at a new type. Validating a GPU IQ1_M
   kernel means first giving one of them a build rule, with the CPU
   dequantiser as the oracle.

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
