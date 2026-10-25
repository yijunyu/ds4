# AMD (gfx1151) prefill and decode tuning — measured 2026-10-17

## Correction to an earlier claim

Earlier notes reported AMD prefill at "72–151 tok/s" and called it 3–6× behind the
Macs. **That was not a prefill measurement.** It came from `prompt_per_second` on
~20-token prompts during 256-token generations, where fixed per-request overhead
dominates. Measured properly with a cold 5,500-token prompt, AMD does **1,055 tok/s**
and is ~28% behind the M1 Ultra, not several times behind.

## Method

Cold prefill: a 5,526-token prompt, unique per run (a random marker in the first
tokens) so the prefix cache cannot help; 8 output tokens so decode barely contributes;
three runs, median reported. Decode: three interleaved 256-token generations on the
same Q4_K_M file Studio uses. One server at a time — each holds the full 21 GB model,
and four concurrent instances exhausted 113 of 122 GB and failed to load.

## Prefill results (cold, 5,526 tokens)

| Config | Prefill | vs baseline |
| --- | ---: | ---: |
| baseline (no `-fa`, default batches) | 1,055 | — |
| `-fa on` | 1,066 | +1% |
| `-fa on -b 4096 -ub 1024` | 1,174 | +11% |
| **`-fa on -b 8192 -ub 2048`** | **1,255** | **+19%** |
| `-fa on -b 16384 -ub 4096` | 1,201 | +14% (regresses) |
| `-b 8192 -ub 2048 -t 24` | 1,221 | +16% (threads do not help) |
| Vulkan, `-fa on -b 8192 -ub 2048` | 1,127 | +7% |
| *Studio M1 Ultra, reference* | *1,468* | |

Bigger prefill batches are the lever, up to a point: 8192/2048 is the optimum and
16384/4096 regresses. `-fa` alone is nearly free but worth little. Thread count is
irrelevant, which is consistent with prefill being GPU-bound.

## Decode results — Vulkan wins here

| Backend | Decode (256 tok) |
| --- | ---: |
| ROCm/HIP, `-fa -b 8192 -ub 2048` | 58.0 – 58.4 |
| **Vulkan, `-fa -b 8192 -ub 2048`** | **68.2 – 68.9 (median 68.6)** |
| PolyStrata HIP (production today) | 61.4 |

Vulkan is **+18% on decode** over ROCm llama.cpp and **+12%** over PolyStrata, but
**−10% on prefill**. The two backends trade places, which matches the Strix Halo
reports that backend choice here is model- and phase-specific rather than universal.

## What this means

* Adopting `-fa on -b 8192 -ub 2048` on ROCm buys **+19% prefill** at no decode cost.
* Switching to Vulkan buys **+18% decode** but costs ~10% prefill. For interactive
  chat, decode dominates; for long-document work, prefill does.
* Neither closes the gap to the M1 Ultra, which is bandwidth (800 vs ~215 GB/s
  sustained), not tunable.

## Reproducing

```bash
# Vulkan needs the LunarG SDK; it extracts without root.
export VULKAN_SDK=/tmp/vksdk/x86_64
export LD_LIBRARY_PATH=/tmp/vksdk/x86_64/lib:/tmp/vksdk/x86_64/lib/VulkanLoader/lib
cmake -B build-vulkan -DGGML_VULKAN=ON -DGGML_HIP=OFF -DLLAMA_CURL=OFF \
  -DVulkan_INCLUDE_DIR=$VULKAN_SDK/include \
  -DVulkan_LIBRARY=$VULKAN_SDK/lib/VulkanLoader/lib/libvulkan.so
cmake --build build-vulkan --target llama-server -j16
./build-vulkan/bin/llama-server --list-devices   # Radeon 8060S (RADV STRIX_HALO)
```

The SDK lives in `/tmp` and will vanish on reboot; keep it in a durable path if this
backend is adopted.

## Adopted: llama.cpp + Vulkan is now the AMD backend

Switched in production on 2026-10-17, replacing PolyStrata:

| | PolyStrata HIP | llama.cpp Vulkan | change |
| --- | ---: | ---: | ---: |
| Decode (256 tok) | 61.4 | **68.2 – 68.8** | **+12%** |
| Prefill (cold 5.5K) | ~1,055 | 1,127 | +7% |
| Model id | mixed-Q3 | Q4_K_M (same file as the Macs) | — |

Vulkan also beats ROCm llama.cpp on decode (68.6 vs 58.0, +18%), which is why the
backend changed rather than just the flags. The prefill cost versus ROCm's tuned
1255 tok/s is accepted: this service is interactive.

### Where things live

- `~/.config/systemd/user/ds4-minicpm-vulkan.service` — the backend, enabled at boot,
  bound to the LAN with the shared key.
- `~/.config/systemd/user/ds4-minicpm-lan.service` — the PolyStrata unit, **kept on
  disk and disabled** for rollback (`systemctl --user start ds4-minicpm-lan` after
  stopping the Vulkan one).
- `/home/yijun/llama-vulkan/bin/` — the Vulkan build, copied out of the source tree
  so a rebuild cannot remove it.
- `/home/yijun/vulkan-sdk/x86_64/` — the LunarG SDK (1.9 GB), moved out of `/tmp` so
  it survives a reboot. Extracted without root; the system has the RADV ICD
  (`radeon_icd.json`) but not the dev headers.
- `/home/yijun/models/minicpm/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf` — byte-identical to
  what the Macs serve (SHA-256 bf3d1c67…df6).

### Rollback

```bash
systemctl --user stop ds4-minicpm-vulkan
systemctl --user start ds4-minicpm-lan
```

Only one 21 GB model may be resident; the units are mutually exclusive by design.

## tile-rs q4_K kernels: investigated, not adopted (2026-10-17)

Question raised: could tile-rs kernels decode faster than llama.cpp on gfx1151?
Answer: **cannot be measured on this codebase**, for structural reasons found by
reading the source, not by giving up.

### What does work

All six q4_K MoE kernels in `tile-rs-metal` (`tile_std/src/tile.rs`) emit and compile:

| Kernel | Lines | `xcrun metal` |
| --- | ---: | --- |
| `mul_mv_id_q4_K_f32` (decode) | 147 | ✅ |
| `mul_mv_id_q4_K_pair_f32` | 175 | ✅ |
| `mul_mv_id_q4_K_pair_swiglu_f32` | 195 | ✅ |
| `mul_mv_id_q4_K_sum6_f32` | 128 | ✅ |
| `mul_mm_id_q4_K_f32` (prefill) | 204 | ✅ |
| `mul_mm_id_q4_K_f16` | 204 | ✅ |

Emitted with `print_msl <kernel>` (in `glm52-main-wt`), assembled into one library
(1,043 lines → 29,808-byte AIR), loaded at runtime, entry point resolves, GPU executes
it deterministically. The dequantisation is genuine q4_K: `sc16[4]`, `KMASK1/2/3`
scale unpacking, `yl[16]`/`yh[16]` nibble lanes, 144-byte blocks.

**Correction to an earlier note in this repo:** `ds4_msl/queue.csv` records
`dense_mul_mv_ext_q4, failed:extract — "emitter has no q4 dequantize_q4_t4 dispatch"`.
That note is **stale**. `mlir_to_msl.rs` lowers all six q4_K ops
(`KernelType::MulMvIdQ4KF32 => emit_mul_mv_id_q4_K_f32_msl(out)`, line 2348).

### Why no throughput number exists

Three routes were tried and each is closed by the code's own structure:

1. **Standalone harness** (four variants, ending with the authoritative
   `ds4_gpu_make_mul_mv_id_args` field layout from `ds4_metal.m`): the GPU output is
   stable across runs while references shift, so the simdgroup reduction
   (`NSG=2`, `NR0=2`, `ids`-indexed expert selection) was never reproduced. No number
   was published rather than publish an unverified one.
2. **`DS4_METAL_MOE_SOURCE` swap**: the engine requests **39** named MoE kernels from
   `moe.metal`; tile-rs emits **6**, and the emitted names carry a `ds4_` prefix the
   engine does not ask for. No partial substitution exists.
3. **In-engine test**: `ds4_gpu_matmul_q4_K_pair_decode_tensor` is declared in
   `ds4_gpu.h` and called in `ds4.c`, but has **no implementation**, and the call site
   is `#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) …` — it is CUDA-only. On
   Apple/Metal the only implemented q4_K GPU functions are
   `ds4_gpu_attention_output_q4_K_batch_tensor` and
   `ds4_gpu_attention_output_low_q4_K_slice_tensor`. The q4_K MoE matvec is reachable
   only through the internal `g_moe_mul_mv_id_q4_k_pipeline`, with no public entry
   point.

Obtaining the number would require **adding an engine API, wiring it into
`ds4_metal.m`, and writing a `test_fill_q4_K_weights` helper** — engine work, not
measurement.

### Why it is not worth that work here

- AMD decode runs at **68.6 tok/s ≈ 84% of the 256 GB/s bus**, so a perfect kernel
  caps out ~16% higher.
- The only comparable measured kernel, `dense_mul_mv_f32_4`, is **1.036× — a tie**
  (`ds4_msl/results.csv`).
- tile-rs's real wins are elementwise: `glu` 0.276× (3.6× faster), `softmax` 0.466×,
  median **0.802×** across six kernels — none of which is the decode bottleneck.
- Vulkan already delivered **+12%** in production.

If tile-rs is pursued, it belongs on **DS4/GLM**, its design target, where all 39 MoE
kernels exist and `moe.metal` binds them correctly.

## Service settings and llama.cpp update — measured 2026-10-23/24

### Corrections to the section above

* **`-ub 2048` is superseded by `-ub 1024`.** Measured again on the Vulkan service
  (`pp2048`, 10 repeats, graphics queue on): `-ub 1024` is **1,179 ± 17** against
  **1,152 ± 17** for `-ub 2048` and **1,046 ± 20** at the default `-ub 512`.
* **The service no longer runs from `/home/yijun/llama-vulkan/bin/`.** It runs from
  `/home/yijun/llama-tile/bin/`. The old path still exists on disk but is unused.
* **Decode is no longer 68.6 tok/s.** It is ~73 on the live service, from two changes:
  the graphics-queue setting and the llama.cpp update. (`-ub 1024` affects prefill
  only, and n-gram only copy-heavy prompts.) The "84% of the bus, so a perfect kernel
  caps out ~16% higher" reasoning still holds in shape, but the baseline moved.

### Adopted

| Setting | Effect | Output impact |
| --- | --- | --- |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | decode **73.0 ± 0.6** vs 70.2–70.4 stock (`llama-bench`, 10 reps) | none — greedy text identical, KL 0 |
| `-ub 1024` | prefill +2% over `-ub 2048`, +13% over the default | changes prefill numerics: KL **0.0040 ± 0.0001**, top-1 agreement 96.6%, PPL ratio 1.0020 ± 0.0014 |
| `--spec-type ngram-simple --spec-draft-n-max 4` | same copy-heavy prompt, same 74 tokens: **90.4** with vs **69.5** without, **+30%**; free-form unchanged | none — greedy text byte-identical to plain decoding on that prompt |
| tile-rs q4_K/q6_K shaders (kept) | neutral: 69.9 vs 69.8 mean, A/B on the same binary | correctness 108/108 + 3/3 vs CPU reference |

The `-ub 1024` numerics change was accepted deliberately: the perplexity difference is
0.2% against a 0.14% uncertainty, so it is not a measurable quality loss, but ~3.4% of
positions pick a different top token, so greedy answers can differ from the `-ub 512`
path. The tile shaders are kept by choice, not for speed — they measure neutral.

### Vision enabled (2026-10-24)

The service had been serving a **vision** model text-only: there was no `--mmproj`, so
any request carrying an image failed. Fixed by adding

```
--mmproj /home/yijun/models/minicpm/mmproj-model-f16-4x.gguf
```

This is not a tuning knob and it is not free of consequence, so it is recorded here
rather than in the table above. Notes:

* **The `-4x` variant, matching what the Studio Mac already serves**, so the two
  deployments do not diverge on image handling. The repo also offers
  `mmproj-model-f16.gguf`; the difference between them was not measured.
* **868 MB download**, and resident memory did not move measurably: 32 GB of 122 GB
  used, the same as before, on top of the 21 GB language model.
* **Decode is unaffected**: 72.2 tok/s on the text path after the change.
* **Verified with two differently coloured images**, red and yellow, each read back
  correctly. One image would not have distinguished seeing the pixels from guessing a
  plausible colour. The projector was tested on a spare port first, so an
  incompatibility with the freshly updated llama.cpp build could not have taken the
  service down.

### llama.cpp updated (2026-10-24)

Updated from 1,172 commits behind upstream to `30322097`. The experimental Vulkan
tile hook rebased onto it **with no conflicts** and built with **zero errors**; it is
one commit, `2881e5b8`.

Verified before keeping it:

* **Quality unchanged** — perplexity **9.5414 ± 0.316** against **9.5274 ± 0.315**
  before, on the same corpus and settings.
* **Tile shaders still correct** — `test-backend-ops MUL_MAT_ID` passes 108/108
  (q4_K) and 3/3 (q6_K) with 20 confirmed shader dispatches. Upstream changed the
  declaration form of `vk_mat_vec_id_push_constants` but not its layout, so the
  pre-compiled `.spv` files remain compatible. **Re-run this check after any future
  upstream update** — a layout change would corrupt results silently.

Measured on the live service, before and after:

| Prompt type | Before | After |
| --- | ---: | ---: |
| Free-form | 69.5, 68.4 | **73.6, 72.8** |
| Copy-heavy (n-gram) | 89.5, 92.6 | **103.1, 105.6** |

The copy-heavy runs are only 74 tokens each, short enough to be noisy; the gap is
large and held across runs.

**Caveat:** greedy output text changed. On "The capital of France is" both builds
answer Paris, but the intermediate reasoning text and final phrasing differ. The
perplexity result says this is not a quality regression — most likely upstream
template handling plus near-tie token flips.

### MTP is not available for this model

`--spec-type draft-mtp` **fails and the server exits** with "failed to create MTP
context". The Q4_K_M checkpoint has 733 tensors and none are MTP, `nextn` or EAGLE,
and `openbmb/MiniCPM-V-4.7-35B-A3B-gguf` ships only `dspark-*` drafts. MTP needs a
head either embedded in the checkpoint or as a separate file; this model has neither.
`ngram-simple` is the speculative type that works here.

### Tested and rejected — do not re-investigate without new information

* **Draft-model speculative decoding.** Both drafts lose badly against the ~70 tok/s
  baseline. The 1B draft (`MiniCPM-V-4.7-1B` Q4_K_M) gets 19% acceptance → **21 tok/s**
  at `n_max 8`, ~35 at `n_max 3`. The *official* DSpark draft, trained for this exact
  target, gets 9% (reasoning off) to 15% (reasoning on), and 9.6–21.8% per prompt →
  **18–34 tok/s**. Two causes, both measured: the target's cost per verification batch
  is 13.8 ms (1 token), 18.1 (2), 22.1 (4), 31.2 (8) but **67.6 ms at 9** — and
  `n_max 8` produces a 9-token batch, landing on that cliff; and a speculative step
  must yield ~2.5 accepted tokens to break even, which these acceptance rates cannot.
* **BF16 target, to test whether quantization caused the low acceptance.** Converted
  the full BF16 weights on the Studio Mac and re-ran: f16 target acceptance
  26.4/11.2/21.7/10.8% against 21.0/8.1/10.4/11.0% for Q4_K_M-from-f16, with no-draft
  baselines of 69.3 and 103.1 tok/s. Quantization explains only a few points;
  speculation still loses. **This is settled — the 70 GB download is not worth
  repeating.**
* **Lower-bit requantization** (experts → Q3_K, optionally LM head → Q4_K): +5% to
  +10% decode, but KL **0.031–0.037** and top-1 agreement 89.6–91.2%, roughly 8–9×
  the divergence of the accepted `-ub 1024` change. Rejected on quality.
* **Link-time optimization**: 72.4–72.7 against 72.7–72.8. No gain.
* ~~**Prompt-prefix caching**: no reuse on this model. `cache_n=0` even with a matching
  prefix (`f_sim_best = 0.972`) and with `-ctxcp 8 -cms 256`. The hybrid
  `GATED_DELTA_NET` layers carry recurrent state that cannot be truncated to a shared
  prefix.~~ **SUPERSEDED 2026-10-25 — prefix caching now works, and it is the largest
  remaining win. See "Prompt-prefix caching works after the update" below.** The old
  result was real but was measured on the pre-update build; the 1,172-commit llama.cpp
  update evidently taught the hybrid recurrent layers to carry a reusable prefix.
* **CPU/GPU overlap**: the gap between GPU completion and the next submit is ~120 µs
  per wait, ~3 waits per token ≈ 0.4 ms of a 14 ms token (~3%). The spin-wait in
  `ggml_vk_wait_for_fence` dominates CPU samples but is idle time, not recoverable work.
* **Gate/up matvec fusion**: not implemented. Estimated under 1% — the expert matvec
  already runs at ~174 GB/s and the LM head at ~219 GB/s, against a measured
  read-only ceiling of 212 GB/s on this GPU.
* **Concurrency** is the one real throughput lever, and it is **already on — no flag
  needed** (re-measured 2026-10-25). The unit carries no `-np`/`--parallel`, yet the
  server starts `n_slots = 4, n_ctx_slot = 131072, kv_unified = 'true'`: four slots, each
  able to use the *full* 128K context, because the unified KV cache is shared rather than
  divided per slot. Nothing was changed to get this. Measured decode on the live service,
  distinct prompts per slot so the n-gram draft cannot copy across them:

  | concurrent | per-slot decode | total decode | solo speed kept |
  |---|---|---|---|
  | 1 | 73.0 tok/s | 73.0 | 100% |
  | 2 | 51.8 / 50.1 | **101.9** (1.40×) | 70% |
  | 4 | 34.5 / 35.2 / 34.4 / 35.1 | **139.2** (1.91×) | 48% |

  This confirms the earlier `-np 2` figure (51.7 + 52.1 ≈ 104 against 51.8 + 50.1 ≈ 102)
  and adds the 4-slot point. **Measure it as the sum of per-slot decode rates, not as
  total tokens over wall time**: generations stop at different lengths, so the
  wall-clock aggregate swings with where EOS lands (the same n=1 case read 65.5 and then
  52.9 tok/s across two runs while per-slot held steady at 72.8 and 73.0). Useful for a
  multi-user service; for one user it only costs per-request latency.
* **Matrix cores for prefill — already on, nothing to enable** (2026-10-25). Prefill
  measured on the live service: **1291.7 tok/s** for an 8605-token prompt (6661.9 ms),
  decode 66.9 tok/s in the same request. At ~3B active parameters that is only ~7.8
  effective TFLOPS, which looks like headroom until you check the path: RADV exposes
  `VK_KHR_cooperative_matrix` (rev 2) and `VK_KHR_shader_integer_dot_product`, and
  `llama-bench --list-devices` reports the backend already selecting them —
  `warp size: 64 | int dot: 1 | matrix cores: KHR_coopmat`. So the low TFLOPS figure is
  what MoE prefill *achieves with* matrix cores, not evidence they are idle; routing
  fragments the batch so the GEMMs never reach dense-peak shape. Check it with that one
  command before assuming otherwise.
* **Reviving the tile-rs q4_K HIP emitter** (2026-10-25). Parked, and it should stay
  parked. Two independent reasons. (1) *Wrong stack:* tile-rs has **no** HIP backend —
  `BackendKind` is Ascend/Cuda/Musa/Aie/Nki/Spirv/Msl/Bang/Gaudi. Its ROCm support is
  one seam inside the CUDA emitter (`TILERS_GPU_MMA=rocwmma` in
  `crates/rustc_codegen_tile/src/mlir_to_gpu.rs`, selecting `rocwmma` over
  `nvcuda::wmma`, with a 64-bit lane mask because `hipcc` rejects CUDA's 32-bit one).
  That seam feeds the **HIP/ds4 engine**, which the production service does not use — it
  runs Vulkan, where the equivalent is the `KHR_coopmat` path above. (2) *No headroom:*
  the hand-written q4_K down kernel already reaches 215 GB/s against production's 217
  (0.99×) with error 2.1e-7, and the read-only probe ceiling is 212 GB/s, so an emitted
  kernel is competing for a couple of percent of a bandwidth limit. `rocwmma` *is*
  installed here (`/opt/rocm/include/rocwmma/rocwmma.hpp`, hipcc 7.1 / clang 21), so the
  seam's precondition holds — it is the payoff that is missing, not the toolchain. Note
  also that grepping tile-rs for `hip` is misleading: 55 files match, but only 5 at a
  word boundary and the rest are `chip`.
  *The tile-rs source is on the Studio Mac at `/Users/yijun/tile-rs` (branch `main`),
  not on this box — that absence was the original reason the thread stalled.*

### Prompt-prefix caching works after the update (2026-10-25)

This reverses the rejected entry above, and it is the biggest prefill win on this box.
Measured against the live service with `cache_prompt: true`.

Identical prompt sent three times, 4,102 tokens:

| run | prompt tokens processed | prefill |
|---|---:|---:|
| 1 (cold) | 4,102 | 3,070.6 ms |
| 2 | **4** | **51.5 ms** |
| 3 | **4** | **51.4 ms** |

A growing conversation — the pattern that actually matters, since every chat turn
re-sends the whole history:

| turn | new tokens processed | total prompt | prefill |
|---|---:|---:|---:|
| 1 (cold) | 3,733 | 3,733 | 2,759.1 ms |
| 2 (prefix + new turn) | **19** | 3,748 | **132.6 ms** |
| 3 (prefix + new turn) | **18** | 3,762 | **131.0 ms** |

So a follow-up turn costs ~130 ms of prefill instead of ~2,800 — a **~21× cut in
time-to-first-token**, and ~60× on an exactly repeated prompt. Nothing was configured
to get this; `cache_prompt` defaults on for the chat endpoints. It needs no flag in the
unit.

**The cache is per-slot, not shared — this is the part to understand with 4 slots.**
Four *identical* prompts fired at once: one slot reported 4 tokens processed, the other
three reprocessed all 4,102. Each of the 4 slots keeps its own prefix cache, and
llama.cpp routes a request to the slot with the longest matching prefix. Consequences:

- Up to **4 concurrent sessions each reuse their own history independently** — the
  normal multi-user case works.
- A **5th** active session evicts one of the four, and that session pays a cold prefill
  on its next turn.
- Several slots needing the *same* long prefix simultaneously cannot share it; each pays
  for its own copy.

Caveat when benchmarking: pass `cache_prompt: false` for any prefill or decode number
meant to be comparable to the figures elsewhere in this document, or the second run will
look ~60× faster than the hardware can actually do cold.

### Where things live (updated)

- `~/.config/systemd/user/ds4-minicpm-vulkan.service` — the live unit, with the four
  adopted settings plus `--mmproj`. Enabled at boot.
- `/home/yijun/llama-tile/bin/` — the installed build the service runs.
  **Rollback:** `/home/yijun/llama-tile/bin.before_update_20261024`.
- `/home/yijun/llama-tile/shaders/{moe_q4k,moe_q6k}.spv` — the tile shaders.
- `/home/yijun/models/minicpm/` — the Q4_K_M language model (21 GB) and
  `mmproj-model-f16-4x.gguf` (868 MB), the vision projector.
- `~/llama.cpp-tile`, branch `tile-q4k-hook` — the source. One commit
  (`2881e5b8`) on upstream `30322097`, and pushed to the fork at that commit.
  Pre-update state is on branch `pre-update-backup` (`1396c023`), which is also the
  commit the fork carried before the force-push.
- `~/ds4-rs-amd/tile-hip-q4k/llama_hook/` — shader sources, benchmark logs, and
  dated backups of the unit file (`*.before_gfxq`, `*.before_ub1024`,
  `*.before_ngram`, `*.before_vision`).

### The Studio Mac runs the same model as a service too

Not systemd: a launchd agent, `~/Library/LaunchAgents/com.yijun.minicpm-studio.plist`,
with `RunAtLoad` and `KeepAlive` both true, serving the same Q4_K_M file from the
Hugging Face cache snapshot on port **8011** with the `-4x` projector. Worth knowing
before concluding that a MiniCPM endpoint on the LAN is this machine's.
