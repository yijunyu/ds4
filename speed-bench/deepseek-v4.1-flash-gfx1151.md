# DeepSeek V4.1 Flash on this box — engine assessment (2026-10-03)

Box: Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB LPDDR5X (~122 GiB usable,
GTT 124 GiB), Kingston OM8TAP42048K1 NVMe (PCIe4 x4, ~3.2 GB/s measured direct read),
Ubuntu, kernel 7.0, system ROCm/HIP 7.1.

V4.1 Flash is 552B backbone. Smallest practical files: ds4 Q2 = 365.7 GB
(152 GiB main weights + 189 GiB Engram tables), llama.cpp Q2_K = 264.5 GB.
Neither fits in RAM, so every engine must stream from SSD.

## Candidates

| Engine | Status on this box |
|---|---|
| kyuz0/ds4 `feat/rocm-deepseek41-halo` @ d32febf (antirez/ds4 PR #1036) | **Measured. Winner.** |
| evolvemarketingitalia PR #4 on kyuz0/ds4 @ 78b3de1 (expert split, DSpark) | Measured. Slower, unstable. |
| llama.cpp PR #28696 (vcruz305 `feat/deepseek-v41-convert` @ 2076412), HIP | Measured. Not usable (<0.4 tok/s prefill). |
| ~~antirez/ds4 main~~ | ~~No ROCm V4.1 graph (Metal/CUDA only). Excluded.~~ **SUPERSEDED 2026-10-25 — see below.** |
| SGLang dev-dsv41 / vLLM | Blackwell-only patch / unmerged PRs, no ROCm. Excluded. |
| Lucebox DS41 profile | Needs R9700 dGPU + Strix Halo. Excluded. |
| Ollama / LM Studio local | Cannot load V4.1. Excluded. |

## Method

Same streaming OpenAI-API client for every engine (`loadgen.py`): prefill = prompt
tokens / time-to-first-token; decode = (completion tokens - 1) / (last - first token).
256 output tokens, temp 0, thinking off, unique nonce per request (no prefix-cache
reuse), one warm-up request excluded. ds4: SSD streaming, 76 GB expert cache, ctx 20480.

- short  = ~940 tokens
- medium = ~6,580 tokens
- long   = 15,626 tokens = the captured first request of a real Claude Code session
  (system prompt + 20 tool schemas + first user turn), replayed verbatim.
  Interactive sessions with MCP servers can reach 20-51K.

## Results — single user (primary KPI), tok/s, prefill / decode, 2 runs each

| Engine | short (940) | medium (6.6K) | long, Claude Code (15.6K) |
|---|---|---|---|
| **kyuz0 ds4** | 26.7-31.9 / 6.9 | 78.3 / 7.0-7.2 | **135.1 / 6.9** |
| PR #4 ds4 (split) | 15.2-18.6 / 7.3 | 57.1-64.5 / 4.7-5.8 | 83.7-106.6 / 3.0-7.2 |
| llama.cpp HIP | <0.4 / n.a. (aborted after 40 min) | not run | not run |

Time to first token on the Claude Code prompt with kyuz0 ds4: ~116 s.

## Results — 4 parallel users (auxiliary), aggregate tok/s

| Engine | short | medium | long |
|---|---|---|---|
| kyuz0 ds4 | 8.4 / 2.05 (per user 6.5 / 1.1) | ~8.3 prefill agg; aborted after 17 min | not run (projected hours) |
| PR #4 ds4 | 5.8 / 1.5 | skipped | skipped |

4 users is worse than 1 user in aggregate: on ROCm + SSD streaming, ds4 serves batched
sessions in order (no native batching) and four contexts thrash the expert cache;
after the first 2048-token chunk, concurrent prefill drops to ~6.5 tok/s per session.

## Verdict

kyuz0's ds4 ROCm branch is the only engine that runs V4.1 Flash usefully here:
~7 tok/s decode flat across context, prefill 27 -> 135 tok/s rising with prompt size.
Single-user only; do not serve 4 concurrent users.

Our decode is ~30% below the branch's published 10.2-10.5 tok/s. Likely causes:
HIP 7.1 here vs ROCm 10.0 / HIP 7.15 there (their tuned Engram matmul path needs
hipBLASLt 100401), no TuneD accelerator-performance profile, different SSD.
Installing ROCm 10.0 is the first thing to try for more speed.

## Mainline ds4 gained a ROCm V4.1 path (2026-10-25)

The "no ROCm V4.1 graph" exclusion above was true on 2026-10-03 and is no longer.
Merging antirez/ds4 main into `adopt-pr-1221` (merge commit `7a47fc5`, 23 upstream
commits) brought it in. The backend gate in `ds4.c` changed:

    before (0f179aa):  "V4.1 requires Metal or single-GPU CUDA per rank
                        (optional network tensor parallelism)"
    after  (fafca4b):  "V4.1 requires Metal, single-GPU CUDA or ROCm
                        (network TP on Metal/CUDA)"

`4c64ae4 Enable V4.1 DSpark on Metal, CUDA and ROCm` is the enabling commit, and
`ds4.c` now also carries a ROCm-specific drafter path ("ROCm failed to make the
V4.1 drafter resident"). `docs/STRIX_HALO.md` documents the recipe directly:

    ./ds4 --rocm -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming --ctx 4096

with upstream's own caveats: V4.1 vision and network TP unsupported on ROCm, and
"large prefills are not yet tuned on this backend and can be slow". Nothing about
the measured numbers above is invalidated — a third-party fork is simply no longer
required, so this is now maintainable from the fork we track.

Not re-measured. Whether mainline matches kyuz0's 135 tok/s prefill / 7 tok/s
decode is unknown; the Engram and expert-read paths were touched upstream since.

### What still blocks standing it up as a service

1. **Memory is either/or with the live MiniCPM service, not a swap.** The known-good
   config is a 76 GB expert cache; the sweep above shows the real ceiling:

        requested 84GB -> 95.67 GiB planned (KV 0.13 + buffers 2.18 + resident model
                          9.37 + expert cache 76.87 + prefill reserve 7.12)  KILLED
        requested 88GB -> 99.67 GiB planned                                   KILLED

   Both died to the watchdog at MemAvailable < 3 GiB on an *otherwise idle* box with
   ~122 GiB usable, so planned-vs-actual already runs hot. MiniCPM-V 4.7 currently
   holds ~35 GB resident, leaving ~86 GiB available — below the ~88 GiB the
   known-good 76 GB config plans for. One or the other, not both.

2. **Port 8000 collides.** `ds4-minicpm-vulkan.service` (llama-server, `--port 8000`)
   is the live LAN service; `ds4-server.service` is also configured for 8000. Any
   V4.1 unit needs a different port or MiniCPM stopped first.

3. **Checkpoint revision is unverified.** The local file is
   `/data/gguf/DeepSeek-V4.1-Flash-Q2.gguf` (365,713,686,528 B, valid GGUF magic,
   dated 2026-10-03) and its size matches the ds4 Q2 figure above. Upstream's
   `download_model.sh` now defaults `ds41f-q2` to
   `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` (commit `5379c1c`), a later
   revision. Unconfirmed whether the merged ROCm path expects the 1017 layout.

Correction to an earlier note: the SSD-streaming cache is a **RAM** cache of routed
experts, not a disk cache (`docs/SSD_STREAMING.md`). The 87 GiB freed on `/data` by
deleting the two obsolete V4 GGUFs is not what streaming needs — the V4.1 GGUF was
already resident on that disk. Freeing it helps headroom and KV checkpoints, nothing
about the expert cache.

`ds4-server.service` is currently **disabled**, with the V4-era unit preserved at
`~/.config/systemd/user/ds4-server.service.before_v4_obsolete`. A ready-to-install
V4.1 variant that addresses (1) and (2) is at
`/data/v41bench/ds4-v41.service.ready` — not installed.

## Raw data
progress.txt (all rows), ds4_kyuz0/ ds4_pr4_split/ llamacpp_hip/ (server logs),
loadgen.py, run_ds4.sh, run_llama.sh, cc_req_0.json (captured Claude Code request).
cache_84GB/ cache_88GB/ cache_92GB/ memline.txt (expert-cache sizing, 2026-10-03).
