# MiniCPM-V 4.7 35B-A3B on gfx1151

## Scope

- Hardware: Framework Desktop, AMD Strix Halo `gfx1151`, 128 GB unified memory
- Runtime: ROCm 7.1.52801, HIP `7.1.52801`
- Model: `openbmb/MiniCPM-V-4.7-35B-A3B-gguf` (mixed-Q3, 15.5 GB)
- Engine: PolyStrata `strata-poly` (HIP build, `gfx1151` architecture)
- Context: **131,072 tokens** (128K allocation)
- Configuration: `--kv-whole --no-draft --window 1 --lookup 0 --tail-skip 0`

## Performance Results

### Decode Throughput

| Workload | Tokens | Throughput |
|---|---:|---:|
| 256-token decode (128K config) | 256 | **60.3–61.0 tok/s** (median 60.6) |
| 4,236-token prompt (fresh) | 4,236 | **1,299.6 prompt tok/s**, 3.41 s total |

### Long-Context Recall

| Workload | Prompt Tokens | Wall Time | Recall |
|---|---:|---:|---|
| 70,074-token prompt (routed) | 70,074 | **141.6 s** | All 3 codes (OPAL-6302, SILVER-9284, AMBER-5176) |
| Cached follow-up (70,097 reused) | 21 | **0.757 s** | BETA code (SILVER-9284) |

### Streaming & Latency

| Metric | Result |
|---|---|
| First content (256 tokens) | **0.202 s** |
| Total 256-token response | **4.381 s** |
| Routed 24-token reply | **29.1 tok/s** |
| Cached 9-token reply | **31.5 tok/s** |

## Service Compatibility

- **82 no-model tests pass**: 22 controller/router, 12 telemetry, 48 native reasoning/monitor
- **60 live requests pass**: 39 repair regressions, 15 primary API/performance, 2 long-context, 4 hot-switching
- **Thinking-mode fix**: `reasoning_close_retry` enabled; high-effort requests now return final content
- **Pyright diagnostics**: 0 errors across 9 repaired files
- **Active LSP probes**: 9/9 clean

## Coding Evaluation

| Category | Result |
|---|---|
| Independent | **2/4 tasks solved** (dependency resolver, invoice CLI) |
| Assisted recovery | **4/4 tasks, 47/47 test methods** (assertion-driven repair) |
| Tool protocol | **35 valid tool calls, zero errors** in independent run |

## Build & Configuration

- **PolyStrata build**: `STRATA_ENABLE_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx1151`, `STRATA_PREFILL_MMQ=ON`
- **Compiler**: `/usr/lib/llvm-21/bin/clang++`, ROCm root `/usr`
- **GGUF**: llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, archived in `deps/llama-pinned`
- **Config**: `minicpm/strata-minicpm-service.json` (131072 context, 16 threads, 16 GiB reserve)

## Limitations

- Full 128K prompt validation not completed (70K validated)
- No multi-language or broad coding benchmark
- Not a SWE-bench or representative success rate
- No DeepSeek comparison established
- Thinking-enabled coding not qualified
