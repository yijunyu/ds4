// Maps the CUDA names the tile-rs GPU emitter uses onto HIP, so the generated .cu
// compiles with hipcc and runs on gfx1151. Only used for validation on this machine.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define cuda_runtime_h_shim 1
#define __shfl_xor_sync(mask, v, off) __shfl_xor((v), (off))
#define __syncwarp() __builtin_amdgcn_wave_barrier()
#define __align__(n) __attribute__((aligned(n)))
#define cudaStream_t hipStream_t
// CUDA's __dp4a: signed 4-element int8 dot product plus accumulator. Uses the same HIP
// device function the engine's ds4_rocm.h uses, so it lowers to the hardware dot.
static __device__ __forceinline__ int __dp4a(int a, int b, int c) {
    union { int i; char4 v; } av, bv;
    av.i = a;
    bv.i = b;
    return amd_mixed_dot(av.v, bv.v, c, false);
}
