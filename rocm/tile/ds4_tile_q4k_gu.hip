// tile-rs generated q4_K gate/up MoE kernel, wrapped for the DS4 ROCm engine.
//
// The generated kernel (gumoe_2048x6144.cu) takes an fp32 activation and re-quantises it.
// The engine already holds the activation as q8_K, so this wrapper writes the q8_K values
// straight into the kernel's scratch layout and calls the matvec directly. That keeps the
// arithmetic identical to the engine's own kernel.
//
// Only one shape is compiled in. ds4_tile_q4k_gu_mid returns nonzero for any other shape,
// and the caller then runs the engine's kernel.

#include <stdint.h>
#include <hip/hip_runtime.h>
#include "gumoe_2048x6144.cu"

#define DS4_TILE_M 2048      // expert_mid_dim (rows per expert)
#define DS4_TILE_K 6144      // hidden size (inputs per row)
#define DS4_TILE_SLOTS 8     // selected experts per token

// Same layout as the engine's cuda_block_q8_K.
struct ds4_tile_q8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

// Scratch layout of the generated kernel: int8 quants, then int32 group sums, then floats.
#define DS4_TILE_OFF_XBS DS4_TILE_K
#define DS4_TILE_OFF_XD (DS4_TILE_OFF_XBS + 4 * (DS4_TILE_K / 16))

__global__ void ds4_tile_fill_scratch(const ds4_tile_q8_K * __restrict__ xq, char * __restrict__ scratch) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DS4_TILE_K) {
        ((int8_t *)scratch)[i] = xq[i / 256].qs[i % 256];
    }
    if (i < DS4_TILE_K / 16) {
        ((int32_t *)(scratch + DS4_TILE_OFF_XBS))[i] = xq[i / 16].bsums[i % 16];
    }
    if (i < DS4_TILE_K / 256) {
        ((float *)(scratch + DS4_TILE_OFF_XD))[i] = xq[i].d;
    }
}

__global__ void ds4_tile_scale_mid(float * __restrict__ mid, const float * __restrict__ weights) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < DS4_TILE_SLOTS * DS4_TILE_M) {
        mid[idx] *= weights[idx / DS4_TILE_M];
    }
}

static void * ds4_tile_scratch = NULL;

extern "C" int ds4_tile_q4k_gu_mid(float *mid,
                                   const void *gate_w,
                                   const void *up_w,
                                   const void *xq,
                                   const int32_t *selected,
                                   const float *weights,
                                   uint64_t gate_expert_bytes,
                                   uint64_t gate_row_bytes,
                                   uint32_t xq_blocks,
                                   uint32_t expert_mid_dim,
                                   uint32_t n_expert) {
    if (expert_mid_dim != DS4_TILE_M || xq_blocks * 256u != DS4_TILE_K || n_expert != DS4_TILE_SLOTS) {
        return 1;
    }
    if (gate_row_bytes != (uint64_t)xq_blocks * 144u || gate_expert_bytes != (uint64_t)DS4_TILE_M * gate_row_bytes) {
        return 1;
    }
    if (ds4_tile_scratch == NULL) {
        if (hipMalloc(&ds4_tile_scratch, gumoe_scratch_bytes()) != hipSuccess) {
            return 1;
        }
    }
    // The generated kernel reads the expert slabs as one [n_expert][M][K/256 * 144] array,
    // so gate_w and up_w must be the expert base pointers, which the engine provides.
    ds4_tile_fill_scratch<<<(DS4_TILE_K + 255) / 256, 256>>>((const ds4_tile_q8_K *)xq, (char *)ds4_tile_scratch);
    gumoe<<<dim3(DS4_TILE_M / 8, DS4_TILE_SLOTS), 256>>>(gate_w, up_w, selected, NULL, mid, ds4_tile_scratch);
    ds4_tile_scale_mid<<<(DS4_TILE_SLOTS * DS4_TILE_M + 255) / 256, 256>>>(mid, weights);
    return hipGetLastError() == hipSuccess ? 0 : 2;
}
