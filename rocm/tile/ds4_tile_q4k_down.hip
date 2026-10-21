// q4_K down projection, summed over the routed experts (sum6 variant), HIP.
//   out[row] = sum over sel of dot(Wd[ids[sel]][row], midq[sel])
// Same weight layout, same q4_K dequant as mul_mv_gu_q4_K_mid_q8_f32 in fused_matvec.hip.
// Routing weights are not applied: the production sum6 kernel does not apply them either.
//
// Mapping follows the engine's moe_down_q4K_sum6_qwarp32_kernel: 8 lanes per row, 32 rows
// per 256-thread block, each lane takes whole q4_K blocks (b = lane, lane + 8, ...).
//
// Weights: [n_exp][out_dim][nb * 144] GGML block_q4_K, nb = mid_dim / 256.
// Activations: nb q8_K blocks per slot, slot s at xq[s * nb].
// Launch: grid (ceil(out_dim / 32), 1), block 256.
#ifndef DS4_TILE_Q4K_DOWN_HIP
#define DS4_TILE_Q4K_DOWN_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define DD_K 256
#define DD_BYTES 144

// Same layout as the engine's cuda_block_q8_K.
typedef struct {
    float d;
    int8_t qs[DD_K];
    int16_t bsums[DD_K / 16];
} ds4_down_q8_K;

// Dot product of one q4_K block with one q8_K block (256 values) in one lane. Integer
// accumulation per 32-value sub-block, with the min term taken from bsums, as in the
// engine's dev_dot_q4_K_q8_K_block.
__device__ __forceinline__ static float ds4_tile_dot_q4k_block(const uint8_t * __restrict__ blk,
                                                               const ds4_down_q8_K * __restrict__ xb) {
    const uint32_t kmask1 = 0x3f3f3f3fu, kmask2 = 0x0f0f0f0fu, kmask3 = 0x03030303u;
    const float d = __half2float(*(const __half *)(blk + 0));
    const float dmin = __half2float(*(const __half *)(blk + 2));
    const uint32_t w0 = *(const uint32_t *)(blk + 4);
    const uint32_t w1 = *(const uint32_t *)(blk + 8);
    const uint32_t w2 = *(const uint32_t *)(blk + 12);
    const uint32_t u0 = w0 & kmask1;
    const uint32_t u1 = (w2 & kmask2) | (((w0 >> 6) & kmask3) << 4);
    const uint32_t u2 = w1 & kmask1;
    const uint32_t u3 = ((w2 >> 4) & kmask2) | (((w1 >> 6) & kmask3) << 4);

    int isum = 0, summs = 0;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int g = s >> 1, hi = s & 1;
        const uint32_t sc_w = (s < 4) ? u0 : u1, mn_w = (s < 4) ? u2 : u3;
        const int sc = (int)((sc_w >> (8 * (s & 3))) & 0xFF);
        const int mn = (int)((mn_w >> (8 * (s & 3))) & 0xFF);
        summs += mn * (int)(xb->bsums[2 * s] + xb->bsums[2 * s + 1]);
        int dot = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t q = *(const uint32_t *)(blk + 16 + 32 * g + 4 * j);
            const int32_t x4 = *(const int32_t *)(xb->qs + 32 * s + 4 * j);
            // Four nibbles (0..15) in one word; one packed int8 dot instruction for the four products.
            const uint32_t nib4 = (hi ? ((q >> 4) & 0x0F0F0F0Fu) : (q & 0x0F0F0F0Fu));
            dot = __builtin_amdgcn_sudot4(false, (unsigned)nib4, true, x4, dot, false);
        }
        isum += sc * dot;
    }
    return xb->d * (d * (float)isum - dmin * (float)summs);
}

__global__ void ds4_tile_down_q4k_sum6(float * __restrict__ out,
                                       const uint8_t * __restrict__ Wd,
                                       const int32_t * __restrict__ ids,
                                       const ds4_down_q8_K * __restrict__ xq,
                                       int nb,
                                       int out_dim,
                                       int n_sel) {
    const int lane = threadIdx.x & 7;
    const int row = blockIdx.x * 32 + (threadIdx.x >> 3);
    if (row >= out_dim) return;

    float total = 0.0f;
    for (int sel = 0; sel < n_sel; ++sel) {
        int expert = ids[sel];
        if (expert < 0) expert = 0;
        const uint8_t * wrow = Wd + ((size_t)expert * out_dim + row) * (size_t)nb * DD_BYTES;
        const ds4_down_q8_K * xs = xq + (size_t)sel * nb;
        float acc = 0.0f;
        for (int b = lane; b < nb; b += 8) {
            acc += ds4_tile_dot_q4k_block(wrow + (size_t)b * DD_BYTES, xs + b);
        }
        acc += __shfl_xor(acc, 4, 8);
        acc += __shfl_xor(acc, 2, 8);
        acc += __shfl_xor(acc, 1, 8);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

#endif

// Engine entry point (DS4_TILE_Q4K_DOWN=1). Returns 0 when it ran, nonzero when the layout is
// not the contiguous [n_expert][out_dim][midq_blocks * 144] that the kernel assumes.
extern "C" int ds4_tile_q4k_down_sum6(float *out, const void *down_w, const void *midq,
                                      const int32_t *selected, uint64_t down_expert_bytes,
                                      uint64_t down_row_bytes, uint32_t midq_blocks,
                                      uint32_t out_dim, uint32_t n_expert) {
    if (midq_blocks == 0u || n_expert == 0u || down_row_bytes != (uint64_t)midq_blocks * DD_BYTES ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        return 1;
    }
    ds4_tile_down_q4k_sum6<<<(out_dim + 31u) / 32u, 256>>>(out, (const uint8_t *)down_w, selected,
                                                           (const ds4_down_q8_K *)midq,
                                                           (int)midq_blocks, (int)out_dim, (int)n_expert);
    return hipGetLastError() == hipSuccess ? 0 : 2;
}
