// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    //const int TH = 128;
    //const int TW = 128;


    for (int oy = 0; oy < H; ++oy) {
        // Step by 16 instead of 8
        for (int ox = 0; ox < W; ox += 16) {
            // Two accumulators(independent)
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 w = _mm256_broadcast_ss(ker + ky * K + kx);
                    
                    // Load two chunks of 8 input floats
                    __m256 v0 = _mm256_loadu_ps(in + (oy + ky) * in_stride + ox + kx + 0);
                    __m256 v1 = _mm256_loadu_ps(in + (oy + ky) * in_stride + ox + kx + 8);
                    
                    // Fused multiply-add into independent accumulators
                    acc0 = _mm256_fmadd_ps(w, v0, acc0);
                    acc1 = _mm256_fmadd_ps(w, v1, acc1);
                }
            }

            // Store both chunks back to memory
            _mm256_storeu_ps(out + oy * W + ox + 0, acc0);
            _mm256_storeu_ps(out + oy * W + ox + 8, acc1);
        }
    }

    //conv_naive(in, out, ker, H, W, K);
}
