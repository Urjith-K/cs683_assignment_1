// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 8) {
            __m256 acc = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {

                for (int kx = 0; kx < K; ++kx) {
                    __m256 w = _mm256_broadcast_ss(ker + ky * K + kx);
                    __m256 v = _mm256_loadu_ps(in + (oy + ky) * in_stride + ox + kx);
                    acc = _mm256_fmadd_ps(w, v, acc);
                }
            }

            _mm256_storeu_ps(out + oy * W + ox, acc);
        }
    }
    conv_naive(in, out, ker, H, W, K);
}
