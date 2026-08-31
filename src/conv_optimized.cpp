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
    const int TH = 32;
    const int TW = 256;

    for (int y0 = 0; y0 < H; y0 += TH) {
        int yend = (y0 + TH < H) ? (y0 + TH) : H;

        for (int x0 = 0; x0 < W; x0 += TW) {
            int xend = (x0 + TW < W) ? (x0 + TW) : W;

            for (int oy = y0; oy < yend; ++oy) {
                for (int ox = x0; ox < xend; ++ox) {
                    out[oy * W + ox] = 0.0f;
                }
            }

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 w = _mm256_broadcast_ss(&ker[ky * K + kx]);

                    for (int oy = y0; oy < yend; ++oy) {
                        for (int ox = x0; ox < xend; ox += 8) {
                            __m256 v = _mm256_loadu_ps(in + (oy + ky) * in_stride + kx +ox);
                            __m256 acc = _mm256_loadu_ps( out + oy * W + ox);
                            _mm256_storeu_ps(out + oy * W + ox, _mm256_fmadd_ps(w, v, acc)); 
                        }
                    }
                }
            }
        }
    }
    //conv_naive(in, out, ker, H, W, K);
}
