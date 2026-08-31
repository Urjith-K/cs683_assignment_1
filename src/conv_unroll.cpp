// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 4) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float w = ker[ky * K + kx];
                    a0 += w * in[(oy + ky) * in_stride + ox + kx + 0];
                    a1 += w * in[(oy + ky) * in_stride + ox + kx + 1];
                    a2 += w * in[(oy + ky) * in_stride + ox + kx + 2];
                    a3 += w * in[(oy + ky) * in_stride + ox + kx + 3];
                }
            }

            out[oy * W + ox + 0] = a0;
            out[oy * W + ox + 1] = a1;
            out[oy * W + ox + 2] = a2;
            out[oy * W + ox + 3] = a3;
        }
    }
    //conv_naive(in, out, ker, H, W, K);
}
