// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
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
                    float acc = 0.0f;

                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] *
                                   ker[ky * K + kx];
                        }
                    }

                    out[oy * W + ox] = acc;
                }
            }
        }
    }
    //conv_naive(in, out, ker, H, W, K);
}
