// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

static inline float hsum256_ps(__m256 v) {
    __m128 vlow   = _mm256_castps256_ps128(v);
    __m128 vhigh  = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(vlow, vhigh);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
    return _mm_cvtss_f32(sum128);
}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    const int NC_BLOCK = 256;

    for (int j_blk = 0; j_blk < N; j_blk += NC_BLOCK) {
        int j_end = (j_blk + NC_BLOCK < N) ? (j_blk + NC_BLOCK) : N;

        int i = 0;
        for (; i <= M - 4; i += 4) {
            const float* a0 = A + static_cast<long>(i + 0) * lda;
            const float* a1 = A + static_cast<long>(i + 1) * lda;
            const float* a2 = A + static_cast<long>(i + 2) * lda;
            const float* a3 = A + static_cast<long>(i + 3) * lda;

            int j = j_blk;
            for (; j <= j_end - 2; j += 2) {
                const float* b0 = B + static_cast<long>(j + 0) * ldb;
                const float* b1 = B + static_cast<long>(j + 1) * ldb;

                const float* b0_next = (j + 2 < j_end) ? (B + static_cast<long>(j + 2) * ldb) : nullptr;
                const float* b1_next = (j + 3 < j_end) ? (B + static_cast<long>(j + 3) * ldb) : nullptr;

                __m256 acc00 = _mm256_setzero_ps();
                __m256 acc01 = _mm256_setzero_ps();
                __m256 acc10 = _mm256_setzero_ps();
                __m256 acc11 = _mm256_setzero_ps();
                __m256 acc20 = _mm256_setzero_ps();
                __m256 acc21 = _mm256_setzero_ps();
                __m256 acc30 = _mm256_setzero_ps();
                __m256 acc31 = _mm256_setzero_ps();

                int p = 0;
                for (; p <= K - 16; p += 16) {
                    _mm_prefetch(reinterpret_cast<const char*>(a0 + p + 16), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(a1 + p + 16), _MM_HINT_T0);

                    if (b0_next && p < 64) {
                        _mm_prefetch(reinterpret_cast<const char*>(b0_next + p), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char*>(b1_next + p), _MM_HINT_T1);
                    }

                    __m256 va0 = _mm256_loadu_ps(a0 + p);
                    __m256 va1 = _mm256_loadu_ps(a1 + p);
                    __m256 va2 = _mm256_loadu_ps(a2 + p);
                    __m256 va3 = _mm256_loadu_ps(a3 + p);

                    __m256 vb0 = _mm256_loadu_ps(b0 + p);
                    __m256 vb1 = _mm256_loadu_ps(b1 + p);

                    acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                    acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                    acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                    acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                    acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                    acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                    acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                    acc31 = _mm256_fmadd_ps(va3, vb1, acc31);

                    va0 = _mm256_loadu_ps(a0 + p + 8);
                    va1 = _mm256_loadu_ps(a1 + p + 8);
                    va2 = _mm256_loadu_ps(a2 + p + 8);
                    va3 = _mm256_loadu_ps(a3 + p + 8);

                    vb0 = _mm256_loadu_ps(b0 + p + 8);
                    vb1 = _mm256_loadu_ps(b1 + p + 8);

                    acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                    acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                    acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                    acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                    acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                    acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                    acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                    acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                }

                for (; p <= K - 8; p += 8) {
                    __m256 va0 = _mm256_loadu_ps(a0 + p);
                    __m256 va1 = _mm256_loadu_ps(a1 + p);
                    __m256 va2 = _mm256_loadu_ps(a2 + p);
                    __m256 va3 = _mm256_loadu_ps(a3 + p);

                    __m256 vb0 = _mm256_loadu_ps(b0 + p);
                    __m256 vb1 = _mm256_loadu_ps(b1 + p);

                    acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                    acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                    acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                    acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                    acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                    acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                    acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                    acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                }

                float s00 = hsum256_ps(acc00), s01 = hsum256_ps(acc01);
                float s10 = hsum256_ps(acc10), s11 = hsum256_ps(acc11);
                float s20 = hsum256_ps(acc20), s21 = hsum256_ps(acc21);
                float s30 = hsum256_ps(acc30), s31 = hsum256_ps(acc31);

                for (; p < K; ++p) {
                    s00 += a0[p] * b0[p];
                    s01 += a0[p] * b1[p];
                    s10 += a1[p] * b0[p];
                    s11 += a1[p] * b1[p];
                    s20 += a2[p] * b0[p];
                    s21 += a2[p] * b1[p];
                    s30 += a3[p] * b0[p];
                    s31 += a3[p] * b1[p];
                }

                C[static_cast<long>(i + 0) * ldc + (j + 0)] = s00;
                C[static_cast<long>(i + 0) * ldc + (j + 1)] = s01;
                C[static_cast<long>(i + 1) * ldc + (j + 0)] = s10;
                C[static_cast<long>(i + 1) * ldc + (j + 1)] = s11;
                C[static_cast<long>(i + 2) * ldc + (j + 0)] = s20;
                C[static_cast<long>(i + 2) * ldc + (j + 1)] = s21;
                C[static_cast<long>(i + 3) * ldc + (j + 0)] = s30;
                C[static_cast<long>(i + 3) * ldc + (j + 1)] = s31;
            }

            for (; j < j_end; ++j) {
                for (int r = 0; r < 4; ++r) {
                    const float* ar = A + static_cast<long>(i + r) * lda;
                    const float* bj = B + static_cast<long>(j) * ldb;
                    __m256 acc = _mm256_setzero_ps();
                    int p = 0;
                    for (; p <= K - 8; p += 8) {
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ar + p), _mm256_loadu_ps(bj + p), acc);
                    }
                    float sum = hsum256_ps(acc);
                    for (; p < K; ++p) sum += ar[p] * bj[p];
                    C[static_cast<long>(i + r) * ldc + j] = sum;
                }
            }
        }

        for (; i < M; ++i) {
            const float* ar = A + static_cast<long>(i) * lda;
            for (int j = j_blk; j < j_end; ++j) {
                const float* bj = B + static_cast<long>(j) * ldb;
                __m256 acc = _mm256_setzero_ps();
                int p = 0;
                for (; p <= K - 8; p += 8) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(ar + p), _mm256_loadu_ps(bj + p), acc);
                }
                float sum = hsum256_ps(acc);
                for (; p < K; ++p) sum += ar[p] * bj[p];
                C[static_cast<long>(i) * ldc + j] = sum;
            }
        }
    }
}