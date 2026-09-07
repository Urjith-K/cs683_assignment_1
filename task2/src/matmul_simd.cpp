// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(vlow, vhigh);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
    return _mm_cvtss_f32(sum128);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a_row = A + static_cast<long>(i) * lda;

        for (int j = 0; j < N; ++j) {
            const float* b_col = B + static_cast<long>(j) * ldb;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            int p = 0;
            // Unroll K by 16 floats (64 bytes = 1 full cache line)
            for (; p <= K - 16; p += 16) {
                __m256 va0 = _mm256_loadu_ps(a_row + p);
                __m256 vb0 = _mm256_loadu_ps(b_col + p);
                acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

                __m256 va1 = _mm256_loadu_ps(a_row + p + 8);
                __m256 vb1 = _mm256_loadu_ps(b_col + p + 8);
                acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
            }

            // Remainder 8 floats
            for (; p <= K - 8; p += 8) {
                __m256 va = _mm256_loadu_ps(a_row + p);
                __m256 vb = _mm256_loadu_ps(b_col + p);
                acc0 = _mm256_fmadd_ps(va, vb, acc0);
            }

            float sum = hsum256_ps(_mm256_add_ps(acc0, acc1));

            // Scalar tail
            for (; p < K; ++p) {
                sum += a_row[p] * b_col[p];
            }

            C[static_cast<long>(i) * ldc + j] = sum;
        }
    }
}