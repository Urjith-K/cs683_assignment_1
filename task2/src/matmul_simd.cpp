// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // Process M in steps of 4, N in steps of 8
    int i = 0;
    for (; i <= M - 4; i += 4) {
        int j = 0;
        for (; j <= N - 8; j += 8) {
            // 4 AVX2 registers to hold 4x8 accumulators initialized to 0.0
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            const float* a0 = A + (i + 0) * static_cast<long>(lda);
            const float* a1 = A + (i + 1) * static_cast<long>(lda);
            const float* a2 = A + (i + 2) * static_cast<long>(lda);
            const float* a3 = A + (i + 3) * static_cast<long>(lda);

            const float* b_base = B + j * static_cast<long>(ldb);

            for (int p = 0; p < K; ++p) {
                // Load 8 contiguous elements of B row j..j+7 at index p
                __m256 b_vec = _mm256_loadu_ps(b_base + p * static_cast<long>(ldb));

                // Broadcast single float elements from A into 256-bit registers
                __m256 a0_vec = _mm256_set1_ps(a0[p]);
                __m256 a1_vec = _mm256_set1_ps(a1[p]);
                __m256 a2_vec = _mm256_set1_ps(a2[p]);
                __m256 a3_vec = _mm256_set1_ps(a3[p]);

                // Perform Fused Multiply-Add: acc += a * b
                acc0 = _mm256_fmadd_ps(a0_vec, b_vec, acc0);
                acc1 = _mm256_fmadd_ps(a1_vec, b_vec, acc1);
                acc2 = _mm256_fmadd_ps(a2_vec, b_vec, acc2);
                acc3 = _mm256_fmadd_ps(a3_vec, b_vec, acc3);
            }

            // Store accumulator vectors into output matrix C
            _mm256_storeu_ps(C + (i + 0) * static_cast<long>(ldc) + j, acc0);
            _mm256_storeu_ps(C + (i + 1) * static_cast<long>(ldc) + j, acc1);
            _mm256_storeu_ps(C + (i + 2) * static_cast<long>(ldc) + j, acc2);
            _mm256_storeu_ps(C + (i + 3) * static_cast<long>(ldc) + j, acc3);
        }

        // Cleanup loop for remaining N elements (if N is not divisible by 8)
        for (; j < N; ++j) {
            for (int sub_i = i; sub_i < i + 4; ++sub_i) {
                float acc = 0.0f;
                const float* a = A + sub_i * static_cast<long>(lda);
                const float* b = B + j * static_cast<long>(ldb);
                for (int p = 0; p < K; ++p) {
                    acc += a[p] * b[p];
                }
                C[sub_i * static_cast<long>(ldc) + j] = acc;
            }
        }
    }

    // Cleanup loop for remaining M elements (if M is not divisible by 4)
    for (; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + i * static_cast<long>(lda);
            const float* b = B + j * static_cast<long>(ldb);
            for (int p = 0; p < K; ++p) {
                acc += a[p] * b[p];
            }
            C[i * static_cast<long>(ldc) + j] = acc;
        }
    }
}
