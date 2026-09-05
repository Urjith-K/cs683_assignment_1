// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
     
    // Cache tile sizes tuned for L1/L2 capacity
    constexpr int MC = 64;
    constexpr int NC = 64;
    constexpr int KC = 256;
    constexpr int PREFETCH_DIST = 16; // Prefetch 16 floats ahead (1 cache line)

    for (int i0 = 0; i0 < M; i0 += MC) {
        int i_max = (i0 + MC < M) ? i0 + MC : M;
        for (int j0 = 0; j0 < N; j0 += NC) {
            int j_max = (j0 + NC < N) ? j0 + NC : N;
            for (int k0 = 0; k0 < K; k0 += KC) {
                int k_max = (k0 + KC < K) ? k0 + KC : K;

                // Tiled SIMD Micro-kernel
                int i = i0;
                for (; i <= i_max - 4; i += 4) {
                    int j = j0;
                    for (; j <= j_max - 8; j += 8) {
                        __m256 acc0 = (k0 == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(C + (i + 0) * static_cast<long>(ldc) + j);
                        __m256 acc1 = (k0 == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(C + (i + 1) * static_cast<long>(ldc) + j);
                        __m256 acc2 = (k0 == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(C + (i + 2) * static_cast<long>(ldc) + j);
                        __m256 acc3 = (k0 == 0) ? _mm256_setzero_ps() : _mm256_loadu_ps(C + (i + 3) * static_cast<long>(ldc) + j);

                        const float* a0 = A + (i + 0) * static_cast<long>(lda);
                        const float* a1 = A + (i + 1) * static_cast<long>(lda);
                        const float* a2 = A + (i + 2) * static_cast<long>(lda);
                        const float* a3 = A + (i + 3) * static_cast<long>(lda);

                        const float* b_base = B + j * static_cast<long>(ldb);

                        for (int p = k0; p < k_max; ++p) {
                            // Software prefetching next memory locations into L1 cache (_MM_HINT_T0)
                            _mm_prefetch(reinterpret_cast<const char*>(a0 + p + PREFETCH_DIST), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b_base + (p + PREFETCH_DIST) * static_cast<long>(ldb)), _MM_HINT_T0);

                            __m256 b_vec = _mm256_loadu_ps(b_base + p * static_cast<long>(ldb));

                            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[p]), b_vec, acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[p]), b_vec, acc1);
                            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[p]), b_vec, acc2);
                            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[p]), b_vec, acc3);
                        }

                        _mm256_storeu_ps(C + (i + 0) * static_cast<long>(ldc) + j, acc0);
                        _mm256_storeu_ps(C + (i + 1) * static_cast<long>(ldc) + j, acc1);
                        _mm256_storeu_ps(C + (i + 2) * static_cast<long>(ldc) + j, acc2);
                        _mm256_storeu_ps(C + (i + 3) * static_cast<long>(ldc) + j, acc3);
                    }

                    // N-tail scalar cleanup inside block
                    for (; j < j_max; ++j) {
                        for (int sub_i = i; sub_i < i + 4; ++sub_i) {
                            float acc = (k0 == 0) ? 0.0f : C[sub_i * static_cast<long>(ldc) + j];
                            const float* a = A + sub_i * static_cast<long>(lda);
                            const float* b = B + j * static_cast<long>(ldb);
                            for (int p = k0; p < k_max; ++p) acc += a[p] * b[p];
                            C[sub_i * static_cast<long>(ldc) + j] = acc;
                        }
                    }
                }

                // M-tail scalar cleanup inside block
                for (; i < i_max; ++i) {
                    for (int j = j0; j < j_max; ++j) {
                        float acc = (k0 == 0) ? 0.0f : C[i * static_cast<long>(ldc) + j];
                        const float* a = A + i * static_cast<long>(lda);
                        const float* b = B + j * static_cast<long>(ldb);
                        for (int p = k0; p < k_max; ++p) acc += a[p] * b[p];
                        C[i * static_cast<long>(ldc) + j] = acc;
                    }
                }
            }
        }
    }
}


