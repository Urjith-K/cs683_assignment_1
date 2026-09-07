// matmul_optimized.cpp — STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment — loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2) — and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

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

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int NC = 256;

    for (int j_blk = 0; j_blk < N; j_blk += NC) {
        int j_end = (j_blk + NC < N) ? (j_blk + NC) : N;

        int i = 0;
        // 4x3 register microkernel (12 accumulators)
        for (; i <= M - 4; i += 4) {
            const float* a0 = A + static_cast<long>(i + 0) * lda;
            const float* a1 = A + static_cast<long>(i + 1) * lda;
            const float* a2 = A + static_cast<long>(i + 2) * lda;
            const float* a3 = A + static_cast<long>(i + 3) * lda;

            if (i + 4 < M) {
                _mm_prefetch(reinterpret_cast<const char*>(A + static_cast<long>(i + 4) * lda), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(A + static_cast<long>(i + 5) * lda), _MM_HINT_T0);
            }

            int j = j_blk;
            for (; j <= j_end - 3; j += 3) {
                const float* b0 = B + static_cast<long>(j + 0) * ldb;
                const float* b1 = B + static_cast<long>(j + 1) * ldb;
                const float* b2 = B + static_cast<long>(j + 2) * ldb;

                const float* b0_next = (j + 3 < j_end) ? (B + static_cast<long>(j + 3) * ldb) : nullptr;

                __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
                __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
                __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
                __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();

                int p = 0;
                // Unroll by 32 floats along K (4 AVX2 vectors = 128 bytes)
                for (; p <= K - 32; p += 32) {
                    _mm_prefetch(reinterpret_cast<const char*>(a0 + p + 16), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(a1 + p + 16), _MM_HINT_T0);
                    if (b0_next && p < 64) {
                        _mm_prefetch(reinterpret_cast<const char*>(b0_next + p), _MM_HINT_T1);
                    }

                    // Vector 0 (floats 0..7)
                    __m256 vb0 = _mm256_loadu_ps(b0 + p);
                    __m256 vb1 = _mm256_loadu_ps(b1 + p);
                    __m256 vb2 = _mm256_loadu_ps(b2 + p);

                    __m256 va = _mm256_loadu_ps(a0 + p);
                    c00 = _mm256_fmadd_ps(va, vb0, c00);
                    c01 = _mm256_fmadd_ps(va, vb1, c01);
                    c02 = _mm256_fmadd_ps(va, vb2, c02);

                    va = _mm256_loadu_ps(a1 + p);
                    c10 = _mm256_fmadd_ps(va, vb0, c10);
                    c11 = _mm256_fmadd_ps(va, vb1, c11);
                    c12 = _mm256_fmadd_ps(va, vb2, c12);

                    va = _mm256_loadu_ps(a2 + p);
                    c20 = _mm256_fmadd_ps(va, vb0, c20);
                    c21 = _mm256_fmadd_ps(va, vb1, c21);
                    c22 = _mm256_fmadd_ps(va, vb2, c22);

                    va = _mm256_loadu_ps(a3 + p);
                    c30 = _mm256_fmadd_ps(va, vb0, c30);
                    c31 = _mm256_fmadd_ps(va, vb1, c31);
                    c32 = _mm256_fmadd_ps(va, vb2, c32);

                    // Vector 1 (floats 8..15)
                    vb0 = _mm256_loadu_ps(b0 + p + 8);
                    vb1 = _mm256_loadu_ps(b1 + p + 8);
                    vb2 = _mm256_loadu_ps(b2 + p + 8);

                    va = _mm256_loadu_ps(a0 + p + 8);
                    c00 = _mm256_fmadd_ps(va, vb0, c00);
                    c01 = _mm256_fmadd_ps(va, vb1, c01);
                    c02 = _mm256_fmadd_ps(va, vb2, c02);

                    va = _mm256_loadu_ps(a1 + p + 8);
                    c10 = _mm256_fmadd_ps(va, vb0, c10);
                    c11 = _mm256_fmadd_ps(va, vb1, c11);
                    c12 = _mm256_fmadd_ps(va, vb2, c12);

                    va = _mm256_loadu_ps(a2 + p + 8);
                    c20 = _mm256_fmadd_ps(va, vb0, c20);
                    c21 = _mm256_fmadd_ps(va, vb1, c21);
                    c22 = _mm256_fmadd_ps(va, vb2, c22);

                    va = _mm256_loadu_ps(a3 + p + 8);
                    c30 = _mm256_fmadd_ps(va, vb0, c30);
                    c31 = _mm256_fmadd_ps(va, vb1, c31);
                    c32 = _mm256_fmadd_ps(va, vb2, c32);

                    // Vector 2 (floats 16..23)
                    vb0 = _mm256_loadu_ps(b0 + p + 16);
                    vb1 = _mm256_loadu_ps(b1 + p + 16);
                    vb2 = _mm256_loadu_ps(b2 + p + 16);

                    va = _mm256_loadu_ps(a0 + p + 16);
                    c00 = _mm256_fmadd_ps(va, vb0, c00);
                    c01 = _mm256_fmadd_ps(va, vb1, c01);
                    c02 = _mm256_fmadd_ps(va, vb2, c02);

                    va = _mm256_loadu_ps(a1 + p + 16);
                    c10 = _mm256_fmadd_ps(va, vb0, c10);
                    c11 = _mm256_fmadd_ps(va, vb1, c11);
                    c12 = _mm256_fmadd_ps(va, vb2, c12);

                    va = _mm256_loadu_ps(a2 + p + 16);
                    c20 = _mm256_fmadd_ps(va, vb0, c20);
                    c21 = _mm256_fmadd_ps(va, vb1, c21);
                    c22 = _mm256_fmadd_ps(va, vb2, c22);

                    va = _mm256_loadu_ps(a3 + p + 16);
                    c30 = _mm256_fmadd_ps(va, vb0, c30);
                    c31 = _mm256_fmadd_ps(va, vb1, c31);
                    c32 = _mm256_fmadd_ps(va, vb2, c32);

                    // Vector 3 (floats 24..31)
                    vb0 = _mm256_loadu_ps(b0 + p + 24);
                    vb1 = _mm256_loadu_ps(b1 + p + 24);
                    vb2 = _mm256_loadu_ps(b2 + p + 24);

                    va = _mm256_loadu_ps(a0 + p + 24);
                    c00 = _mm256_fmadd_ps(va, vb0, c00);
                    c01 = _mm256_fmadd_ps(va, vb1, c01);
                    c02 = _mm256_fmadd_ps(va, vb2, c02);

                    va = _mm256_loadu_ps(a1 + p + 24);
                    c10 = _mm256_fmadd_ps(va, vb0, c10);
                    c11 = _mm256_fmadd_ps(va, vb1, c11);
                    c12 = _mm256_fmadd_ps(va, vb2, c12);

                    va = _mm256_loadu_ps(a2 + p + 24);
                    c20 = _mm256_fmadd_ps(va, vb0, c20);
                    c21 = _mm256_fmadd_ps(va, vb1, c21);
                    c22 = _mm256_fmadd_ps(va, vb2, c22);

                    va = _mm256_loadu_ps(a3 + p + 24);
                    c30 = _mm256_fmadd_ps(va, vb0, c30);
                    c31 = _mm256_fmadd_ps(va, vb1, c31);
                    c32 = _mm256_fmadd_ps(va, vb2, c32);
                }

                // Cleanup tail in steps of 8 floats
                for (; p <= K - 8; p += 8) {
                    __m256 vb0 = _mm256_loadu_ps(b0 + p);
                    __m256 vb1 = _mm256_loadu_ps(b1 + p);
                    __m256 vb2 = _mm256_loadu_ps(b2 + p);

                    __m256 va = _mm256_loadu_ps(a0 + p);
                    c00 = _mm256_fmadd_ps(va, vb0, c00);
                    c01 = _mm256_fmadd_ps(va, vb1, c01);
                    c02 = _mm256_fmadd_ps(va, vb2, c02);

                    va = _mm256_loadu_ps(a1 + p);
                    c10 = _mm256_fmadd_ps(va, vb0, c10);
                    c11 = _mm256_fmadd_ps(va, vb1, c11);
                    c12 = _mm256_fmadd_ps(va, vb2, c12);

                    va = _mm256_loadu_ps(a2 + p);
                    c20 = _mm256_fmadd_ps(va, vb0, c20);
                    c21 = _mm256_fmadd_ps(va, vb1, c21);
                    c22 = _mm256_fmadd_ps(va, vb2, c22);

                    va = _mm256_loadu_ps(a3 + p);
                    c30 = _mm256_fmadd_ps(va, vb0, c30);
                    c31 = _mm256_fmadd_ps(va, vb1, c31);
                    c32 = _mm256_fmadd_ps(va, vb2, c32);
                }

                float s00 = hsum256_ps(c00), s01 = hsum256_ps(c01), s02 = hsum256_ps(c02);
                float s10 = hsum256_ps(c10), s11 = hsum256_ps(c11), s12 = hsum256_ps(c12);
                float s20 = hsum256_ps(c20), s21 = hsum256_ps(c21), s22 = hsum256_ps(c22);
                float s30 = hsum256_ps(c30), s31 = hsum256_ps(c31), s32 = hsum256_ps(c32);

                for (; p < K; ++p) {
                    s00 += a0[p] * b0[p]; s01 += a0[p] * b1[p]; s02 += a0[p] * b2[p];
                    s10 += a1[p] * b0[p]; s11 += a1[p] * b1[p]; s12 += a1[p] * b2[p];
                    s20 += a2[p] * b0[p]; s21 += a2[p] * b1[p]; s22 += a2[p] * b2[p];
                    s30 += a3[p] * b0[p]; s31 += a3[p] * b1[p]; s32 += a3[p] * b2[p];
                }

                C[static_cast<long>(i + 0) * ldc + (j + 0)] = s00;
                C[static_cast<long>(i + 0) * ldc + (j + 1)] = s01;
                C[static_cast<long>(i + 0) * ldc + (j + 2)] = s02;

                C[static_cast<long>(i + 1) * ldc + (j + 0)] = s10;
                C[static_cast<long>(i + 1) * ldc + (j + 1)] = s11;
                C[static_cast<long>(i + 1) * ldc + (j + 2)] = s12;

                C[static_cast<long>(i + 2) * ldc + (j + 0)] = s20;
                C[static_cast<long>(i + 2) * ldc + (j + 1)] = s21;
                C[static_cast<long>(i + 2) * ldc + (j + 2)] = s22;

                C[static_cast<long>(i + 3) * ldc + (j + 0)] = s30;
                C[static_cast<long>(i + 3) * ldc + (j + 1)] = s31;
                C[static_cast<long>(i + 3) * ldc + (j + 2)] = s32;
            }

            // Cleanup remaining columns
            for (; j < j_end; ++j) {
                const float* bj = B + static_cast<long>(j) * ldb;
                for (int r = 0; r < 4; ++r) {
                    const float* ar = A + static_cast<long>(i + r) * lda;
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

        // Cleanup residual rows (and M=1 for llama token generation)
        for (; i < M; ++i) {
            const float* ar = A + static_cast<long>(i) * lda;
            int j = j_blk;
            for (; j <= j_end - 2; j += 2) {
                const float* b0 = B + static_cast<long>(j + 0) * ldb;
                const float* b1 = B + static_cast<long>(j + 1) * ldb;

                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                int p = 0;
                for (; p <= K - 16; p += 16) {
                    __m256 va0 = _mm256_loadu_ps(ar + p);
                    __m256 va1 = _mm256_loadu_ps(ar + p + 8);
                    acc0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + p), acc0);
                    acc1 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b1 + p), acc1);
                    acc0 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b0 + p + 8), acc0);
                    acc1 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + p + 8), acc1);
                }
                for (; p <= K - 8; p += 8) {
                    __m256 va = _mm256_loadu_ps(ar + p);
                    acc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p), acc0);
                    acc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p), acc1);
                }
                float s0 = hsum256_ps(acc0);
                float s1 = hsum256_ps(acc1);
                for (; p < K; ++p) {
                    s0 += ar[p] * b0[p];
                    s1 += ar[p] * b1[p];
                }
                C[static_cast<long>(i) * ldc + (j + 0)] = s0;
                C[static_cast<long>(i) * ldc + (j + 1)] = s1;
            }
            for (; j < j_end; ++j) {
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