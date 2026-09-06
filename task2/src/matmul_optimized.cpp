// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.


#include <immintrin.h>
#include "matmul.h"

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int RT = 32;
    const int CT = 32;
    const int PD = 64;

    for (int i0 = 0; i0 < M; i0 += RT) {
        int i_end = (i0 + RT < M) ? (i0 + RT) : M;
        for (int j0 = 0; j0 < N; j0 += CT) {
            int j_end = (j0 + CT < N) ? (j0 + CT) : N;

            for (int i = i0; i < i_end; ++i) {
                int j = j0;

                for (; j + 3 < j_end; j += 4) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    int p = 0;
                    for (; p + 7 < K; p += 8) {
                        if (p + PD + 7 < K) {
                            _mm_prefetch(reinterpret_cast<const char*>(&A[i * lda + p + PD]), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(&B[j * ldb + p + PD]), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(&B[(j + 1) * ldb + p + PD]), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(&B[(j + 2) * ldb + p + PD]), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(&B[(j + 3) * ldb + p + PD]), _MM_HINT_T0);
                        }

                        __m256 a = _mm256_loadu_ps(&A[i * lda + p]);
                        __m256 b0 = _mm256_loadu_ps(&B[j * ldb + p]);
                        __m256 b1 = _mm256_loadu_ps(&B[(j + 1) * ldb + p]);
                        __m256 b2 = _mm256_loadu_ps(&B[(j + 2) * ldb + p]);
                        __m256 b3 = _mm256_loadu_ps(&B[(j + 3) * ldb + p]);

                        acc0 = _mm256_fmadd_ps(a, b0, acc0);
                        acc1 = _mm256_fmadd_ps(a, b1, acc1);
                        acc2 = _mm256_fmadd_ps(a, b2, acc2);
                        acc3 = _mm256_fmadd_ps(a, b3, acc3);
                    }

                    float temp0[8], temp1[8], temp2[8], temp3[8];
                    _mm256_storeu_ps(temp0, acc0);
                    _mm256_storeu_ps(temp1, acc1);
                    _mm256_storeu_ps(temp2, acc2);
                    _mm256_storeu_ps(temp3, acc3);

                    float result0 = temp0[0] + temp0[1] + temp0[2] + temp0[3] + temp0[4] + temp0[5] + temp0[6] + temp0[7];
                    float result1 = temp1[0] + temp1[1] + temp1[2] + temp1[3] + temp1[4] + temp1[5] + temp1[6] + temp1[7];
                    float result2 = temp2[0] + temp2[1] + temp2[2] + temp2[3] + temp2[4] + temp2[5] + temp2[6] + temp2[7];
                    float result3 = temp3[0] + temp3[1] + temp3[2] + temp3[3] + temp3[4] + temp3[5] + temp3[6] + temp3[7];

                    for (; p < K; ++p) {
                        result0 += A[i * lda + p] * B[j * ldb + p];
                        result1 += A[i * lda + p] * B[(j + 1) * ldb + p];
                        result2 += A[i * lda + p] * B[(j + 2) * ldb + p];
                        result3 += A[i * lda + p] * B[(j + 3) * ldb + p];
                    }

                    C[static_cast<long>(i) * ldc + j] = result0;
                    C[static_cast<long>(i) * ldc + j + 1] = result1;
                    C[static_cast<long>(i) * ldc + j + 2] = result2;
                    C[static_cast<long>(i) * ldc + j + 3] = result3;
                }

                for (; j < j_end; ++j) {
                    __m256 acc = _mm256_setzero_ps();
                    int p = 0;

                    for (; p + 7 < K; p += 8) {
                        if (p + PD + 7 < K) {
                            _mm_prefetch(reinterpret_cast<const char*>(&A[i * lda + p + PD]), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(&B[j * ldb + p + PD]), _MM_HINT_T0);
                        }

                        __m256 a = _mm256_loadu_ps(&A[i * lda + p]);
                        __m256 b = _mm256_loadu_ps(&B[j * ldb + p]);
                        acc = _mm256_fmadd_ps(a, b, acc);
                    }

                    float temp[8];
                    _mm256_storeu_ps(temp, acc);

                    float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

                    for (; p < K; ++p) {
                        result += A[i * lda + p] * B[j * ldb + p];
                    }

                    C[static_cast<long>(i) * ldc + j] = result;
                }
            }
        }
    }
}