/* Copyright 2020 Jeng Bai-Cheng
*
* Permission is hereby granted, free of charge, to any person obtaining a copy of
* this software and associated documentation files (the "Software"), to deal in
* the Software without restriction, including without limitation the rights to
* use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
* of the Software, and to permit persons to whom the Software is furnished to do
* so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
*  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "cublasGemmEx.h"
#include <cfloat>
#include <iostream>
#include "macro.h"
#include "verify.h"

std::vector<cublasGemmAlgo_t> AllCudaCoreAlgo() {
#if (CUBLAS_VER_MAJOR * 10 + CUBLAS_VER_MINOR) < 90
    const cublasGemmAlgo_t CUBLAS_GEMM_DEFAULT = CUBLAS_GEMM_DFALT;
#endif
    const static std::vector<cublasGemmAlgo_t> kAlgos{
        CUBLAS_GEMM_DEFAULT,
        CUBLAS_GEMM_ALGO0,
        CUBLAS_GEMM_ALGO1,
        CUBLAS_GEMM_ALGO2,
        CUBLAS_GEMM_ALGO3,
        CUBLAS_GEMM_ALGO4,
        CUBLAS_GEMM_ALGO5,
        CUBLAS_GEMM_ALGO6,
        CUBLAS_GEMM_ALGO7,
#if (CUBLAS_VER_MAJOR * 10 + CUBLAS_VER_MINOR) >= 90
        CUBLAS_GEMM_ALGO8,
        CUBLAS_GEMM_ALGO9,
        CUBLAS_GEMM_ALGO10,
        CUBLAS_GEMM_ALGO11,
        CUBLAS_GEMM_ALGO12,
        CUBLAS_GEMM_ALGO13,
        CUBLAS_GEMM_ALGO14,
        CUBLAS_GEMM_ALGO15,
        CUBLAS_GEMM_ALGO16,
        CUBLAS_GEMM_ALGO17,
#endif
#if (CUBLAS_VER_MAJOR * 10 + CUBLAS_VER_MINOR) >= 92
        CUBLAS_GEMM_ALGO18,
        CUBLAS_GEMM_ALGO19,
        CUBLAS_GEMM_ALGO20,
        CUBLAS_GEMM_ALGO21,
        CUBLAS_GEMM_ALGO22,
        CUBLAS_GEMM_ALGO23
#endif
    };
    return kAlgos;
}

std::vector<cublasGemmAlgo_t> AllTensorCoreAlgo() {
    const static std::vector<cublasGemmAlgo_t> kAlgos = {
#if (CUBLAS_VER_MAJOR * 10 + CUBLAS_VER_MINOR) >= 90
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        CUBLAS_GEMM_ALGO0_TENSOR_OP,
        CUBLAS_GEMM_ALGO1_TENSOR_OP,
        CUBLAS_GEMM_ALGO2_TENSOR_OP,
#endif
#if (CUBLAS_VER_MAJOR * 10 + CUBLAS_VER_MINOR) >= 92
        CUBLAS_GEMM_ALGO3_TENSOR_OP,
        CUBLAS_GEMM_ALGO4_TENSOR_OP,
        CUBLAS_GEMM_ALGO5_TENSOR_OP,
        CUBLAS_GEMM_ALGO6_TENSOR_OP,
        CUBLAS_GEMM_ALGO7_TENSOR_OP,
        CUBLAS_GEMM_ALGO8_TENSOR_OP,
        CUBLAS_GEMM_ALGO9_TENSOR_OP,
        CUBLAS_GEMM_ALGO10_TENSOR_OP,
        CUBLAS_GEMM_ALGO11_TENSOR_OP,
        CUBLAS_GEMM_ALGO12_TENSOR_OP,
        CUBLAS_GEMM_ALGO13_TENSOR_OP,
        CUBLAS_GEMM_ALGO14_TENSOR_OP,
        CUBLAS_GEMM_ALGO15_TENSOR_OP,
#endif
    };
    return kAlgos;
}

std::vector<ProfResult_t> ProfileGemm(const GemmParam_t& param,
    const std::vector<cublasGemmAlgo_t>& algos, int loop, double threshold, bool debug) {

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaEvent_t start;
    cudaEvent_t end;
    cublasStatus_t ret;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    long long int strideA = param.m * param.k;
    long long int strideB = param.n * param.k;
    long long int strideC = param.m * param.n;

    std::vector<ProfResult_t> results;
    for (auto algo : algos) {

        //param.algo = algo;
        float time = 0.f;
        bool fault = false;

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < loop; ++i) {
#if 1
            ret = cublasGemmStridedBatchedEx(handle,
                param.transa, param.transb,
                param.m, param.n, param.k,
                param.alpha,
                param.A, param.dtype.A, param.lda, strideA,
                param.B, param.dtype.B, param.ldb, strideB,
                param.beta,
                param.C, param.dtype.C, param.ldc, strideC,
                param.b,
                param.dtype.compute_type, algo);
#else
            ret = cublasGemmEx(handle,
                               param.transa, param.transb,
                               param.m, param.n, param.k,
                               param.alpha, param.A, param.dtype.A, param.lda,
                               param.B, param.dtype.B, param.ldb, param.beta,
                               param.C, param.dtype.C, param.ldc,
                               param.dtype.compute_type, algo);
#endif
            if (ret != CUBLAS_STATUS_SUCCESS) {
                fault = true;
                if (debug) {
                    std::cerr << "cublasGemmEx" << ", " << AlgoToString(algo) << 
                        ", " << cublasGetErrorString(ret) << std::endl;
                }
                break;
            }
        }
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&time, start, end));

        if (!fault) {
            auto relative_err = Verify(param.C, param.D, param.b * param.m * param.n, param.dtype.C);
            if (relative_err > threshold) fault = true;
            if (fault && debug) {
                std::cerr << "cublasGemmEx" << ", " << AlgoToString(algo) << ", verification failed" << std::endl;
                PrintMatrix(param.A, param.m, param.k, param.lda, param.dtype.A);
                PrintMatrix(param.B, param.k, param.n, param.ldb, param.dtype.B);
                PrintMatrix(param.C, param.m, param.n, param.ldc, param.dtype.C);
                PrintMatrix(param.D, param.m, param.n, param.ldc, param.dtype.C);
            }
        }
        CUDA_CHECK(cudaMemset(param.C, 0, param.m * param.n * DtypeToSize(param.dtype.C)));

        time = fault ? FLT_MAX : (time / loop);
        results.push_back(ProfResult_t{algo, time});
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUBLAS_CHECK(cublasDestroy(handle));
    return results;
}
