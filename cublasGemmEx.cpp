#include "cublasGemmEx.h"
#include "macro.h"
#include "verify.h"
#include <algorithm>
#include <cfloat>

std::vector<cublasGemmAlgo_t> AllCudaCoreAlgo() {
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
        CUBLAS_GEMM_ALGO18,
        CUBLAS_GEMM_ALGO19,
        CUBLAS_GEMM_ALGO20,
        CUBLAS_GEMM_ALGO21,
        CUBLAS_GEMM_ALGO22,
        CUBLAS_GEMM_ALGO23
    };
    return kAlgos;
}

std::vector<cublasGemmAlgo_t> AllTensorCoreAlgo() {
    const static std::vector<cublasGemmAlgo_t> kAlgos = {
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        CUBLAS_GEMM_ALGO0_TENSOR_OP,
        CUBLAS_GEMM_ALGO1_TENSOR_OP,
        CUBLAS_GEMM_ALGO2_TENSOR_OP,
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
    };
    return kAlgos;
}

std::vector<ProfResult_t> ProfileGemm(const GemmParam_t& param,
    const std::vector<cublasGemmAlgo_t>& algos, int loop, bool debug) {

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaEvent_t start;
    cudaEvent_t end;
    cublasStatus_t ret;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    std::vector<ProfResult_t> results;
    for (auto algo : algos) {

        //param.algo = algo;
        float time = 0.f;
        bool fault = false;

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < loop; ++i) {
            ret = cublasGemmEx(handle,
                               param.transa, param.transb,
                               param.m, param.n, param.k,
                               param.alpha, param.A, param.dtype.Atype, param.lda,
                               param.B, param.dtype.Btype, param.ldb, param.beta,
                               param.C, param.dtype.Ctype, param.ldc,
                               param.dtype.computeType, algo);
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
            fault = !Verify(param.C, param.D, param.m * param.n, param.dtype.Ctype);
            if (fault && debug) {
                std::cerr << "cublasGemmEx" << ", " << AlgoToString(algo) << ", verification failed" << std::endl;
                PrintMatrix(param.A, param.m, param.k, param.lda, param.dtype.Atype);
                PrintMatrix(param.B, param.k, param.n, param.ldb, param.dtype.Btype);
                PrintMatrix(param.C, param.m, param.n, param.ldc, param.dtype.Ctype);
                PrintMatrix(param.D, param.m, param.n, param.ldc, param.dtype.Ctype);
            }
        }
        CUDA_CHECK(cudaMemset(param.C, 0, param.m * param.n * DtypeToSize(param.dtype.Ctype)));

        time = fault ? FLT_MAX : (time / loop);
        results.push_back(ProfResult_t{AlgoToString(algo), time});
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUBLAS_CHECK(cublasDestroy(handle));
    return results;
}
