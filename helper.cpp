#include "helper.h"
#include "macro.h"
#include <thrust/complex.h>
#include <algorithm>
#include <map>
#include <vector>
#include <cassert>

std::string cublasGetErrorString(cublasStatus_t err) {
    const static std::map<cublasStatus_t, std::string> kErr2Str{
        ADD_KEY_AND_STR(CUBLAS_STATUS_SUCCESS),
        ADD_KEY_AND_STR(CUBLAS_STATUS_NOT_INITIALIZED),
        ADD_KEY_AND_STR(CUBLAS_STATUS_ALLOC_FAILED),
        ADD_KEY_AND_STR(CUBLAS_STATUS_INVALID_VALUE),
        ADD_KEY_AND_STR(CUBLAS_STATUS_ARCH_MISMATCH),
        ADD_KEY_AND_STR(CUBLAS_STATUS_MAPPING_ERROR),
        ADD_KEY_AND_STR(CUBLAS_STATUS_EXECUTION_FAILED),
        ADD_KEY_AND_STR(CUBLAS_STATUS_INTERNAL_ERROR),
        ADD_KEY_AND_STR(CUBLAS_STATUS_NOT_SUPPORTED),
        ADD_KEY_AND_STR(CUBLAS_STATUS_LICENSE_ERROR)
    };
    return kErr2Str.at(err);
}

GemmDtype_t GetGemmDtype(int id) {
    const static std::vector<GemmDtype_t> kGemmDtypes{
        GemmDtype_t{CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F},
        GemmDtype_t{CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I},
        GemmDtype_t{CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F},
        GemmDtype_t{CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F},
        GemmDtype_t{CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F},
        GemmDtype_t{CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F},
        GemmDtype_t{CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F},
        GemmDtype_t{CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F},
        GemmDtype_t{CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F},
        GemmDtype_t{CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F},
    };
    return kGemmDtypes.at(id);
}

int Dtype2Size(cudaDataType_t dtype) {
    const static std::map<cudaDataType_t, int> kDtype2Size{
        {CUDA_R_8I,   1},
        {CUDA_R_16F,  2},
        {CUDA_R_32I,  4},
        {CUDA_R_32F,  4},
        {CUDA_R_64F,  8},
        {CUDA_C_8I,   2},
        {CUDA_C_32F,  8},
        {CUDA_C_64F, 16}
    };
    return kDtype2Size.at(dtype);
}

std::string Operation2Str(cublasOperation_t op) {
    const static std::map<cublasOperation_t, std::string> kOperation2Str{
        ADD_KEY_AND_STR(CUBLAS_OP_N),
        ADD_KEY_AND_STR(CUBLAS_OP_T)
    };
    return kOperation2Str.at(op);
}

std::string Dtype2Str(cudaDataType_t dtype) {
    const static std::map<cudaDataType_t, std::string> kDtype2Str{
        ADD_KEY_AND_STR(CUDA_R_8I),
        ADD_KEY_AND_STR(CUDA_R_16F),
        ADD_KEY_AND_STR(CUDA_R_32I),
        ADD_KEY_AND_STR(CUDA_R_32F),
        ADD_KEY_AND_STR(CUDA_R_64F),
        ADD_KEY_AND_STR(CUDA_C_8I),
        ADD_KEY_AND_STR(CUDA_C_32F),
        ADD_KEY_AND_STR(CUDA_C_64F)
    };
    return kDtype2Str.at(dtype);
}

std::string Algo2Str(cublasGemmAlgo_t algo) {
    const cublasGemmAlgo_t CUBLASLT_IMMA_ALG_ = static_cast<cublasGemmAlgo_t>(CUBLASLT_IMMA_ALG);
    const cublasGemmAlgo_t CUBLASLT_HEURISTIC_ALG_ = static_cast<cublasGemmAlgo_t>(CUBLASLT_HEURISTIC_ALG);
    const static std::map<cublasGemmAlgo_t, std::string> kAlgo2Str{
        ADD_KEY_AND_STR(CUBLASLT_IMMA_ALG_),
        ADD_KEY_AND_STR(CUBLASLT_HEURISTIC_ALG_),
        ADD_KEY_AND_STR(CUBLAS_GEMM_DEFAULT),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO0),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO1),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO2),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO3),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO4),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO5),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO6),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO7),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO8),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO9),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO10),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO11),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO12),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO13),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO14),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO15),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO16),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO17),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO18),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO19),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO20),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO21),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO22),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO23),
        ADD_KEY_AND_STR(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO0_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO1_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO2_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO3_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO4_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO5_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO6_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO7_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO8_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO9_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO10_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO11_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO12_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO13_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO14_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO15_TENSOR_OP)
    };
    return kAlgo2Str.at(algo);
}

void* AllocAlphaScale(cudaDataType_t dtype) {
    void* ptr = nullptr;
    ptr = malloc(Dtype2Size(dtype));
    switch (dtype) {
        case CUDA_R_8I:
            *(reinterpret_cast<char*>(ptr)) = 1;
            break;
        case CUDA_R_16F:
            *(reinterpret_cast<half*>(ptr)) = 1.f;
            break;
        case CUDA_R_32I:
            *(reinterpret_cast<int*>(ptr)) = 1;
            break;
        case CUDA_R_32F:
            *(reinterpret_cast<float*>(ptr)) = 1.f;
            break;
        case CUDA_R_64F:
            *(reinterpret_cast<double*>(ptr)) = 1.0;
            break;
        case CUDA_C_8I:
            *(reinterpret_cast< thrust::complex<char>* >(ptr)) = 1;
            break;
        case CUDA_C_32F:
            *(reinterpret_cast< thrust::complex<float>* >(ptr)) = 1.f;
            break;
        case CUDA_C_64F:
            *(reinterpret_cast< thrust::complex<double>* >(ptr)) = 1.0;
            break;
        default:
            assert(false);
    }
    return ptr;
}

std::string Mask2Str(const std::vector<bool>& mask) {
    std::string info;
    auto count = std::count_if(mask.begin(), mask.end(),
        [](bool x) { return x; });
    if (count == mask.size()) {
        info = "all meet, ";
    }
    else {
        info = "(";
        for (auto bit : mask) {
            info += std::to_string(static_cast<int>(bit)) + ".";
        }
        info += "), ";
    }
    return info;
}

std::string Dp4aRestrictions(const GemmParam_t& param) {
    std::vector<bool> mask(2);
    mask[0] = param.lda % 4 == 0;
    mask[1] = param.ldb % 4 == 0;
    return Mask2Str(mask);
}

std::string TensorCoreRestrictions(const GemmParam_t& param) {
    // refer to https://docs.nvidia.com/cuda/cublas/#tensorop-restrictions
    std::vector<bool> mask(8);
    mask[0] = param.m % 4 == 0;
    mask[1] = param.k % 8 == 0;
    mask[2] = reinterpret_cast<intptr_t>(param.A) % 16 == 0;
    mask[3] = reinterpret_cast<intptr_t>(param.B) % 16 == 0;
    mask[4] = reinterpret_cast<intptr_t>(param.C) % 16 == 0;
    mask[5] = param.lda % (16 / Dtype2Size(param.dtype.Atype)) == 0;
    mask[6] = param.ldb % (16 / Dtype2Size(param.dtype.Btype)) == 0;
    mask[7] = param.ldc % (16 / Dtype2Size(param.dtype.Ctype)) == 0;
    return Mask2Str(mask);
}

bool SortResult (const ProfResult_t& x, const ProfResult_t& y) { 
    return (x.time < y.time); 
}

void PrintResult(const char dev_name[], const GemmParam_t& param,
    const std::vector<ProfResult_t>& results) {
    std::cout << "device, op(A), op(B), "
        "m, n, k, ComputeType, Atype, Btype, Ctype, "
        "Dp4aRestrictions(lda.ldb), TensorCoreRestrictions(m.k.A.B.C.lda.ldb.ldc), "
        "algo, time(ms), GFLOPS" << std::endl;

    std::string all_info;
    all_info = std::string(dev_name) + ", "
        + Operation2Str(param.transa) + ", "
        + Operation2Str(param.transb) + ", "
        + std::to_string(param.m) + ", "
        + std::to_string(param.n) + ", "
        + std::to_string(param.k) + ", "
        + Dtype2Str(param.dtype.computeType) + ", "
        + Dtype2Str(param.dtype.Atype) + ", "
        + Dtype2Str(param.dtype.Btype) + ", "
        + Dtype2Str(param.dtype.Ctype) + ", ";

    all_info += Dp4aRestrictions(param);
    all_info += TensorCoreRestrictions(param);

    float workload = (2.f * param.m * param.n * param.k) * 1e-9;

    std::vector<ProfResult_t> order = results;
    std::sort(order.begin(), order.end(), SortResult);

    for (auto result : order) {
        float gflops = workload / (result.time * 1e-3);

        std::cout << all_info << result.algo << ", " << 
            (result.time == FLT_MAX ? NAN : result.time) << ", " <<
            (result.time == FLT_MAX ? NAN : gflops) << std::endl;
    }
}