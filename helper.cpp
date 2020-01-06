#include "helper.h"
#include "macro.h"
#include <thrust/complex.h>
#include <map>
#include <vector>
#include <cassert>

Dtypes_t GetGemmDtype(int id) {
    const static std::vector<Dtypes_t> kGemmDtypes{
        Dtypes_t{CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F},
        Dtypes_t{CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I},
        Dtypes_t{CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F},
        Dtypes_t{CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F},
        Dtypes_t{CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F},
        Dtypes_t{CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F},
        Dtypes_t{CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F},
        Dtypes_t{CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F},
        Dtypes_t{CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F},
        Dtypes_t{CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F},
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

std::string cublasGetErrorString(cublasStatus_t err)
{
    const static std::map<cublasStatus_t, std::string> kErr2Str{
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

std::ostream& operator<<(std::ostream& os, const Result_t& x) {
    os << Algo2Str(x.algo) << ", " << x.time << ", " << x.gflops;
    return os;
}

bool SortResult (const Result_t& x, const Result_t& y) { 
    return (x.time < y.time); 
}

void* AllocAlphaScale(cudaDataType_t dtype)
{
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
