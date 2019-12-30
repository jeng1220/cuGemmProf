#include "helper.h"
#include "macro.h"
#include <map>

int Dtype2Size(cudaDataType_t dtype) 
{
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