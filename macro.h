#pragma once
#include "helper.h"
#include <cuda_runtime.h>

#define ADD_KEY_AND_STR(x) {x, #x}

#define CUDA_CHECK(apiFuncCall)                                                \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
            __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define CUBLAS_CHECK(apiFuncCall)                                                     \
do {                                                                                  \
    cublasStatus_t _status = apiFuncCall;                                             \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                           \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",          \
            __FILE__, __LINE__, #apiFuncCall, cublasGetErrorString(_status).c_str()); \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
} while (0)

#define __CUBLASLT_DEFAULT_ALG__       -2
#define __CUBLASLT_DEFAULT_IMMA_ALG__  -3
#define __CUBLASLT_1ST_HEURISTIC_ALG__ -4
#define __CUBLASLT_ALL_ALG__           -5