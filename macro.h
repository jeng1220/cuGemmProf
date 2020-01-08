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

#define CUBLASLT_IMMA_ALG -5000
#define CUBLASLT_HEURISTIC_ALG -2