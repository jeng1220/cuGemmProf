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