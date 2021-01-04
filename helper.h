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
#include <string>
#include <vector>
#include <cublas_v2.h>

std::string cublasGetErrorString(cublasStatus_t err);

struct GemmDtype_t {
#if (CUBLAS_VER_MAJOR * 10 + CUBLAS_VER_MINOR) >= 110
    cublasComputeType_t compute_type;
    cudaDataType_t scale_type;
#else
    cudaDataType_t compute_type;
#endif
    cudaDataType_t A;
    cudaDataType_t B;
    cudaDataType_t C;
};

struct GemmParam_t {
    cublasOperation_t transa;
    cublasOperation_t transb;
    int b;
    int m;
    int n;
    int k;
    void *alpha;
    void *A;
    int lda;
    void *B;
    int ldb;
    void *beta;
    void *C;
    int ldc;
    GemmDtype_t dtype;
    void *D;
    size_t workspace_size;
    void* workspace;
};

struct ProfResult_t {
    cublasGemmAlgo_t algo;
    float time;
};

struct LtGemmAlgoAttr_t {
    int algo_id;
    int tile_id;
    int splite_k;
    int reduction_scheme;
    int swizzle;
    int custom_option;
    size_t workspace_size;
    float wave_count;
};

struct LtProfResult_t
{
    LtGemmAlgoAttr_t attr;
    ProfResult_t info;
};

GemmDtype_t GemmDtype(int id);
int DtypeToSize(cudaDataType_t dtype);
void* AllocAlphaScale(cudaDataType_t dtype);
std::string AlgoToString(cublasGemmAlgo_t algo);
void PrintResultTile();
void PrintResult(const GemmParam_t& param,
    const std::vector<ProfResult_t>& results, int rank);
void PrintLtResult(const GemmParam_t& param,
    const std::vector<LtProfResult_t>& results, int rank);
