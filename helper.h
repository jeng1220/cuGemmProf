#pragma once
#include <cublas_v2.h>
#include <iostream>
#include <string>
#include <vector>

std::string cublasGetErrorString(cublasStatus_t err);

struct GemmDtype_t {
    cudaDataType_t computeType;
    cudaDataType_t Atype;
    cudaDataType_t Btype;
    cudaDataType_t Ctype;
};

struct GemmParam_t {
    cublasOperation_t transa;
    cublasOperation_t transb;
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
    std::string algo;
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

GemmDtype_t GetGemmDtype(int id);
int DtypeToSize(cudaDataType_t dtype);
std::string OperationToString(cublasOperation_t op);
std::string Dtype2String(cudaDataType_t dtype);
std::string AlgoToString(cublasGemmAlgo_t algo);
void* AllocAlphaScale(cudaDataType_t dtype);
void PrintResult(const char dev_name[], const GemmParam_t& param,
    const std::vector<ProfResult_t>& results);
void PrintLtResult(const char dev_name[], const GemmParam_t& param,
    const std::vector<LtProfResult_t>& results);