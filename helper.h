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

struct Param_t {
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

struct Result_t {
    std::string algo;
    float time;
};

GemmDtype_t GetGemmDtype(int id);
int Dtype2Size(cudaDataType_t dtype);
std::string Operation2Str(cublasOperation_t op);
std::string Dtype2Str(cudaDataType_t dtype);
std::string Algo2Str(cublasGemmAlgo_t algo);
void* AllocAlphaScale(cudaDataType_t dtype);
void PrintResult(const char dev_name[], const Param_t& param,
    const std::vector<Result_t>& results);