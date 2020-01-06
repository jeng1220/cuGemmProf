#pragma once
#include <cublas_v2.h>
#include <iostream>
#include <string>
int Dtype2Size(cudaDataType_t dtype);
std::string cublasGetErrorString(cublasStatus_t err);
std::string Operation2Str(cublasOperation_t op);
std::string Dtype2Str(cudaDataType_t dtype);
std::string Algo2Str(cublasGemmAlgo_t algo);

struct Dtypes_t {
    cudaDataType_t computeType;
    cudaDataType_t Atype;
    cudaDataType_t Btype;
    cudaDataType_t Ctype;
    std::string GetDtypeInfo() {
        return Dtype2Str(computeType) + ", "
             + Dtype2Str(Atype) + ", "
             + Dtype2Str(Btype) + ", "
             + Dtype2Str(Ctype) + ", ";
    }
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
    Dtypes_t dtype;
    void *D;
    size_t workspace_size;
    void* workspace;
    std::string GetDimsInfo(void) {
        return Operation2Str(transa) + ", "
             + Operation2Str(transb) + ", "
             + std::to_string(m) + ", "
             + std::to_string(n) + ", "
             + std::to_string(k) + ", ";
    }
};

struct Result_t {
    cublasGemmAlgo_t algo;
    float time;
    float gflops;
    friend std::ostream& operator<<(std::ostream& os, const Result_t& x);
};

bool SortResult (const Result_t& x, const Result_t& y);