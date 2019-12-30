#pragma once
#include <cublas_v2.h>
void* AllocAlphaScale(cudaDataType_t dtype);
void InitMatrix(void* ptr, int w, int h, int ld, cudaDataType_t dtype);
void NaiveGemm(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const void* A, cudaDataType_t a_type, int lda,
    const void* B, cudaDataType_t b_type, int ldb,
    void* C, cudaDataType_t c_type, int ldc,
    cudaDataType_t compute_type);
bool Verify(const void* x, const void* y, int count, cudaDataType_t dtype);
void PrintMatrix(const void* dev_ptr, int w, int h,
    int ld, cudaDataType_t dtype);