#pragma once
#include <cublas_v2.h>
void* AllocAlphaScale(cudaDataType_t dtype);
void InitMatrix(void* ptr, int w, int h, int ld, cudaDataType_t dtype);
void NaiveGemmNN(
    int m, int n, int k,
    void* A, int lda,
    void* B, int ldb,
    void* C, int ldc,
    int gemm_type);
bool Verify(void* x, void* y, int count, cudaDataType_t dtype);
void PrintMatrix(float* dev_ptr, int w, int h, int ld);