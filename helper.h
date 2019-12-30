#pragma once
#include <cublas_v2.h>
#include <string>
int Dtype2Size(cudaDataType_t dtype);
std::string cublasGetErrorString(cublasStatus_t err);