#pragma once
#include "helper.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

std::vector<cublasGemmAlgo_t> AllCudaCoreAlgo();
std::vector<cublasGemmAlgo_t> AllTensorCoreAlgo();

void ProfileGemm(const Param_t& param, const std::vector<cublasGemmAlgo_t>& algos,
    const std::string& config_info, int loop);
