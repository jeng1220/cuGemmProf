#pragma once
#include "helper.h"
#include <string>
#include <vector>

std::vector<LtProfResult_t> ProfileLtGemm(const GemmParam_t& param, bool all_algo, int loop, bool debug);
