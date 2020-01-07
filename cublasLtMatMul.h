#pragma once
#include "helper.h"
#include <string>
#include <vector>

std::vector<Result_t> ProfileGemmLt(const Param_t& param, bool all_algo, int loop, bool debug);