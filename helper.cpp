#include "helper.h"
#include "macro.h"
#include <cublasLt.h>
#include <thrust/complex.h>
#include <algorithm>
#include <map>
#include <vector>
#include <cassert>

std::string cublasGetErrorString(cublasStatus_t err) {
    const static std::map<cublasStatus_t, std::string> kErr2Str{
        ADD_KEY_AND_STR(CUBLAS_STATUS_SUCCESS),
        ADD_KEY_AND_STR(CUBLAS_STATUS_NOT_INITIALIZED),
        ADD_KEY_AND_STR(CUBLAS_STATUS_ALLOC_FAILED),
        ADD_KEY_AND_STR(CUBLAS_STATUS_INVALID_VALUE),
        ADD_KEY_AND_STR(CUBLAS_STATUS_ARCH_MISMATCH),
        ADD_KEY_AND_STR(CUBLAS_STATUS_MAPPING_ERROR),
        ADD_KEY_AND_STR(CUBLAS_STATUS_EXECUTION_FAILED),
        ADD_KEY_AND_STR(CUBLAS_STATUS_INTERNAL_ERROR),
        ADD_KEY_AND_STR(CUBLAS_STATUS_NOT_SUPPORTED),
        ADD_KEY_AND_STR(CUBLAS_STATUS_LICENSE_ERROR)
    };
    return kErr2Str.at(err);
}

GemmDtype_t GetGemmDtype(int id) {
    const static std::vector<GemmDtype_t> kGemmDtypes{
        GemmDtype_t{CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F},
        GemmDtype_t{CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I},
        GemmDtype_t{CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F},
        GemmDtype_t{CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F},
        GemmDtype_t{CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F},
        GemmDtype_t{CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F},
        GemmDtype_t{CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F},
        GemmDtype_t{CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F},
        GemmDtype_t{CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F},
        GemmDtype_t{CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F},
    };
    return kGemmDtypes.at(id);
}

int DtypeToSize(cudaDataType_t dtype) {
    const static std::map<cudaDataType_t, int> kDtype2Size{
        {CUDA_R_8I,   1},
        {CUDA_R_16F,  2},
        {CUDA_R_32I,  4},
        {CUDA_R_32F,  4},
        {CUDA_R_64F,  8},
        {CUDA_C_8I,   2},
        {CUDA_C_32F,  8},
        {CUDA_C_64F, 16}
    };
    return kDtype2Size.at(dtype);
}

std::string OperationToString(cublasOperation_t op) {
    const static std::map<cublasOperation_t, std::string> kOperation2Str{
        ADD_KEY_AND_STR(CUBLAS_OP_N),
        ADD_KEY_AND_STR(CUBLAS_OP_T)
    };
    return kOperation2Str.at(op);
}

std::string DtypeToString(cudaDataType_t dtype) {
    const static std::map<cudaDataType_t, std::string> kDtype2Str{
        ADD_KEY_AND_STR(CUDA_R_8I),
        ADD_KEY_AND_STR(CUDA_R_16F),
        ADD_KEY_AND_STR(CUDA_R_32I),
        ADD_KEY_AND_STR(CUDA_R_32F),
        ADD_KEY_AND_STR(CUDA_R_64F),
        ADD_KEY_AND_STR(CUDA_C_8I),
        ADD_KEY_AND_STR(CUDA_C_32F),
        ADD_KEY_AND_STR(CUDA_C_64F)
    };
    return kDtype2Str.at(dtype);
}

std::string AlgoToString(cublasGemmAlgo_t algo) {
    const cublasGemmAlgo_t CUBLASLT_IMMA_ALG_ = static_cast<cublasGemmAlgo_t>(CUBLASLT_IMMA_ALG);
    const cublasGemmAlgo_t CUBLASLT_HEURISTIC_ALG_ = static_cast<cublasGemmAlgo_t>(CUBLASLT_HEURISTIC_ALG);
    const static std::map<cublasGemmAlgo_t, std::string> kAlgo2Str{
        ADD_KEY_AND_STR(CUBLASLT_IMMA_ALG_),
        ADD_KEY_AND_STR(CUBLASLT_HEURISTIC_ALG_),
        ADD_KEY_AND_STR(CUBLAS_GEMM_DEFAULT),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO0),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO1),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO2),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO3),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO4),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO5),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO6),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO7),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO8),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO9),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO10),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO11),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO12),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO13),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO14),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO15),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO16),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO17),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO18),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO19),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO20),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO21),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO22),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO23),
        ADD_KEY_AND_STR(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO0_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO1_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO2_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO3_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO4_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO5_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO6_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO7_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO8_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO9_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO10_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO11_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO12_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO13_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO14_TENSOR_OP),
        ADD_KEY_AND_STR(CUBLAS_GEMM_ALGO15_TENSOR_OP)
    };
    return kAlgo2Str.at(algo);
}

void* AllocAlphaScale(cudaDataType_t dtype) {
    void* ptr = nullptr;
    ptr = malloc(DtypeToSize(dtype));
    switch (dtype) {
        case CUDA_R_8I:
            *(reinterpret_cast<char*>(ptr)) = 1;
            break;
        case CUDA_R_16F:
            *(reinterpret_cast<half*>(ptr)) = 1.f;
            break;
        case CUDA_R_32I:
            *(reinterpret_cast<int*>(ptr)) = 1;
            break;
        case CUDA_R_32F:
            *(reinterpret_cast<float*>(ptr)) = 1.f;
            break;
        case CUDA_R_64F:
            *(reinterpret_cast<double*>(ptr)) = 1.0;
            break;
        case CUDA_C_8I:
            *(reinterpret_cast< thrust::complex<char>* >(ptr)) = 1;
            break;
        case CUDA_C_32F:
            *(reinterpret_cast< thrust::complex<float>* >(ptr)) = 1.f;
            break;
        case CUDA_C_64F:
            *(reinterpret_cast< thrust::complex<double>* >(ptr)) = 1.0;
            break;
        default:
            assert(false);
    }
    return ptr;
}

std::string Mask2Str(const std::vector<bool>& mask) {
    std::string info;
    auto count = std::count_if(mask.begin(), mask.end(),
        [](bool x) { return x; });
    if (count == mask.size()) {
        info = "all meet, ";
    }
    else {
        info = "(";
        for (auto bit : mask) {
            info += std::to_string(static_cast<int>(bit)) + ".";
        }
        info += "), ";
    }
    return info;
}

std::string Dp4aRestrictions(const GemmParam_t& param) {
    std::vector<bool> mask(2);
    mask[0] = param.lda % 4 == 0;
    mask[1] = param.ldb % 4 == 0;
    return Mask2Str(mask);
}

std::string TensorCoreRestrictions(const GemmParam_t& param) {
    // refer to https://docs.nvidia.com/cuda/cublas/#tensorop-restrictions
    std::vector<bool> mask(8);
    mask[0] = param.m % 4 == 0;
    mask[1] = param.k % 8 == 0;
    mask[2] = reinterpret_cast<intptr_t>(param.A) % 16 == 0;
    mask[3] = reinterpret_cast<intptr_t>(param.B) % 16 == 0;
    mask[4] = reinterpret_cast<intptr_t>(param.C) % 16 == 0;
    mask[5] = param.lda % (16 / DtypeToSize(param.dtype.Atype)) == 0;
    mask[6] = param.ldb % (16 / DtypeToSize(param.dtype.Btype)) == 0;
    mask[7] = param.ldc % (16 / DtypeToSize(param.dtype.Ctype)) == 0;
    return Mask2Str(mask);
}

bool SortResult (const ProfResult_t& x, const ProfResult_t& y) { 
    return (x.time < y.time); 
}

void PrintResult(const char dev_name[], const GemmParam_t& param,
    const std::vector<ProfResult_t>& results) {
    std::cout << "device, op(A), op(B), "
        "m, n, k, ComputeType, Atype, Btype, Ctype, "
        "Dp4aRestrictions(lda.ldb), TensorCoreRestrictions(m.k.A.B.C.lda.ldb.ldc), "
        "algo, time(ms), GFLOPS" << std::endl;

    std::string all_info;
    all_info = std::string(dev_name) + ", "
        + OperationToString(param.transa) + ", "
        + OperationToString(param.transb) + ", "
        + std::to_string(param.m) + ", "
        + std::to_string(param.n) + ", "
        + std::to_string(param.k) + ", "
        + DtypeToString(param.dtype.computeType) + ", "
        + DtypeToString(param.dtype.Atype) + ", "
        + DtypeToString(param.dtype.Btype) + ", "
        + DtypeToString(param.dtype.Ctype) + ", ";

    all_info += Dp4aRestrictions(param);
    all_info += TensorCoreRestrictions(param);

    float workload = (2.f * param.m * param.n * param.k) * 1e-9;

    std::vector<ProfResult_t> order = results;
    std::sort(order.begin(), order.end(), SortResult);

    for (auto result : order) {
        float gflops = workload / (result.time * 1e-3);

        std::cout << all_info << result.algo << ", " << 
            (result.time == FLT_MAX ? NAN : result.time) << ", " <<
            (result.time == FLT_MAX ? NAN : gflops) << std::endl;
    }
}

std::string TileIdToString(int id) {
    const static std::map<cublasLtMatmulTile_t, std::string> TileIdToString{
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_UNDEFINED),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_8x8),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_8x16),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_16x8),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_8x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_16x16),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x8),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_8x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_16x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x16),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x8),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_32x128),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_128x32),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x128),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_128x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x256),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_128x128),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_256x64),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_64x512),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_128x256),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_256x128),
        ADD_KEY_AND_STR(CUBLASLT_MATMUL_TILE_512x64),
    };
    return TileIdToString.at(static_cast<cublasLtMatmulTile_t>(id));
}

std::string ReductionSchemeToString(int id) {
    const static std::map<cublasLtReductionScheme_t, std::string> kRedSch2Str{
        ADD_KEY_AND_STR(CUBLASLT_REDUCTION_SCHEME_NONE),
        ADD_KEY_AND_STR(CUBLASLT_REDUCTION_SCHEME_INPLACE),
        ADD_KEY_AND_STR(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE),
        ADD_KEY_AND_STR(CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE	),
    };
    return kRedSch2Str.at(static_cast<cublasLtReductionScheme_t>(id));
}

bool SortLtResult (const LtProfResult_t& x, const LtProfResult_t& y) { 
    return (x.info.time < y.info.time); 
}

void PrintLtResult(const char dev_name[], const GemmParam_t& param,
    const std::vector<LtProfResult_t>& results) {
    std::cout << "device, op(A), op(B), "
        "m, n, k, ComputeType, Atype, Btype, Ctype, "
        "Dp4aRestrictions(lda.ldb), TensorCoreRestrictions(m.k.A.B.C.lda.ldb.ldc), "
        "algo, time(ms), GFLOPS, "
        "LtAlgoId, TileId, SpliteK, Red.Sch, Swizzle, CustomId, WorkSpaceSize, WaveCount" << std::endl;

    std::string all_info;
    all_info = std::string(dev_name) + ", "
        + OperationToString(param.transa) + ", "
        + OperationToString(param.transb) + ", "
        + std::to_string(param.m) + ", "
        + std::to_string(param.n) + ", "
        + std::to_string(param.k) + ", "
        + DtypeToString(param.dtype.computeType) + ", "
        + DtypeToString(param.dtype.Atype) + ", "
        + DtypeToString(param.dtype.Btype) + ", "
        + DtypeToString(param.dtype.Ctype) + ", ";

    all_info += Dp4aRestrictions(param);
    all_info += TensorCoreRestrictions(param);

    float workload = (2.f * param.m * param.n * param.k) * 1e-9;

    std::vector<LtProfResult_t> order = results;
    std::sort(order.begin(), order.end(), SortLtResult);

    for (auto result : order) {
        float gflops = workload / (result.info.time * 1e-3);

        std::cout << all_info << result.info.algo << ", " << 
            (result.info.time == FLT_MAX ? NAN : result.info.time) << ", " <<
            (result.info.time == FLT_MAX ? NAN : gflops) << ", ";

        if (result.attr.wave_count != 0.f) {
            std::cout << std::to_string(result.attr.algo_id) << ", " <<
                TileIdToString(result.attr.tile_id) << ", " <<
                std::to_string(result.attr.splite_k) << ", " << 
                ReductionSchemeToString(result.attr.reduction_scheme) << ", " <<
                std::to_string(result.attr.swizzle) << ", " <<
                std::to_string(result.attr.custom_option) << ", " <<
                std::to_string(result.attr.workspace_size) << ", " <<
                std::to_string(result.attr.wave_count) << std::endl;
        }
        else {
            std::cout << "NA" << std::endl;
        }
    }
}
