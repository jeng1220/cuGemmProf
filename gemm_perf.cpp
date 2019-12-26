#include "cxxopts.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cfloat>

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
            __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define ADD_KEY_AND_STR(x) {x, #x}

const std::map<cublasStatus_t, std::string> kErr2Str = {
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

#define CUBLAS_API_CALL(apiFuncCall)                                            \
do {                                                                            \
    cublasStatus_t _status = apiFuncCall;                                       \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                     \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
            __FILE__, __LINE__, #apiFuncCall, kErr2Str.at(_status).c_str());    \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

const std::map<cudaDataType_t, std::string> kDtype2Str = {
    ADD_KEY_AND_STR(CUDA_R_8I),
    ADD_KEY_AND_STR(CUDA_R_16F),
    ADD_KEY_AND_STR(CUDA_R_32I),
    ADD_KEY_AND_STR(CUDA_R_32F),
    ADD_KEY_AND_STR(CUDA_R_64F),
    ADD_KEY_AND_STR(CUDA_C_8I),
    ADD_KEY_AND_STR(CUDA_C_32F),
    ADD_KEY_AND_STR(CUDA_C_64F)
};

const std::map<cublasOperation_t, std::string> kOperation2Str = {
    ADD_KEY_AND_STR(CUBLAS_OP_N),
    ADD_KEY_AND_STR(CUBLAS_OP_T)
};

const std::map<cublasGemmAlgo_t, std::string> kAlgo2Str = {
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

struct Dtypes_t {
    cudaDataType_t computeType;
    cudaDataType_t Atype;
    cudaDataType_t Btype;
    cudaDataType_t Ctype;
};

struct Param_t {
    cublasHandle_t handle;
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
    cublasGemmAlgo_t algo;
    Dtypes_t dtype;
};

cxxopts::ParseResult Parse(int argc, const char* argv[]) {
  try {
    cxxopts::Options options(argv[0], "GEMM testing");

    options.positional_help("[optional args]").show_positional_help();

    options.add_options()
    ("m", "m dimension", cxxopts::value<int>()->default_value("32"))
    ("n", "n dimension", cxxopts::value<int>()->default_value("32"))
    ("k", "k dimension", cxxopts::value<int>()->default_value("32"))
    ("d", "device ID", cxxopts::value<int>()->default_value("0"))
    ("l", "loop", cxxopts::value<int>()->default_value("1"))
    ("ta", "set A to CUBLAS_OP_T, else CUBLAS_OP_N")
    ("tb", "set B to CUBLAS_OP_T, else CUBLAS_OP_N")
    ("type", "slect combination of types",
        cxxopts::value< std::vector<int> >()->default_value("5"))
    ("help", "print help");
    
    auto result = options.parse(argc, (char**&)argv);

    std::stringstream type_info;
    type_info << "available combination of types:\n";
    type_info << "ID, ComputeType, Atype,      Btype,      Ctype\n";
    type_info << "0,  {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F}\n";
    type_info << "1,  {CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I}\n";
    type_info << "2,  {CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F}\n";
    type_info << "3,  {CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F}\n";
    type_info << "4,  {CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F}\n";
    type_info << "5,  {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F}\n";
    type_info << "6,  {CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F}\n";
    type_info << "7,  {CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F}\n";
    type_info << "8,  {CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F}\n";
    type_info << std::endl;

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::cout << type_info.str();
        exit(EXIT_SUCCESS);
    }

    return result;

  } catch (const cxxopts::OptionException& e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

struct Result_t {
    cublasGemmAlgo_t algo;
    float time;
    float gflops;
    friend std::ostream& operator<<(std::ostream& os, const Result_t& dt);
};

std::ostream& operator<<(std::ostream& os, const Result_t& dt) {
    os << kAlgo2Str.at(dt.algo) << ", " << dt.time << ", " << dt.gflops;
    return os;
}

bool SortResult (Result_t i,Result_t j) { return (i.time < j.time); }

Result_t ProfileGemm(const Param_t& param, int loop) {

    cudaEvent_t start;
    cudaEvent_t end;
    cudaError_t err;
    cublasStatus_t ret;

    RUNTIME_API_CALL(cudaEventCreate(&start));
    RUNTIME_API_CALL(cudaEventCreate(&end));
    float time = 0.f;
    bool fault = false;

    RUNTIME_API_CALL(cudaEventRecord(start));
    for (int i = 0; i < loop; ++i) {
        ret = cublasGemmEx(param.handle,
                           param.transa, param.transb,
                           param.m, param.n, param.k,
                           param.alpha, param.A, param.dtype.Atype, param.lda,
                           param.B, param.dtype.Btype, param.ldb, param.beta,
                           param.C, param.dtype.Ctype, param.ldc,
                           param.dtype.computeType, param.algo);
        if (ret != CUBLAS_STATUS_SUCCESS) {
            fault = true;
            if (ret != CUBLAS_STATUS_NOT_SUPPORTED) {
                CUBLAS_API_CALL(ret);
            }
            break;
        }
    }
    RUNTIME_API_CALL(cudaEventRecord(end));
    RUNTIME_API_CALL(cudaEventSynchronize(end));
    RUNTIME_API_CALL(cudaEventElapsedTime(&time, start, end));
    RUNTIME_API_CALL(cudaEventDestroy(start));
    RUNTIME_API_CALL(cudaEventDestroy(end));

    if (fault) { time = FLT_MAX; }
    time /= loop;
    float workload = (2.f * param.m * param.n * param.k) * 1e-9;
    float gflops = (fault) ? NAN : workload / (time * 1e-3);

    Result_t result{param.algo, time, gflops};
    return result;
}

int main (int argc, const char* argv[]) {

    auto result = Parse(argc, argv);

    const Dtypes_t dtype_config[] = {
        Dtypes_t{CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F},
        Dtypes_t{CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I},
        Dtypes_t{CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F},
        Dtypes_t{CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F},
        Dtypes_t{CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F},
        Dtypes_t{CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F},
        Dtypes_t{CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F},
        Dtypes_t{CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F},
        Dtypes_t{CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F},
    };

    const cublasGemmAlgo_t cuda_algo_config[] = {
        CUBLAS_GEMM_DEFAULT,
        CUBLAS_GEMM_ALGO0,
        CUBLAS_GEMM_ALGO1,
        CUBLAS_GEMM_ALGO2,
        CUBLAS_GEMM_ALGO3,
        CUBLAS_GEMM_ALGO4,
        CUBLAS_GEMM_ALGO5,
        CUBLAS_GEMM_ALGO6,
        CUBLAS_GEMM_ALGO7,
        CUBLAS_GEMM_ALGO8,
        CUBLAS_GEMM_ALGO9,
        CUBLAS_GEMM_ALGO10,
        CUBLAS_GEMM_ALGO11,
        CUBLAS_GEMM_ALGO12,
        CUBLAS_GEMM_ALGO13,
        CUBLAS_GEMM_ALGO14,
        CUBLAS_GEMM_ALGO15,
        CUBLAS_GEMM_ALGO16,
        CUBLAS_GEMM_ALGO17,
        CUBLAS_GEMM_ALGO18,
        CUBLAS_GEMM_ALGO19,
        CUBLAS_GEMM_ALGO20,
        CUBLAS_GEMM_ALGO21,
        CUBLAS_GEMM_ALGO22,
        CUBLAS_GEMM_ALGO23
    };

    const cublasGemmAlgo_t tensor_algo_config[] = {
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        CUBLAS_GEMM_ALGO0_TENSOR_OP,
        CUBLAS_GEMM_ALGO1_TENSOR_OP,
        CUBLAS_GEMM_ALGO2_TENSOR_OP,
        CUBLAS_GEMM_ALGO3_TENSOR_OP,
        CUBLAS_GEMM_ALGO4_TENSOR_OP,
        CUBLAS_GEMM_ALGO5_TENSOR_OP,
        CUBLAS_GEMM_ALGO6_TENSOR_OP,
        CUBLAS_GEMM_ALGO7_TENSOR_OP,
        CUBLAS_GEMM_ALGO8_TENSOR_OP,
        CUBLAS_GEMM_ALGO9_TENSOR_OP,
        CUBLAS_GEMM_ALGO10_TENSOR_OP,
        CUBLAS_GEMM_ALGO11_TENSOR_OP,
        CUBLAS_GEMM_ALGO12_TENSOR_OP,
        CUBLAS_GEMM_ALGO13_TENSOR_OP,
        CUBLAS_GEMM_ALGO14_TENSOR_OP,
        CUBLAS_GEMM_ALGO15_TENSOR_OP,
    };

    const std::map<cudaDataType_t, int> dtype2size = {
        {CUDA_R_8I,   1},
        {CUDA_R_16F,  2},
        {CUDA_R_32I,  4},
        {CUDA_R_32F,  4},
        {CUDA_R_64F,  8},
        {CUDA_C_8I,   2},
        {CUDA_C_32F,  8},
        {CUDA_C_64F, 16}
    };

    RUNTIME_API_CALL(cudaSetDevice(result["d"].as<int>()));

    Param_t param;
    CUBLAS_API_CALL(cublasCreate(&param.handle));
    
    param.m = result["m"].as<int>();
    param.n = result["n"].as<int>();
    param.k = result["k"].as<int>();
    param.transa = result.count("ta") ? CUBLAS_OP_T : CUBLAS_OP_N;
    param.lda = (param.transa == CUBLAS_OP_N) ? param.m : param.k; 
    param.transb = result.count("tb") ? CUBLAS_OP_T : CUBLAS_OP_N;
    param.ldb = (param.transb == CUBLAS_OP_N) ? param.k : param.n;
    param.ldc = param.m;

    std::cout << "op(A), op(B), m, n, k, Atype, Btype, Ctype, "
        "ComputeType, time(ms), GFLOPS" << std::endl;

    std::stringstream dims_ss;
    dims_ss << kOperation2Str.at(param.transa) << ", "
            << kOperation2Str.at(param.transb) << ", "
            << param.m << ", "
            << param.n << ", "
            << param.k << ", ";

    auto selected_dtypes = result["type"].as< std::vector<int> >();

    for (auto dtype_id : selected_dtypes) {

        auto dtypes = dtype_config[dtype_id];
        param.dtype = dtypes;

        std::stringstream dtype_ss;
        dtype_ss << kDtype2Str.at(param.dtype.Atype) << ", "
           << kDtype2Str.at(param.dtype.Btype) << ", "
           << kDtype2Str.at(param.dtype.Ctype) << ", "
           << kDtype2Str.at(param.dtype.computeType) << ", ";

        auto src_dtype_size = dtype2size.at(dtypes.Atype);
        auto dst_dtype_size = dtype2size.at(dtypes.Ctype);

        void* dev_A;
        RUNTIME_API_CALL(cudaMalloc(&dev_A, param.m * param.k * src_dtype_size));
        void* dev_B;
        RUNTIME_API_CALL(cudaMalloc(&dev_B, param.k * param.n * src_dtype_size));
        void* dev_C;
        RUNTIME_API_CALL(cudaMalloc(&dev_C, param.m * param.n * dst_dtype_size));

        param.A = dev_A;
        param.B = dev_B;
        param.C = dev_C;

        auto compute_dtype_size = dtype2size.at(dtypes.computeType);
 
        char* host_alpha;
        host_alpha = (char*)malloc(compute_dtype_size);
        memset(host_alpha, 0, compute_dtype_size);
        host_alpha[0] = 1; 

        char* host_beta;
        host_beta = (char*)malloc(compute_dtype_size);
        memset(host_beta, 0, compute_dtype_size);

        param.alpha = (void*)host_alpha;
        param.beta  = (void*)host_beta;

        auto loop = result["l"].as<int>();

        std::vector<Result_t> results;
        for (auto algo : cuda_algo_config) {
            
            param.algo = algo;
            auto result = ProfileGemm(param, loop);
            results.push_back(result);
        }
        std::cout << dims_ss.str() << dtype_ss.str() << results[0] << std::endl;
        std::sort(results.begin(), results.end(), SortResult);
        std::cout << dims_ss.str() << dtype_ss.str() << results[0] << std::endl;

        results.clear();
        for (auto algo : tensor_algo_config) {
            
            param.algo = algo;
            auto result = ProfileGemm(param, loop);
            results.push_back(result);
        }
        std::cout << dims_ss.str() << dtype_ss.str() << results[0] << std::endl;
        std::sort(results.begin(), results.end(), SortResult);
        std::cout << dims_ss.str() << dtype_ss.str()  << results[0] << std::endl;

        RUNTIME_API_CALL(cudaFree(dev_A));
        RUNTIME_API_CALL(cudaFree(dev_B));
        RUNTIME_API_CALL(cudaFree(dev_C));
        free(host_alpha);
        free(host_beta);
    }

    CUBLAS_API_CALL(cublasDestroy(param.handle));
    return 0;
}
