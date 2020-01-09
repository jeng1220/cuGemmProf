/* Copyright 2020 Jeng Bai-Cheng
*
* Permission is hereby granted, free of charge, to any person obtaining a copy of
* this software and associated documentation files (the "Software"), to deal in
* the Software without restriction, including without limitation the rights to
* use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
* of the Software, and to permit persons to whom the Software is furnished to do
* so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
*  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "cuGemmProf.h"
#include <cstdlib>
#include <cstring>
#include <cxxopts.hpp>

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
        ("algo", "assgin algorithm ID (0~23)", cxxopts::value< std::vector<int> >())
        ("tensor_algo", "assgin TensorOp algorithm ID (0~15)", cxxopts::value< std::vector<int> >())
        ("all_algo", "run all algorithms")
        ("w, workspace", "workspace size, unit: MiB", cxxopts::value<size_t>()->default_value("0"))
        ("g, debug", "dump matrices if verification is failed")
        ("r, rank", "only print n-th fast algorithms", cxxopts::value<int>()->default_value("3"))
        ("h, help", "print help");

        auto result = options.parse(argc, (char**&)argv);

        std::string type_info;
        type_info = "available combination of types:\n"
                    "ID, ComputeType, A,          B,          C\n"
                    "0,  {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F}\n"
                    "1,  {CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I}\n"
                    "2,  {CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F}\n"
                    "3,  {CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F}\n"
                    "4,  {CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F}\n"
                    "5,  {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F}\n"
                    "6,  {CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F}\n"
                    "7,  {CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F}\n"
                    "8,  {CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F}\n"
                    "9,  {CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F}\n";

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            std::cout << type_info;
            exit(EXIT_SUCCESS);
        }
        return result;
    } 
    catch (const cxxopts::OptionException& e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main (int argc, const char* argv[]) {

    auto result = Parse(argc, argv);

    auto device_id = result["d"].as<int>();
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    GemmParam_t param;    
    param.m = result["m"].as<int>();
    param.n = result["n"].as<int>();
    param.k = result["k"].as<int>();
    param.transa = result.count("ta") ? CUBLAS_OP_T : CUBLAS_OP_N;
    param.lda = (param.transa == CUBLAS_OP_N) ? param.m : param.k; 
    param.transb = result.count("tb") ? CUBLAS_OP_T : CUBLAS_OP_N;
    param.ldb = (param.transb == CUBLAS_OP_N) ? param.k : param.n;
    param.ldc = param.m;
    param.workspace_size = result["w"].as<size_t>() << 20;
    if (param.workspace_size) {
        CUDA_CHECK(cudaMalloc(&param.workspace, param.workspace_size));
    }
    else {
        param.workspace = nullptr;
    } 

    std::vector<cublasGemmAlgo_t> selected_cuda_algo{CUBLAS_GEMM_DEFAULT};
    std::vector<cublasGemmAlgo_t> selected_tensor_algo{CUBLAS_GEMM_DEFAULT_TENSOR_OP};
    auto run_all_algo = result.count("all_algo");

    if (run_all_algo) {
        selected_cuda_algo = AllCudaCoreAlgo();
        selected_tensor_algo = AllTensorCoreAlgo();
    }
    else {
        if (result.count("algo")) {
            auto algos = result["algo"].as< std::vector<int> >();
            selected_cuda_algo.clear();
            for (auto algo : algos)
                selected_cuda_algo.push_back(static_cast<cublasGemmAlgo_t>(algo + CUBLAS_GEMM_ALGO0));
        }
        if (result.count("tensor_algo")) {
            auto algos = result["tensor_algo"].as< std::vector<int> >();
            selected_tensor_algo.clear();
            for (auto algo : algos) 
                selected_tensor_algo.push_back(static_cast<cublasGemmAlgo_t>(algo + CUBLAS_GEMM_ALGO0_TENSOR_OP));
        }
    }

    auto debug = result.count("g");
    auto loop = result["l"].as<int>();
    auto rank = result["r"].as<int>();
    auto selected_dtypes = result["type"].as< std::vector<int> >();

    PrintResultTile();

    for (auto dtype_id : selected_dtypes) {

        auto gemm_dtype = GemmDtype(dtype_id);
        param.dtype = gemm_dtype;

        auto src_dtype_size = DtypeToSize(gemm_dtype.A);
        auto dst_dtype_size = DtypeToSize(gemm_dtype.C);

        void* dev_A;
        CUDA_CHECK(cudaMalloc(&dev_A, param.m * param.k * src_dtype_size));
        InitMatrix(dev_A,
            (param.transa == CUBLAS_OP_N) ? param.m : param.k,
            (param.transa == CUBLAS_OP_N) ? param.k : param.m,
            param.lda, param.dtype.A);
        void* dev_B;
        CUDA_CHECK(cudaMalloc(&dev_B, param.k * param.n * src_dtype_size));
        InitMatrix(dev_B,
            (param.transb == CUBLAS_OP_N) ? param.k : param.n,
            (param.transb == CUBLAS_OP_N) ? param.n : param.k,
            param.ldb, param.dtype.B);
        void* dev_C;
        CUDA_CHECK(cudaMalloc(&dev_C, param.m * param.n * dst_dtype_size));
        CUDA_CHECK(cudaMemset(dev_C, 0, param.m * param.n * dst_dtype_size));
        void* dev_D;
        CUDA_CHECK(cudaMalloc(&dev_D, param.m * param.n * dst_dtype_size));
        CUDA_CHECK(cudaMemset(dev_D, 0, param.m * param.n * dst_dtype_size));

        param.A = dev_A;
        param.B = dev_B;
        param.C = dev_C;
        param.D = dev_D;

        auto compute_dtype_size = DtypeToSize(gemm_dtype.compute_type);
 
        void* host_alpha;
        host_alpha = AllocAlphaScale(gemm_dtype.compute_type);

        void* host_beta;
        host_beta = malloc(compute_dtype_size);
        memset(host_beta, 0, compute_dtype_size);

        param.alpha = host_alpha;
        param.beta  = host_beta;

        NaiveGemm(
            param.transa,
            param.transb,
            param.m, param.n, param.k,
            param.A, param.dtype.A, param.lda,
            param.B, param.dtype.B, param.ldb,
            param.D, param.dtype.C, param.ldc,
            param.dtype.compute_type);

        auto results = ProfileGemm(param, selected_cuda_algo, loop, debug);
        PrintResult(param, results, rank);

        if (prop.major > 6) {
            results = ProfileGemm(param, selected_tensor_algo, loop, debug);
            PrintResult(param, results, rank);
        }

        auto lt_results = ProfileLtGemm(param, run_all_algo, loop, debug);
        PrintLtResult(param, lt_results, rank);

        CUDA_CHECK(cudaFree(dev_A));
        CUDA_CHECK(cudaFree(dev_B));
        CUDA_CHECK(cudaFree(dev_C));
        CUDA_CHECK(cudaFree(dev_D));
        free(host_alpha);
        free(host_beta);
    }
    return 0;
}
