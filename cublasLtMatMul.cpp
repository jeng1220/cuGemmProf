#include "cublasLtMatMul.h"
#include "macro.h"
#include "verify.h"
#include <cublasLt.h>
#include <vector>
#include <cfloat>

int RoundOff(int v, int d) {
   return (v + d - 1) / d * d;
}

struct AlgoAttr_t {
    int splite_k_support;
    int reduction_scheme_mask;
    int swizzling_support;
    int custom_option_max;
    int epilogue_mask;
    std::vector<int> tile_ids;
    AlgoAttr_t(cublasLtMatmulAlgo_t algo) {
        CUBLAS_API_CALL(cublasLtMatmulAlgoCapGetAttribute(&algo,
            CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splite_k_support, sizeof(int), nullptr));
        CUBLAS_API_CALL(cublasLtMatmulAlgoCapGetAttribute(&algo,
            CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &reduction_scheme_mask, sizeof(int), nullptr));
        CUBLAS_API_CALL(cublasLtMatmulAlgoCapGetAttribute(&algo,
            CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzling_support, sizeof(int), nullptr));
        CUBLAS_API_CALL(cublasLtMatmulAlgoCapGetAttribute(&algo,
            CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_option_max, sizeof(int), nullptr));
        CUBLAS_API_CALL(cublasLtMatmulAlgoCapGetAttribute(&algo,
            CUBLASLT_ALGO_CAP_EPILOGUE_MASK, &epilogue_mask, sizeof(int), nullptr));

        size_t size_in_bytes = 0;
        CUBLAS_API_CALL(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS,
            nullptr, 0, &size_in_bytes));

        int nb_tiles = static_cast<int>(size_in_bytes / sizeof(int));
        tile_ids;
        if (nb_tiles > 0) {
            tile_ids.resize(nb_tiles);
            CUBLAS_API_CALL(cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tile_ids.data(), size_in_bytes, nullptr));
        }
        else {
            tile_ids.resize(1);
            tile_ids[0] = static_cast<int>(CUBLASLT_MATMUL_TILE_UNDEFINED);
        }
    }
};

struct CublasLtParam_t {
    cublasLtMatrixLayout_t A_desc;
    cublasLtMatrixLayout_t B_desc;
    cublasLtMatrixLayout_t C_desc;
    void* A;
    void* B;
    void* C;
    cublasLtMatmulAlgo_t* algo;
    void* workspace;
    size_t workspace_size;
};

struct ImmaParam_t {
    bool use_imma;
    cublasLtMatrixTransformDesc_t transform_desc;
    cublasLtMatrixLayout_t origin_desc;
};

Result_t RunMatMul(cublasLtHandle_t handle, cublasLtMatmulDesc_t op_desc,
    const Param_t& param, const CublasLtParam_t& lt_param, 
    const ImmaParam_t& imma_param, int loop, bool debug,
    std::string algo_name)
{
    cublasStatus_t ret;
    cudaEvent_t start;
    cudaEvent_t end;
    bool fault = false;
    RUNTIME_API_CALL(cudaEventCreate(&start));
    RUNTIME_API_CALL(cudaEventCreate(&end));
    RUNTIME_API_CALL(cudaEventRecord(start));
    for (int i = 0; i < loop; ++i) {

        ret = cublasLtMatmul(handle, op_desc, 
                             param.alpha, lt_param.A, lt_param.A_desc,
                             lt_param.B, lt_param.B_desc, param.beta,
                             lt_param.C, lt_param.C_desc,
                             lt_param.C, lt_param.C_desc,
                             lt_param.algo,
                             lt_param.workspace, lt_param.workspace_size, 0);
        if (ret != CUBLAS_STATUS_SUCCESS) {
            fault = true;
            if (debug) {
                std::cerr << "cublasLtMatmul" << ", " << 
                    ", " << cublasGetErrorString(ret) << std::endl;
            }
            break;
        }
    }
    RUNTIME_API_CALL(cudaEventRecord(end));   
    RUNTIME_API_CALL(cudaEventSynchronize(end));

    if (imma_param.use_imma && !fault) {
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_API_CALL(cublasLtMatrixTransform(handle, imma_param.transform_desc,
            &alpha, lt_param.C, lt_param.C_desc,
            &beta, nullptr, nullptr, param.C, imma_param.origin_desc, 0));
        RUNTIME_API_CALL(cudaStreamSynchronize(0));
    }

    if (!fault) {
        fault = !Verify(param.C, param.D, param.m * param.n, param.dtype.Ctype);
        if (fault && debug) {
            std::cerr << "cublasLtMatmul verification failed" << std::endl;
            PrintMatrix(param.A, param.m, param.k, param.lda, param.dtype.Atype);
            PrintMatrix(param.B, param.k, param.n, param.ldb, param.dtype.Btype);
            PrintMatrix(param.C, param.m, param.n, param.ldc, param.dtype.Ctype);
            PrintMatrix(param.D, param.m, param.n, param.ldc, param.dtype.Ctype);
        }
    }
    RUNTIME_API_CALL(cudaMemset(param.C, 0, param.m * param.n * Dtype2Size(param.dtype.Ctype)));

    float time = 0.f;
    RUNTIME_API_CALL(cudaEventElapsedTime(&time, start, end));
    RUNTIME_API_CALL(cudaEventDestroy(start));
    RUNTIME_API_CALL(cudaEventDestroy(end));

    time = fault ? FLT_MAX : (time / loop);
    return Result_t{algo_name, time};
}

std::vector<Result_t> ProfileAllGemmAlgoLt(cublasLtHandle_t handle, cublasLtMatmulDesc_t op_desc,
    const Param_t& param, CublasLtParam_t& lt_param, const ImmaParam_t& imma_param, int loop, bool debug) {

    const int max_algos = 40;
    std::vector<int> algo_ids(max_algos);
    int nb_algo_id = 0;

    CUBLAS_API_CALL(cublasLtMatmulAlgoGetIds(
        handle, param.dtype.computeType, param.dtype.computeType,
        param.dtype.Atype, param.dtype.Btype, param.dtype.Ctype, param.dtype.Ctype,
        max_algos, algo_ids.data(), &nb_algo_id));
    algo_ids.resize(nb_algo_id);

    std::vector<Result_t> results;
    const int max_combine_option = 6000;
    int combine_count = 0;

    for (int idx = 0; (idx < nb_algo_id) && (combine_count < max_combine_option); idx++) {
        cublasLtMatmulAlgo_t algo;
        CUBLAS_API_CALL(cublasLtMatmulAlgoInit(handle, 
            param.dtype.computeType, param.dtype.computeType, 
            param.dtype.Atype, param.dtype.Btype, param.dtype.Ctype, param.dtype.Ctype,
            algo_ids[idx], &algo));
 
        AlgoAttr_t algo_attr(algo);
 
        for (auto tile_id : algo_attr.tile_ids) {
 
            CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(int)));
 
            for (int c = 0; c <= algo_attr.custom_option_max; ++c) {
                CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &c, sizeof(int)));
 
                // is it really needed ?
                for (int s = 0; s <= algo_attr.swizzling_support; ++s) {
 
                    CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &s, sizeof(int)));
 
                    const static std::vector<int> num_split_k_option{1, 2, 3, 4};
                    for (auto splite_k : num_split_k_option) {

                        if (splite_k > 1 && !algo_attr.splite_k_support) continue;

                        CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(&algo,
                            CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splite_k, sizeof(int)));
 
                        const static std::vector<cublasLtReductionScheme_t> reductions{
                            CUBLASLT_REDUCTION_SCHEME_NONE,
                            CUBLASLT_REDUCTION_SCHEME_INPLACE,
                            CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
                            CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE};

                        for (auto reduction : reductions) {
 
                            if (splite_k == 1 && reduction != CUBLASLT_REDUCTION_SCHEME_NONE)
                                continue;

                            if (splite_k > 1 && !(reduction & algo_attr.reduction_scheme_mask))
                                continue;

                            CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(int)));
 
                            cublasLtMatmulHeuristicResult_t heur_result;
                            cublasStatus_t ret;
                            ret = (cublasLtMatmulAlgoCheck(handle, op_desc,
                                lt_param.A_desc, lt_param.B_desc, lt_param.C_desc, lt_param.C_desc,
                                &algo, &heur_result));

                            if (ret == CUBLAS_STATUS_SUCCESS &&
                                heur_result.state == CUBLAS_STATUS_SUCCESS &&
                                heur_result.workspaceSize <= lt_param.workspace_size) {
                                lt_param.algo = &algo;

                                results.push_back(RunMatMul(handle, op_desc, param,
                                    lt_param, imma_param, loop, debug, "CUBLASLT_DEFAULT_ALG"));
                                combine_count++;
                            }
                            else if (debug) {
                                std::cerr << "cublasLtMatmulAlgoCheck, " << cublasGetErrorString(ret) << 
                                    ", needed workspace size, " << heur_result.workspaceSize << 
                                    ", current workspace size, " << lt_param.workspace_size << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
    return results;
}

std::vector<Result_t> ProfileGemmLt(const Param_t& param, int loop, bool debug) {
    cublasLtHandle_t handle;
    CUBLAS_API_CALL(cublasLtCreate(&handle));

    cublasLtMatmulDesc_t op_desc;
    CUBLAS_API_CALL(cublasLtMatmulDescCreate(&op_desc, param.dtype.computeType));
    CUBLAS_API_CALL(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &param.transa, sizeof(cublasOperation_t)));
    CUBLAS_API_CALL(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &param.transb, sizeof(cublasOperation_t)));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CUBLAS_API_CALL(cublasLtMatrixLayoutCreate(
       &Adesc, param.dtype.Atype,
       param.transa == CUBLAS_OP_N ? param.m : param.k,
       param.transa == CUBLAS_OP_N ? param.k : param.m,
       param.lda));
    CUBLAS_API_CALL(cublasLtMatrixLayoutCreate(
       &Bdesc, param.dtype.Btype,
       param.transb == CUBLAS_OP_N ? param.k : param.n,
       param.transb == CUBLAS_OP_N ? param.n : param.k,
       param.ldb));
    CUBLAS_API_CALL(cublasLtMatrixLayoutCreate(
        &Cdesc, param.dtype.Ctype, param.m, param.n, param.ldc));

    CublasLtParam_t lt_param{Adesc, Bdesc, Cdesc,
        param.A, param.B, param.C, 
        nullptr, param.workspace, param.workspace_size};

    bool use_imma = param.dtype.computeType == CUDA_R_32I &&
                    param.transa == CUBLAS_OP_N &&
                    param.transb == CUBLAS_OP_T;

    ImmaParam_t imma_param{false};
    cublasLtMatrixTransformDesc_t transformDesc;
    if (use_imma) {
        void* Atransform;
        void* Btransform;
        void* Ctransform;
        int ldatransform = 32 * param.m;
        int ldbtransform = 32 * RoundOff(param.n, 8);
        int ldctransform = 32 * param.m;
        RUNTIME_API_CALL(cudaMalloc(&Atransform,
            Dtype2Size(param.dtype.Atype) * RoundOff(param.k, 32) / 32 * ldatransform));
        RUNTIME_API_CALL(cudaMalloc(&Btransform,
            Dtype2Size(param.dtype.Btype) * RoundOff(param.k, 32) / 32 * ldbtransform));
        RUNTIME_API_CALL(cudaMalloc(&Ctransform,
            Dtype2Size(param.dtype.Ctype) * RoundOff(param.n, 32) / 32 * ldctransform));

        cublasLtMatrixLayout_t AtransformDesc;
        cublasLtMatrixLayout_t BtransformDesc;
        cublasLtMatrixLayout_t CtransformDesc;
        auto order_COL32 = CUBLASLT_ORDER_COL32;
        CUBLAS_API_CALL(cublasLtMatrixLayoutCreate(&AtransformDesc, param.dtype.Atype,
            param.m, param.k, ldatransform));
        CUBLAS_API_CALL(cublasLtMatrixLayoutSetAttribute(AtransformDesc,
            CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(cublasLtOrder_t)));
        auto order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
        CUBLAS_API_CALL(cublasLtMatrixLayoutCreate(&BtransformDesc, param.dtype.Btype,
            param.n, param.k, ldbtransform));
        CUBLAS_API_CALL(cublasLtMatrixLayoutSetAttribute(BtransformDesc,
            CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(cublasLtOrder_t)));
        CUBLAS_API_CALL(cublasLtMatrixLayoutCreate(&CtransformDesc, param.dtype.Ctype,
            param.m, param.n, ldctransform));
        CUBLAS_API_CALL(cublasLtMatrixLayoutSetAttribute(CtransformDesc,
            CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(cublasLtOrder_t)));

        CUBLAS_API_CALL(cublasLtMatrixTransformDescCreate(&transformDesc,
            CUDA_R_32F));
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_API_CALL(cublasLtMatrixTransform(handle, transformDesc,
            &alpha, param.A, Adesc,
            &beta, nullptr, nullptr, Atransform, AtransformDesc, 0));

        CUBLAS_API_CALL(cublasLtMatrixTransform(handle, transformDesc,
            &alpha, param.B, Bdesc,
            &beta, nullptr, nullptr, Btransform, BtransformDesc, 0));
        RUNTIME_API_CALL(cudaStreamSynchronize(0));

        lt_param.A_desc = AtransformDesc;
        lt_param.B_desc = BtransformDesc;
        lt_param.C_desc = CtransformDesc;
        lt_param.A = Atransform;
        lt_param.B = Btransform;
        lt_param.C = Ctransform;

        imma_param.use_imma = true;
        imma_param.transform_desc = transformDesc;
        imma_param.origin_desc = Cdesc;
    }

    std::vector<Result_t> results;
    results = ProfileAllGemmAlgoLt(handle, op_desc, param, lt_param, imma_param, loop, debug);

    cublasLtMatmulHeuristicResult_t result{};
    std::string algo_name{"CUBLASLT_DEFAULT_ALG"};

    if (use_imma) {
        algo_name = "CUBLASLT_IMMA_ALG";
    }
    else {
        // optional, use heuristic approach to select best GEMM kernel,
        // but not support IMMA currently
        cublasLtMatmulPreference_t preference;
        CUBLAS_API_CALL(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_API_CALL(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &lt_param.workspace_size, sizeof(size_t)));

        int nb_result = 0;
        auto ret = cublasLtMatmulAlgoGetHeuristic(
            handle, op_desc, Adesc, Bdesc, Cdesc, Cdesc, preference,
            1, &result, &nb_result);
        if (nb_result > 0) {
            algo_name = "CUBLASLT_HEURISTIC_ALG";
            lt_param.algo = &result.algo;
        }
        else if (debug) {
            std::cerr << "cublasLtMatmulAlgoGetHeuristic, " << cublasGetErrorString(ret) << std::endl;
        }
        CUBLAS_API_CALL(cublasLtMatmulPreferenceDestroy(preference));
    }

    results.push_back(RunMatMul(handle, op_desc, param, lt_param, imma_param,
        loop, debug, algo_name));

    if (use_imma) {
        RUNTIME_API_CALL(cudaFree(lt_param.A));
        RUNTIME_API_CALL(cudaFree(lt_param.B));
        RUNTIME_API_CALL(cudaFree(lt_param.C));
        CUBLAS_API_CALL(cublasLtMatrixLayoutDestroy(lt_param.A_desc));
        CUBLAS_API_CALL(cublasLtMatrixLayoutDestroy(lt_param.B_desc));
        CUBLAS_API_CALL(cublasLtMatrixLayoutDestroy(lt_param.C_desc));
        CUBLAS_API_CALL(cublasLtMatrixTransformDescDestroy(transformDesc));
    }

    CUBLAS_API_CALL(cublasLtMatrixLayoutDestroy(Cdesc));
    CUBLAS_API_CALL(cublasLtMatrixLayoutDestroy(Bdesc));
    CUBLAS_API_CALL(cublasLtMatrixLayoutDestroy(Adesc));
    CUBLAS_API_CALL(cublasLtMatmulDescDestroy(op_desc));
    CUBLAS_API_CALL(cublasLtDestroy(handle));

    return results;
}
