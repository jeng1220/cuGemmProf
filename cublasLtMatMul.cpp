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

struct CublasLtParam_t
{
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

void ProfileAllGemmAlgoLt(cublasLtHandle_t handle, cublasLtMatmulDesc_t op_desc, const Param_t& param, const CublasLtParam_t& lt_param, int loop) {
    const int max_algos = 40;
    std::vector<int> algo_ids(max_algos);
    int nb_algo_id = 0;

    CUBLAS_API_CALL(cublasLtMatmulAlgoGetIds(
        handle, param.dtype.computeType, param.dtype.computeType,
        param.dtype.Atype, param.dtype.Btype, param.dtype.Ctype, param.dtype.Ctype,
        max_algos, algo_ids.data(), &nb_algo_id));
    algo_ids.resize(nb_algo_id);

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
 
                    const static std::vector<int> num_split_k_option{0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 32};
                    for (auto splite_k : num_split_k_option) {
                        CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(&algo,
                            CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splite_k, sizeof(int)));
 
                        const static std::vector<cublasLtReductionScheme_t> reductions{
                            /*CUBLASLT_REDUCTION_SCHEME_NONE,*/
                            CUBLASLT_REDUCTION_SCHEME_INPLACE,
                            CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
                            CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE,
                            CUBLASLT_REDUCTION_SCHEME_MASK};
                        for (auto reduction : reductions) {
 
                            if (reduction & algo_attr.reduction_scheme_mask) {
                                CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(
                                    &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(int)));
 
                                cublasLtMatmulHeuristicResult_t heur_result;
                                CUBLAS_API_CALL(cublasLtMatmulAlgoCheck(handle, op_desc,
                                    lt_param.A_desc, lt_param.B_desc, lt_param.C_desc, lt_param.C_desc,
                                    &algo, &heur_result));

                                if (heur_result.workspaceSize <= lt_param.workspace_size) {
                                    // call cublasLtMatmul(..., algo, ...); here
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector<Result_t> ProfileGemmLt(const Param_t& param, int loop) {
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
        float transformAlpha = 1.0f;
        float transformBeta = 0.0f;
        CUBLAS_API_CALL(cublasLtMatrixTransform(handle, transformDesc,
            &transformAlpha, param.A, Adesc,
            &transformBeta, nullptr, nullptr, Atransform, AtransformDesc, 0));

        CUBLAS_API_CALL(cublasLtMatrixTransform(handle, transformDesc,
            &transformAlpha, param.B, Bdesc,
            &transformBeta, nullptr, nullptr, Btransform, BtransformDesc, 0));
        RUNTIME_API_CALL(cudaStreamSynchronize(0));

        lt_param.A_desc = AtransformDesc;
        lt_param.B_desc = BtransformDesc;
        lt_param.C_desc = CtransformDesc;
        lt_param.A = Atransform;
        lt_param.B = Btransform;
        lt_param.C = Ctransform;
    }

    // optional, use heuristic approach to select best GEMM kernel,
    // but not support IMMA currently
    bool fault = false;
    cublasLtMatmulHeuristicResult_t result{};
    if (!use_imma) {
        cublasLtMatmulPreference_t preference;
        CUBLAS_API_CALL(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_API_CALL(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &lt_param.workspace_size, sizeof(size_t)));

        //ProfileAllGemmAlgoLt(handle, op_desc, param, lt_param, loop);

        int nb_result = 0;
        cublasStatus_t ret = cublasLtMatmulAlgoGetHeuristic(
            handle, op_desc, Adesc, Bdesc, Cdesc, Cdesc, preference,
            1, &result, &nb_result);
        if (nb_result > 0) {
            lt_param.algo = &result.algo;
        }
        else {
            fault = true;
            std::cerr << cublasGetErrorString(ret) << std::endl;
        }
        CUBLAS_API_CALL(cublasLtMatmulPreferenceDestroy(preference));
    }

    cudaEvent_t start;
    cudaEvent_t end;
    RUNTIME_API_CALL(cudaEventCreate(&start));
    RUNTIME_API_CALL(cudaEventCreate(&end));
    RUNTIME_API_CALL(cudaEventRecord(start));
    for (int i = 0; i < loop && !fault; ++i) {

        CUBLAS_API_CALL(cublasLtMatmul(handle, op_desc, 
                                       param.alpha, lt_param.A, lt_param.A_desc,
                                       lt_param.B, lt_param.B_desc, param.beta,
                                       lt_param.C, lt_param.C_desc,
                                       lt_param.C, lt_param.C_desc,
                                       lt_param.algo,
                                       lt_param.workspace, lt_param.workspace_size, 0));
    }
    RUNTIME_API_CALL(cudaEventRecord(end));   
    RUNTIME_API_CALL(cudaEventSynchronize(end));

    if (use_imma) {
        float transformAlpha = 1.0f;
        float transformBeta = 0.0f;
        CUBLAS_API_CALL(cublasLtMatrixTransform(handle, transformDesc,
            &transformAlpha, lt_param.C, lt_param.C_desc,
            &transformBeta, nullptr, nullptr, param.C, Cdesc, 0));
        RUNTIME_API_CALL(cudaStreamSynchronize(0));
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

    if (!fault) {
        fault = !Verify(param.C, param.D, param.m * param.n, param.dtype.Ctype);
        if (fault) {
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
    auto algo = static_cast<cublasGemmAlgo_t>(use_imma ? CUBLASLT_IMMA_ALG : CUBLASLT_HEURISTIC_ALG);
    std::vector<Result_t> results;
    results.push_back(Result_t{Algo2Str(algo), time});

    return results;
}