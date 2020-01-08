#include "cublasLtMatMul.h"
#include "macro.h"
#include "verify.h"
#include <cublasLt.h>
#include <vector>
#include <cassert>
#include <cfloat>
#include <cstring>

int RoundOff(int v, int d) {
   return (v + d - 1) / d * d;
}

struct cublasLtMatrix_t {
    void* ptr;
    cublasLtMatrixLayout_t desc;
    bool own;
};

struct cublasLtMatrixAttr_t {
    long w;
    long h;
    long ld;
    cudaDataType_t dtype;
};

cublasLtMatrixAttr_t LtMatrixAttr(cublasLtMatrix_t mat) {
    cublasLtMatrixAttr_t attr;
    
    CUBLAS_API_CALL(cublasLtMatrixLayoutGetAttribute(
	    mat.desc, CUBLASLT_MATRIX_LAYOUT_ROWS,
        &attr.h, sizeof(long), nullptr));

    CUBLAS_API_CALL(cublasLtMatrixLayoutGetAttribute(
	    mat.desc, CUBLASLT_MATRIX_LAYOUT_COLS,
        &attr.w, sizeof(long), nullptr));

    CUBLAS_API_CALL(cublasLtMatrixLayoutGetAttribute(
	    mat.desc, CUBLASLT_MATRIX_LAYOUT_LD,
        &attr.ld, sizeof(long), nullptr));

    int dtype;
    CUBLAS_API_CALL(cublasLtMatrixLayoutGetAttribute(
	    mat.desc, CUBLASLT_MATRIX_LAYOUT_TYPE,
        &dtype, sizeof(int), nullptr));
    attr.dtype = static_cast<cublasDataType_t>(dtype);

    return attr;
}

cublasDataType_t LtMatrixDtype(cublasLtMatrix_t mat) {
    return LtMatrixAttr(mat).dtype;
}

size_t LtMatrixCount(cublasLtMatrix_t mat) {
    auto attr = LtMatrixAttr(mat);
    return attr.ld * attr.h;
}

size_t LtMatrixSizeInBytes(cublasLtMatrix_t mat) {
    auto attr = LtMatrixAttr(mat);
    return attr.ld * attr.h * Dtype2Size(attr.dtype);
}

cublasLtMatrix_t CreateLtMatrix(const void* ptr, int w, int h, int ld,
    cublasDataType_t dtype) {

    cublasLtMatrix_t mat;
    mat.own = false;
    mat.ptr = const_cast<void*>(ptr);
    CUBLAS_API_CALL(cublasLtMatrixLayoutCreate(
       &mat.desc, dtype, w, h, ld));
    
    return mat;
}

cublasLtMatrix_t CreateTransformLtMatrix(int w, int h,
    cublasDataType_t dtype, cublasLtOrder_t order) {

    cublasLtMatrix_t mat;
    mat.own = true;

    if (order == CUBLASLT_ORDER_COL32) {
        int ld = 32 * w;
        RUNTIME_API_CALL(cudaMalloc(&mat.ptr,
            Dtype2Size(dtype) * RoundOff(h, 32) / 32 * ld));
        CUBLAS_API_CALL(cublasLtMatrixLayoutCreate(&mat.desc, dtype,
                w, h, ld));
    }
    else if (order == CUBLASLT_ORDER_COL4_4R2_8C) {
        int ld = 32 * RoundOff(h, 8);
        RUNTIME_API_CALL(cudaMalloc(&mat.ptr,
            Dtype2Size(dtype) * RoundOff(w, 32) / 32 * ld));
        CUBLAS_API_CALL(cublasLtMatrixLayoutCreate(&mat.desc, dtype,
                h, w, ld));
    }
    else {
        assert(-1);
    }
    CUBLAS_API_CALL(cublasLtMatrixLayoutSetAttribute(mat.desc,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(cublasLtOrder_t)));
    return mat;
}

void DestroyLtMatrix(cublasLtMatrix_t mat) {
    if (mat.own) RUNTIME_API_CALL(cudaFree(mat.ptr));
    CUBLAS_API_CALL(cublasLtMatrixLayoutDestroy(mat.desc));
}

void TransformLtMatrix(cublasLtHandle_t handle,
    cublasLtMatrixTransformDesc_t trans_desc,
    cublasLtMatrix_t src, cublasLtMatrix_t dst) {

    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_API_CALL(cublasLtMatrixTransform(handle, trans_desc,
        &alpha, src.ptr, src.desc,
        &beta, nullptr, nullptr, 
        dst.ptr, dst.desc, 0));
}

struct LtGemmParam_t {
    cublasLtMatmulDesc_t op_desc;
    void* alpha;
    void* beta;
    cublasLtMatrix_t A;
    cublasLtMatrix_t B;
    cublasLtMatrix_t C;
    cublasLtMatrix_t D;
    cublasLtMatmulAlgo_t* algo;
    void* workspace;
    size_t workspace_size;
};

GemmDtype_t GemmDtype(LtGemmParam_t lt_param) {
    GemmDtype_t gemm_dtype;
    gemm_dtype.Atype = LtMatrixAttr(lt_param.A).dtype;
    gemm_dtype.Btype = LtMatrixAttr(lt_param.B).dtype;
    gemm_dtype.Ctype = LtMatrixAttr(lt_param.C).dtype;

    int dtype;
    CUBLAS_API_CALL(cublasLtMatmulDescGetAttribute(
	    lt_param.op_desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
	    &dtype, sizeof(int), nullptr));

    gemm_dtype.computeType = static_cast<cublasDataType_t>(dtype);
    return gemm_dtype;
}

LtGemmParam_t CreateLtGemmParameter(const GemmParam_t& param) {
    LtGemmParam_t lt_param;

    CUBLAS_API_CALL(cublasLtMatmulDescCreate(&lt_param.op_desc, param.dtype.computeType));
    CUBLAS_API_CALL(cublasLtMatmulDescSetAttribute(lt_param.op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &param.transa, sizeof(cublasOperation_t)));
    CUBLAS_API_CALL(cublasLtMatmulDescSetAttribute(lt_param.op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &param.transb, sizeof(cublasOperation_t)));

    lt_param.alpha = param.alpha;
    lt_param.beta = param.beta;
    lt_param.A = CreateLtMatrix(param.A,
        param.transa == CUBLAS_OP_N ? param.m : param.k,
        param.transa == CUBLAS_OP_N ? param.k : param.m,
        param.lda, param.dtype.Atype);
    lt_param.B = CreateLtMatrix(param.B,
        param.transb == CUBLAS_OP_N ? param.k : param.n,
        param.transb == CUBLAS_OP_N ? param.n : param.k,
        param.ldb, param.dtype.Btype);   
    lt_param.C = CreateLtMatrix(param.C, 
        param.m, param.n, param.ldc, param.dtype.Ctype);
    lt_param.D = CreateLtMatrix(param.D,
        param.m, param.n, param.ldc, param.dtype.Ctype);
    lt_param.workspace = param.workspace;
    lt_param.workspace_size = param.workspace_size;
    lt_param.algo = nullptr;
    return lt_param;
}

void DestroyLtGemmParameter(LtGemmParam_t lt_param) {
    CUBLAS_API_CALL(cublasLtMatmulDescDestroy(lt_param.op_desc));
    DestroyLtMatrix(lt_param.A);
    DestroyLtMatrix(lt_param.B);
    DestroyLtMatrix(lt_param.C);
    DestroyLtMatrix(lt_param.D);
}

struct LtImmaParam_t {
    cublasLtMatrixTransformDesc_t trans_desc;
    cublasLtMatrix_t trans_A;
    cublasLtMatrix_t trans_B;
    cublasLtMatrix_t trans_C;
};

LtImmaParam_t CreateLtImmaParameter(cublasLtHandle_t handle,
    const GemmParam_t& param, const LtGemmParam_t& lt_param) {

    LtImmaParam_t imma_param;
    imma_param.trans_A = CreateTransformLtMatrix(param.m, param.k, param.dtype.Atype, CUBLASLT_ORDER_COL32);
    imma_param.trans_B = CreateTransformLtMatrix(param.k, param.n, param.dtype.Btype, CUBLASLT_ORDER_COL4_4R2_8C);
    imma_param.trans_C = CreateTransformLtMatrix(param.m, param.n, param.dtype.Ctype, CUBLASLT_ORDER_COL32);

    CUBLAS_API_CALL(cublasLtMatrixTransformDescCreate(&imma_param.trans_desc,
        CUDA_R_32F));

    TransformLtMatrix(handle, imma_param.trans_desc, lt_param.A, imma_param.trans_A);
    TransformLtMatrix(handle, imma_param.trans_desc, lt_param.B, imma_param.trans_B);

    RUNTIME_API_CALL(cudaStreamSynchronize(0));
    return imma_param;
}

void DestroyLtImmaParameter(LtImmaParam_t imma_param) {
    DestroyLtMatrix(imma_param.trans_A);
    DestroyLtMatrix(imma_param.trans_B);
    DestroyLtMatrix(imma_param.trans_C);
    CUBLAS_API_CALL(cublasLtMatrixTransformDescDestroy(imma_param.trans_desc));
}

void PrintMatrix(cublasLtMatrix_t mat) {
    auto attr = LtMatrixAttr(mat);
    PrintMatrix(mat.ptr, attr.w, attr.h, attr.ld, attr.dtype);
}

Result_t LtMatrixMul(cublasLtHandle_t handle, LtGemmParam_t& lt_param,
    LtImmaParam_t& imma_param, int loop, bool debug,
    const std::string& algo_name)
{
    cudaEvent_t start;
    cudaEvent_t end;
    bool fault = false;
    RUNTIME_API_CALL(cudaEventCreate(&start));
    RUNTIME_API_CALL(cudaEventCreate(&end));

    if (imma_param.trans_desc) {
        RUNTIME_API_CALL(cudaEventRecord(start));
        for (int i = 0; i < loop; ++i) {

            auto ret = cublasLtMatmul(handle, lt_param.op_desc, 
                lt_param.alpha, imma_param.trans_A.ptr, imma_param.trans_A.desc,
                imma_param.trans_B.ptr, imma_param.trans_B.desc, lt_param.beta,
                imma_param.trans_C.ptr, imma_param.trans_C.desc,
                imma_param.trans_C.ptr, imma_param.trans_C.desc,
                lt_param.algo,
                lt_param.workspace, lt_param.workspace_size, 0);
            if (ret != CUBLAS_STATUS_SUCCESS) {
                fault = true;
                if (debug) {
                    std::cerr << "cublasLtMatmul, " << 
                        cublasGetErrorString(ret) << std::endl;
                }
                break;
            }
        }
        RUNTIME_API_CALL(cudaEventRecord(end));
    }
    else {
        RUNTIME_API_CALL(cudaEventRecord(start));
        for (int i = 0; i < loop; ++i) {

            auto ret = cublasLtMatmul(handle, lt_param.op_desc, 
                lt_param.alpha, lt_param.A.ptr, lt_param.A.desc,
                lt_param.B.ptr, lt_param.B.desc, lt_param.beta,
                lt_param.C.ptr, lt_param.C.desc,
                lt_param.C.ptr, lt_param.C.desc,
                lt_param.algo,
                lt_param.workspace, lt_param.workspace_size, 0);
            if (ret != CUBLAS_STATUS_SUCCESS) {
                fault = true;
                if (debug) {
                    std::cerr << "cublasLtMatmul, " << 
                        cublasGetErrorString(ret) << std::endl;
                }
                break;
            }
        }
        RUNTIME_API_CALL(cudaEventRecord(end));
    }

    RUNTIME_API_CALL(cudaEventSynchronize(end));

    if (imma_param.trans_desc && !fault) {
        TransformLtMatrix(handle, imma_param.trans_desc, imma_param.trans_C, lt_param.C);
        RUNTIME_API_CALL(cudaStreamSynchronize(0));
    }

    if (!fault) {
        fault = !Verify(lt_param.C.ptr, lt_param.D.ptr, LtMatrixCount(lt_param.C), LtMatrixDtype(lt_param.C));
        if (fault && debug) {
            std::cerr << "cublasLtMatmul verification failed" << std::endl;
            PrintMatrix(lt_param.A);
            PrintMatrix(lt_param.B);
            PrintMatrix(lt_param.C);
            PrintMatrix(lt_param.D);
        }
    }
    RUNTIME_API_CALL(cudaMemset(lt_param.C.ptr, 0, LtMatrixSizeInBytes(lt_param.C)));

    float time = 0.f;
    RUNTIME_API_CALL(cudaEventElapsedTime(&time, start, end));
    RUNTIME_API_CALL(cudaEventDestroy(start));
    RUNTIME_API_CALL(cudaEventDestroy(end));

    time = fault ? FLT_MAX : (time / loop);
    return Result_t{algo_name, time};
}

std::vector<Result_t> ProfileAllLtGemmAlgo(cublasLtHandle_t handle,
    LtGemmParam_t& lt_param, LtImmaParam_t& imma_param, int loop, bool debug) {

    const int max_algos = 40;
    std::vector<int> algo_ids(max_algos);
    int nb_algo_id = 0;

    auto gemm_dtype = GemmDtype(lt_param);

    CUBLAS_API_CALL(cublasLtMatmulAlgoGetIds(
        handle, gemm_dtype.computeType, gemm_dtype.computeType,
        gemm_dtype.Atype, gemm_dtype.Btype, gemm_dtype.Ctype, gemm_dtype.Ctype,
        max_algos, algo_ids.data(), &nb_algo_id));
    algo_ids.resize(nb_algo_id);

    std::vector<Result_t> results;
    const int max_combine_option = 6000;
    int combine_count = 0;

    for (int idx = 0; (idx < nb_algo_id) && (combine_count < max_combine_option); idx++) {
        cublasLtMatmulAlgo_t algo;
        CUBLAS_API_CALL(cublasLtMatmulAlgoInit(handle, 
            gemm_dtype.computeType, gemm_dtype.computeType, 
            gemm_dtype.Atype, gemm_dtype.Btype, gemm_dtype.Ctype, gemm_dtype.Ctype,
            algo_ids[idx], &algo));
 
        int splite_k_support;
        int reduction_scheme_mask;
        int swizzling_support;
        int custom_option_max;
        int epilogue_mask;
        std::vector<int> tile_ids;

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
        if (nb_tiles > 0) {
            tile_ids.resize(nb_tiles);
            CUBLAS_API_CALL(cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tile_ids.data(), size_in_bytes, nullptr));
        }
        else {
            tile_ids.resize(1);
            tile_ids[0] = static_cast<int>(CUBLASLT_MATMUL_TILE_UNDEFINED);
        }

        for (auto tile_id : tile_ids) {
 
            CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(int)));
 
            for (int c = 0; c <= custom_option_max; ++c) {
                CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &c, sizeof(int)));
 
                for (int s = 0; s <= swizzling_support; ++s) {
 
                    CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &s, sizeof(int)));
 
                    const static std::vector<int> num_split_k_option{1, 2, 3, 4};
                    for (auto splite_k : num_split_k_option) {

                        if (splite_k > 1 && !splite_k_support) continue;

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

                            if (splite_k > 1 && !(reduction & reduction_scheme_mask))
                                continue;

                            CUBLAS_API_CALL(cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(int)));
 
                            cublasLtMatmulHeuristicResult_t heur_result;
                            cublasStatus_t ret;
                            ret = (cublasLtMatmulAlgoCheck(handle, lt_param.op_desc,
                                lt_param.A.desc, lt_param.B.desc, lt_param.C.desc, lt_param.C.desc,
                                &algo, &heur_result));

                            if (ret == CUBLAS_STATUS_SUCCESS &&
                                heur_result.state == CUBLAS_STATUS_SUCCESS &&
                                heur_result.workspaceSize <= lt_param.workspace_size) {
                                lt_param.algo = &algo;

                                results.push_back(LtMatrixMul(handle,
                                    lt_param, imma_param, loop, debug, "CUBLASLT_ALL_ALG"));

                                cublasLtAlgoAttr_t algo_attr{idx, tile_id, reduction, s, c,
                                    heur_result.workspaceSize, heur_result.wavesCount};
                                combine_count++;
                            }
                            else if (debug) {
                                std::cerr << "cublasLtMatmulAlgoCheck, " << cublasGetErrorString(ret) << 
                                    ", needed workspace size, " << heur_result.workspaceSize << 
                                    ", current workspace size, " << lt_param.workspace_size << std::endl;
                            }
                        } // end of reduction scheme
                    } // end of splite-k
                } // end of swizzling support
            } // end of cutom option
        } // end of tile size
    } // end of algorithm
    return results;
}

std::vector<cublasLtMatmulHeuristicResult_t> HeuristicLtGemmAlgo(cublasLtHandle_t handle, 
    LtGemmParam_t& lt_param, int num_algo, bool debug) {

    // optional, use heuristic approach to select best GEMM kernel,
    // but not support IMMA currently
    cublasLtMatmulPreference_t preference;
    CUBLAS_API_CALL(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_API_CALL(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &lt_param.workspace_size, sizeof(size_t)));

    int nb_result = 0;
    std::vector<cublasLtMatmulHeuristicResult_t> results(num_algo);

    auto ret = cublasLtMatmulAlgoGetHeuristic(
        handle, lt_param.op_desc, lt_param.A.desc, lt_param.B.desc,
        lt_param.C.desc, lt_param.C.desc, preference,
        num_algo, results.data(), &nb_result);

    if (nb_result > 0) {
        results.resize(nb_result);
    }
    else if (debug) {
        std::cerr << "cublasLtMatmulAlgoGetHeuristic, " << cublasGetErrorString(ret) << std::endl;
    }
    CUBLAS_API_CALL(cublasLtMatmulPreferenceDestroy(preference));
    return results;
}

std::vector<Result_t> ProfileLtGemm(const GemmParam_t& param, bool all_algo, int loop, bool debug) {
    cublasLtHandle_t handle;
    CUBLAS_API_CALL(cublasLtCreate(&handle));

    LtGemmParam_t lt_param = CreateLtGemmParameter(param);

    bool use_imma = param.dtype.computeType == CUDA_R_32I &&
                    param.transa == CUBLAS_OP_N &&
                    param.transb == CUBLAS_OP_T;

    LtImmaParam_t imma_param;
    memset(&imma_param, 0, sizeof(LtImmaParam_t));

    if (use_imma) {
        imma_param = CreateLtImmaParameter(handle, param, lt_param);
    }

    std::vector<Result_t> results;
    if (all_algo) {
        results = ProfileAllLtGemmAlgo(handle, lt_param, imma_param, loop, debug);
    }
    else {
        std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results;
        std::string algo_name{"CUBLASLT_DEFAULT_ALG"};

        if (use_imma) {
            algo_name = "CUBLASLT_IMMA_ALG";
        }
        else {
            heuristic_results = HeuristicLtGemmAlgo(handle, lt_param, 1, debug);
            if (heuristic_results.size() > 0) {
                algo_name = "CUBLASLT_HEURISTIC_ALG";
                lt_param.algo = &heuristic_results[0].algo;
            }
        }

        results.push_back(LtMatrixMul(handle, lt_param, imma_param,
            loop, debug, algo_name));
    }

    if (use_imma) {
        DestroyLtImmaParameter(imma_param);
    }
    DestroyLtGemmParameter(lt_param);
    CUBLAS_API_CALL(cublasLtDestroy(handle));

    return results;
}
