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

#include "cublasLtMatMul.h"
#include <cassert>
#include <cfloat>
#include <cstring>
#include <iostream>
#include <cublasLt.h>
#include "macro.h"
#include "verify.h"

int RoundOff(int v, int d) {
   return (v + d - 1) / d * d;
}

struct LtMatrix_t {
    void* ptr;
    cublasLtMatrixLayout_t desc;
    bool own;
};

struct LtMatrixAttr_t {
    long w;
    long h;
    long ld;
    cudaDataType_t dtype;
};

LtMatrixAttr_t LtMatrixAttr(const LtMatrix_t& mat) {
    LtMatrixAttr_t attr;
    
    CUBLAS_CHECK(cublasLtMatrixLayoutGetAttribute(
	    mat.desc, CUBLASLT_MATRIX_LAYOUT_ROWS,
        &attr.w, sizeof(long), nullptr));

    CUBLAS_CHECK(cublasLtMatrixLayoutGetAttribute(
	    mat.desc, CUBLASLT_MATRIX_LAYOUT_COLS,
        &attr.h, sizeof(long), nullptr));

    CUBLAS_CHECK(cublasLtMatrixLayoutGetAttribute(
	    mat.desc, CUBLASLT_MATRIX_LAYOUT_LD,
        &attr.ld, sizeof(long), nullptr));

    int dtype;
    CUBLAS_CHECK(cublasLtMatrixLayoutGetAttribute(
	    mat.desc, CUBLASLT_MATRIX_LAYOUT_TYPE,
        &dtype, sizeof(int), nullptr));
    attr.dtype = static_cast<cublasDataType_t>(dtype);

    return attr;
}

cublasDataType_t LtMatrixDtype(const LtMatrix_t& mat) {
    return LtMatrixAttr(mat).dtype;
}

size_t LtMatrixCount(const LtMatrix_t& mat) {
    auto attr = LtMatrixAttr(mat);
    return attr.ld * attr.h;
}

size_t LtMatrixSizeInBytes(const LtMatrix_t& mat) {
    auto attr = LtMatrixAttr(mat);
    return attr.ld * attr.h * DtypeToSize(attr.dtype);
}

LtMatrix_t CreateLtMatrix(const void* ptr, int w, int h, int ld,
    cublasDataType_t dtype) {

    LtMatrix_t mat;
    mat.own = false;
    mat.ptr = const_cast<void*>(ptr);
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
       &mat.desc, dtype, w, h, ld));
    
    return mat;
}

LtMatrix_t CreateTransformLtMatrix(int w, int h,
    cublasDataType_t dtype, cublasLtOrder_t order) {

    LtMatrix_t mat;
    mat.own = true;

    if (order == CUBLASLT_ORDER_COL32) {
        int ld = 32 * w;
        CUDA_CHECK(cudaMalloc(&mat.ptr,
            DtypeToSize(dtype) * RoundOff(h, 32) / 32 * ld));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mat.desc, dtype,
                w, h, ld));
    }
    else if (order == CUBLASLT_ORDER_COL4_4R2_8C) {
        int ld = 32 * RoundOff(h, 8);
        CUDA_CHECK(cudaMalloc(&mat.ptr,
            DtypeToSize(dtype) * RoundOff(w, 32) / 32 * ld));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mat.desc, dtype,
                h, w, ld));
    }
    else {
        assert(EXIT_FAILURE);
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(mat.desc,
        CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(cublasLtOrder_t)));
    return mat;
}

void DestroyLtMatrix(LtMatrix_t& mat) {
    if (mat.own) CUDA_CHECK(cudaFree(mat.ptr));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(mat.desc));
}

void TransformLtMatrix(cublasLtHandle_t handle,
    cublasLtMatrixTransformDesc_t trans_desc,
    const LtMatrix_t& src, LtMatrix_t& dst) {

    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatrixTransform(handle, trans_desc,
        &alpha, src.ptr, src.desc,
        &beta, nullptr, nullptr, 
        dst.ptr, dst.desc, 0));
}

struct LtGemmParam_t {
    cublasLtMatmulDesc_t op_desc;
    void* alpha;
    void* beta;
    LtMatrix_t A;
    LtMatrix_t B;
    LtMatrix_t C;
    LtMatrix_t D;
    cublasLtMatmulAlgo_t* algo;
    void* workspace;
    size_t workspace_size;
};

GemmDtype_t GemmDtype(const LtGemmParam_t& lt_param) {
    GemmDtype_t gemm_dtype;
    gemm_dtype.Atype = LtMatrixAttr(lt_param.A).dtype;
    gemm_dtype.Btype = LtMatrixAttr(lt_param.B).dtype;
    gemm_dtype.Ctype = LtMatrixAttr(lt_param.C).dtype;

    int dtype;
    CUBLAS_CHECK(cublasLtMatmulDescGetAttribute(
	    lt_param.op_desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
	    &dtype, sizeof(int), nullptr));

    gemm_dtype.computeType = static_cast<cublasDataType_t>(dtype);
    return gemm_dtype;
}

LtGemmParam_t CreateLtGemmParameter(const GemmParam_t& param) {
    LtGemmParam_t lt_param;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&lt_param.op_desc, param.dtype.computeType));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(lt_param.op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &param.transa, sizeof(cublasOperation_t)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(lt_param.op_desc,
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

void DestroyLtGemmParameter(LtGemmParam_t& lt_param) {
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(lt_param.op_desc));
    DestroyLtMatrix(lt_param.A);
    DestroyLtMatrix(lt_param.B);
    DestroyLtMatrix(lt_param.C);
    DestroyLtMatrix(lt_param.D);
}

struct LtImmaParam_t {
    cublasLtMatrixTransformDesc_t trans_desc;
    LtMatrix_t trans_A;
    LtMatrix_t trans_B;
    LtMatrix_t trans_C;
};

LtImmaParam_t CreateLtImmaParameter(cublasLtHandle_t handle,
    const GemmParam_t& param, const LtGemmParam_t& lt_param) {

    LtImmaParam_t imma_param;
    imma_param.trans_A = CreateTransformLtMatrix(param.m, param.k, param.dtype.Atype, CUBLASLT_ORDER_COL32);
    imma_param.trans_B = CreateTransformLtMatrix(param.k, param.n, param.dtype.Btype, CUBLASLT_ORDER_COL4_4R2_8C);
    imma_param.trans_C = CreateTransformLtMatrix(param.m, param.n, param.dtype.Ctype, CUBLASLT_ORDER_COL32);

    CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&imma_param.trans_desc,
        CUDA_R_32F));

    TransformLtMatrix(handle, imma_param.trans_desc, lt_param.A, imma_param.trans_A);
    TransformLtMatrix(handle, imma_param.trans_desc, lt_param.B, imma_param.trans_B);

    CUDA_CHECK(cudaStreamSynchronize(0));
    return imma_param;
}

void DestroyLtImmaParameter(LtImmaParam_t& imma_param) {
    DestroyLtMatrix(imma_param.trans_A);
    DestroyLtMatrix(imma_param.trans_B);
    DestroyLtMatrix(imma_param.trans_C);
    CUBLAS_CHECK(cublasLtMatrixTransformDescDestroy(imma_param.trans_desc));
}

void PrintMatrix(LtMatrix_t mat) {
    auto attr = LtMatrixAttr(mat);
    PrintMatrix(mat.ptr, attr.w, attr.h, attr.ld, attr.dtype);
}

ProfResult_t LtMatrixMul(cublasLtHandle_t handle, LtGemmParam_t& lt_param,
    const LtImmaParam_t& imma_param, int loop, bool debug,
    cublasGemmAlgo_t algo_name)
{
    cudaEvent_t start;
    cudaEvent_t end;
    bool fault = false;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    if (imma_param.trans_desc) {
        CUDA_CHECK(cudaEventRecord(start));
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
        CUDA_CHECK(cudaEventRecord(end));
    }
    else {
        CUDA_CHECK(cudaEventRecord(start));
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
        CUDA_CHECK(cudaEventRecord(end));
    }

    CUDA_CHECK(cudaEventSynchronize(end));

    if (imma_param.trans_desc && !fault) {
        TransformLtMatrix(handle, imma_param.trans_desc, imma_param.trans_C, lt_param.C);
        CUDA_CHECK(cudaStreamSynchronize(0));
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
    CUDA_CHECK(cudaMemset(lt_param.C.ptr, 0, LtMatrixSizeInBytes(lt_param.C)));

    float time = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    time = fault ? FLT_MAX : (time / loop);
    return ProfResult_t{algo_name, time};
}

std::vector<LtProfResult_t> ProfileAllLtGemmAlgo(cublasLtHandle_t handle,
    LtGemmParam_t& lt_param, const LtImmaParam_t& imma_param, int loop, bool debug) {

    const int max_algos = 40;
    std::vector<int> algo_ids(max_algos);
    int nb_algo_id = 0;

    auto gemm_dtype = GemmDtype(lt_param);

    CUBLAS_CHECK(cublasLtMatmulAlgoGetIds(
        handle, gemm_dtype.computeType, gemm_dtype.computeType,
        gemm_dtype.Atype, gemm_dtype.Btype, gemm_dtype.Ctype, gemm_dtype.Ctype,
        max_algos, algo_ids.data(), &nb_algo_id));
    algo_ids.resize(nb_algo_id);

    std::vector<LtProfResult_t> results;
    const int max_combine_option = 6000;
    int combine_count = 0;

    for (int idx = 0; (idx < nb_algo_id) && (combine_count < max_combine_option); idx++) {
        cublasLtMatmulAlgo_t algo;
        CUBLAS_CHECK(cublasLtMatmulAlgoInit(handle, 
            gemm_dtype.computeType, gemm_dtype.computeType, 
            gemm_dtype.Atype, gemm_dtype.Btype, gemm_dtype.Ctype, gemm_dtype.Ctype,
            algo_ids[idx], &algo));
 
        int splite_k_support;
        int reduction_scheme_mask;
        int swizzling_support;
        int custom_option_max;
        std::vector<int> tile_ids;

        CUBLAS_CHECK(cublasLtMatmulAlgoCapGetAttribute(&algo,
            CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splite_k_support, sizeof(int), nullptr));
        CUBLAS_CHECK(cublasLtMatmulAlgoCapGetAttribute(&algo,
            CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &reduction_scheme_mask, sizeof(int), nullptr));
        CUBLAS_CHECK(cublasLtMatmulAlgoCapGetAttribute(&algo,
            CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzling_support, sizeof(int), nullptr));
        CUBLAS_CHECK(cublasLtMatmulAlgoCapGetAttribute(&algo,
            CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_option_max, sizeof(int), nullptr));

        size_t size_in_bytes = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS,
            nullptr, 0, &size_in_bytes));

        int nb_tiles = static_cast<int>(size_in_bytes / sizeof(int));
        if (nb_tiles > 0) {
            tile_ids.resize(nb_tiles);
            CUBLAS_CHECK(cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tile_ids.data(), size_in_bytes, nullptr));
        }
        else {
            tile_ids.resize(1);
            tile_ids[0] = static_cast<int>(CUBLASLT_MATMUL_TILE_UNDEFINED);
        }

        for (auto tile_id : tile_ids) {
 
            CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(int)));
 
            for (int c = 0; c <= custom_option_max; ++c) {
                CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &c, sizeof(int)));
 
                for (int s = 0; s <= swizzling_support; ++s) {
 
                    CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &s, sizeof(int)));
 
                    const static std::vector<int> num_split_k_option{1, 2, 3, 4};
                    for (auto splite_k : num_split_k_option) {

                        if (splite_k > 1 && !splite_k_support) continue;

                        CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&algo,
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

                            CUBLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(int)));
 
                            cublasLtMatmulHeuristicResult_t heur_result;
                            cublasStatus_t ret;

                            if (imma_param.trans_desc) {
                                ret = (cublasLtMatmulAlgoCheck(handle, lt_param.op_desc,
                                    imma_param.trans_A.desc, imma_param.trans_B.desc,
                                    imma_param.trans_C.desc, imma_param.trans_C.desc,
                                    &algo, &heur_result));
                            }
                            else {
                                ret = (cublasLtMatmulAlgoCheck(handle, lt_param.op_desc,
                                    lt_param.A.desc, lt_param.B.desc, lt_param.C.desc, lt_param.C.desc,
                                    &algo, &heur_result));
                            }

                            if (ret == CUBLAS_STATUS_SUCCESS &&
                                heur_result.state == CUBLAS_STATUS_SUCCESS &&
                                heur_result.workspaceSize <= lt_param.workspace_size) {
                                lt_param.algo = &algo;

                                auto result = LtMatrixMul(handle,
                                    lt_param, imma_param, loop, debug, 
                                    static_cast<cublasGemmAlgo_t>(__CUBLASLT_ALL_ALG__));

                                LtGemmAlgoAttr_t attr{idx, tile_id, splite_k, reduction, s, c,
                                    heur_result.workspaceSize, heur_result.wavesCount};

                                results.push_back(LtProfResult_t{attr, result});
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
    LtGemmParam_t& lt_param, const LtImmaParam_t& imma_param, int num_algo, bool debug) {

    // optional, use heuristic approach to select best GEMM kernel,
    // but not support IMMA currently
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &lt_param.workspace_size, sizeof(size_t)));

    int nb_result = 0;
    std::vector<cublasLtMatmulHeuristicResult_t> results(num_algo);

    cublasStatus_t ret;
    if (imma_param.trans_desc) {
        ret = cublasLtMatmulAlgoGetHeuristic(
            handle, lt_param.op_desc, 
            imma_param.trans_A.desc, imma_param.trans_B.desc,
            imma_param.trans_C.desc, imma_param.trans_C.desc,
            preference, num_algo, results.data(), &nb_result);
    }
    else {
        ret = cublasLtMatmulAlgoGetHeuristic(
            handle, lt_param.op_desc, lt_param.A.desc, lt_param.B.desc,
            lt_param.C.desc, lt_param.C.desc, preference,
            num_algo, results.data(), &nb_result);
    }

    if (nb_result > 0) {
        results.resize(nb_result);
    }
    else if (debug) {
        std::cerr << "cublasLtMatmulAlgoGetHeuristic, " << cublasGetErrorString(ret) << std::endl;
    }
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    return results;
}

LtGemmAlgoAttr_t LtGemmAlgoAttr(const cublasLtMatmulAlgo_t* algo,
    size_t workspace_size, int wave_count) {
    LtGemmAlgoAttr_t attr;
    CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
	    algo, CUBLASLT_ALGO_CONFIG_ID, &attr.algo_id, sizeof(int), nullptr));
    CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
	    algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &attr.tile_id, sizeof(int), nullptr));
    CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
	    algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &attr.splite_k, sizeof(int), nullptr));
    CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
	    algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &attr.reduction_scheme, sizeof(int), nullptr));
    CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
	    algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &attr.swizzle, sizeof(int), nullptr));
    CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
	    algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &attr.custom_option, sizeof(int), nullptr));
    attr.workspace_size = workspace_size;
    attr.wave_count = wave_count;
    return attr;
}

std::vector<LtProfResult_t> ProfileLtGemm(const GemmParam_t& param, bool all_algo, int loop, bool debug) {
    cublasLtHandle_t handle;
    CUBLAS_CHECK(cublasLtCreate(&handle));

    LtGemmParam_t lt_param = CreateLtGemmParameter(param);

    bool use_imma = param.dtype.computeType == CUDA_R_32I &&
                    param.transa == CUBLAS_OP_N &&
                    param.transb == CUBLAS_OP_T;

    LtImmaParam_t imma_param;
    memset(&imma_param, 0, sizeof(LtImmaParam_t));

    if (use_imma) {
        imma_param = CreateLtImmaParameter(handle, param, lt_param);
    }

    std::vector<LtProfResult_t> results;
    if (all_algo) {
        results = ProfileAllLtGemmAlgo(handle, lt_param, imma_param, loop, debug);
    }

    auto algo_name = static_cast<cublasGemmAlgo_t>(__CUBLASLT_DEFAULT_ALG__);
    // clean up
    lt_param.algo = nullptr;
    lt_param.workspace_size = 0;
    lt_param.workspace = nullptr;
    LtGemmAlgoAttr_t attr;
    memset(&attr, 0, sizeof(LtGemmAlgoAttr_t));

    if (use_imma) {
        algo_name = static_cast<cublasGemmAlgo_t>(__CUBLASLT_DEFAULT_IMMA_ALG__);
    }

    auto heuristic_results = HeuristicLtGemmAlgo(handle, lt_param, imma_param, 1, debug);
    if (heuristic_results.size() > 0) {
        algo_name = static_cast<cublasGemmAlgo_t>(__CUBLASLT_1ST_HEURISTIC_ALG__);
        lt_param.algo = &heuristic_results[0].algo;
        attr = LtGemmAlgoAttr(&heuristic_results[0].algo,
            heuristic_results[0].workspaceSize, heuristic_results[0].wavesCount);
    }

    auto result = LtMatrixMul(handle, lt_param, imma_param,
        loop, debug, algo_name);
    results.push_back(LtProfResult_t{attr, result});

    if (use_imma) {
        DestroyLtImmaParameter(imma_param);
    }
    DestroyLtGemmParameter(lt_param);
    CUBLAS_CHECK(cublasLtDestroy(handle));

    return results;
}
