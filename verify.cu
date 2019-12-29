#include "macro.h"
#include "verify.h"
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>

void* AllocAlphaScale(cudaDataType_t dtype) {
    void* ptr = nullptr;
    switch (dtype) {

        case CUDA_R_8I:
            ptr = malloc(1);
            *(reinterpret_cast<char*>(ptr)) = 1;
            break;
        case CUDA_R_16F:
            ptr = malloc(2);
            *(reinterpret_cast<__half*>(ptr)) = __float2half(1.f);
            break;
        case CUDA_R_32I:
            ptr = malloc(4);
            *(reinterpret_cast<int*>(ptr)) = 1;
            break;
        case CUDA_R_32F:
            ptr = malloc(4);
            *(reinterpret_cast<float*>(ptr)) = 1.f;
            break;
        case CUDA_R_64F:
            ptr = malloc(8);
            *(reinterpret_cast<double*>(ptr)) = 1.0;
            break;
        default:
            assert(false);
    }
    return ptr;
}

template <typename data_t>
__global__ void InitMatrixKernal(data_t* ptr, int w, int h, int ld) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < ld && y < h) {
        ptr[y * ld + x] = (x < w) ? static_cast<data_t>(threadIdx.y * blockDim.x + threadIdx.x) : 0;
    }
}

void InitMatrix(void* ptr, int w, int h, int ld, cudaDataType_t dtype) {

    dim3 block(8, 8);
    dim3 grid;
    grid.x = (ld + block.x - 1) / block.x;
    grid.y = ( h + block.y - 1) / block.y;

    switch (dtype) {

        case CUDA_R_8I:
            InitMatrixKernal<char><<<grid, block>>>(reinterpret_cast<char*>(ptr), w, h, ld);
            break;
        case CUDA_R_16F:
            InitMatrixKernal<__half><<<grid, block>>>(reinterpret_cast<__half*>(ptr), w, h, ld);
            break;
        case CUDA_R_32F:
            InitMatrixKernal<float><<<grid, block>>>(reinterpret_cast<float*>(ptr), w, h, ld);
            break;
        case CUDA_R_64F:
            InitMatrixKernal<double><<<grid, block>>>(reinterpret_cast<double*>(ptr), w, h, ld);
        case CUDA_C_8I:
            grid.x = (2 * ld + block.x - 1) / block.x;
            InitMatrixKernal<char><<<grid, block>>>(reinterpret_cast<char*>(ptr), 2 * w, h, 2 * ld);
            break;
        case CUDA_C_32F:
            grid.x = (2 * ld + block.x - 1) / block.x;
            InitMatrixKernal<float><<<grid, block>>>(reinterpret_cast<float*>(ptr), 2 * w, h, 2 * ld);
            break;
        case CUDA_C_64F:
            grid.x = (2 * ld + block.x - 1) / block.x;
            InitMatrixKernal<double><<<grid, block>>>(reinterpret_cast<double*>(ptr), 2 * w, h, 2 * ld);
            break;
        default:
            assert(false);
    }
    RUNTIME_API_CALL(cudaStreamSynchronize(0));
}

template <typename src_t, typename acc_t, typename dst_t>
__global__ void NaiveGemmKernelNN(
    int m, int n, int k,
    src_t* A, int lda,
    src_t* B, int ldb,
    dst_t* C, int ldc) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    acc_t sum = 0;

    if (x < m && y < n) {
        for (int i = 0; i < k; ++i) {
            sum += static_cast<acc_t>(A[i * lda + x]) * static_cast<acc_t>(B[y * ldb + i]);
        }
        C[y * ldc + x] = static_cast<dst_t>(sum);
    }
}

void NaiveGemmNN(
    int m, int n, int k,
    void* A, int lda,
    void* B, int ldb,
    void* C, int ldc,
    int gemm_type) {

    dim3 block(8, 8);
    dim3 grid;
    grid.x = (m + block.x - 1) / block.x;
    grid.y = (n + block.y - 1) / block.y;
    switch (gemm_type) {
        case 0: // CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F
            NaiveGemmKernelNN<__half, float, __half><<<grid, block>>>(m, n, k,
                reinterpret_cast<__half*>(A), lda, 
                reinterpret_cast<__half*>(B), ldb, 
                reinterpret_cast<__half*>(C), ldc);
            break;
        case 1: // CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I
            NaiveGemmKernelNN<char, int, int><<<grid, block>>>(m, n, k,
                reinterpret_cast<char*>(A), lda, 
                reinterpret_cast<char*>(B), ldb, 
                reinterpret_cast<int*>(C), ldc);
            break;
        case 2: // CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F
            NaiveGemmKernelNN<__half, float, __half><<<grid, block>>>(m, n, k,
                reinterpret_cast<__half*>(A), lda, 
                reinterpret_cast<__half*>(B), ldb, 
                reinterpret_cast<__half*>(C), ldc);
            break;
        case 3: // CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F
            NaiveGemmKernelNN<char, float, float><<<grid, block>>>(m, n, k,
                reinterpret_cast<char*>(A), lda, 
                reinterpret_cast<char*>(B), ldb, 
                reinterpret_cast<float*>(C), ldc);
            break;
        case 4: // CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F
            NaiveGemmKernelNN<__half, float, float><<<grid, block>>>(m, n, k,
                reinterpret_cast<__half*>(A), lda, 
                reinterpret_cast<__half*>(B), ldb, 
                reinterpret_cast<float*>(C), ldc);
            break;
        case 5: // CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F
            NaiveGemmKernelNN<float, float, float><<<grid, block>>>(m, n, k,
                reinterpret_cast<float*>(A), lda, 
                reinterpret_cast<float*>(B), ldb, 
                reinterpret_cast<float*>(C), ldc);
            break;
        case 6: // CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F
            NaiveGemmKernelNN<double, double, double><<<grid, block>>>(m, n, k,
                reinterpret_cast<double*>(A), lda, 
                reinterpret_cast<double*>(B), ldb, 
                reinterpret_cast<double*>(C), ldc);
            break;
        case 7: // CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F
        case 8: // CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F
        case 9: // CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F
        default:
            assert(false);
    }
    RUNTIME_API_CALL(cudaStreamSynchronize(0));
}

template<typename T>
struct AbsMinus {
    __thrust_exec_check_disable__
    __host__ __device__ T operator()(const T &lhs, const T &rhs) const {
        return (lhs > rhs) ? lhs - rhs : rhs - lhs;
    }
};

template <typename T>
bool VerifyT(T* x, T* y, int count) {
    T init = 0;
    thrust::maximum<T> binary_op1;
    AbsMinus<T> binary_op2;

    auto result = thrust::inner_product(thrust::device, 
        x, x + count, y, init, binary_op1, binary_op2);

    if (static_cast<double>(result) > 1e-6) {
        std::cerr << "error: " << result << std::endl;
        return false;
    }
    else {
        std::cout << "PASSED" << std::endl;
        return true;
    }
}

bool Verify(void* x, void* y, int count, cudaDataType_t dtype) {
    switch (dtype) {
        case CUDA_R_16F:
            assert(false);
            //return VerifyT<__half>(reinterpret_cast<__half*>(x), reinterpret_cast<__half*>(y), count);
        case CUDA_R_32I:
            return VerifyT<int>(reinterpret_cast<int*>(x), reinterpret_cast<int*>(y), count);
        case CUDA_R_32F:
            return VerifyT<float>(reinterpret_cast<float*>(x), reinterpret_cast<float*>(y), count);
        case CUDA_R_64F:
            return VerifyT<double>(reinterpret_cast<double*>(x), reinterpret_cast<double*>(y), count);
        case CUDA_C_32F:
            return VerifyT<float>(reinterpret_cast<float*>(x), reinterpret_cast<float*>(y), 2 * count);
        case CUDA_C_64F:
            return VerifyT<double>(reinterpret_cast<double*>(x), reinterpret_cast<double*>(y), 2 * count);
        default:
            assert(false);
    }
    return false;
}

void PrintMatrix(float* dev_ptr, int w, int h, int ld)
{
    size_t size = ld * h * sizeof(float);
    float* host_ptr = (float*)malloc(size);
    RUNTIME_API_CALL(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < ld; ++x) {
            printf("%.f, ", host_ptr[y * ld + x]);
        }
        printf("\n");
    }
    printf("\n\n");
    free(host_ptr);
}