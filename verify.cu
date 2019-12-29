#include "verify.h"
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <cuda_fp16.h>
#include <cstdlib>
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
    cudaStreamSynchronize(0);
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
    void* C, int ldc) {

    dim3 block(8, 8);
    dim3 grid;
    grid.x = (m + block.x - 1) / block.x;
    grid.y = (n + block.y - 1) / block.y;
    NaiveGemmKernelNN<float, float, float><<<grid, block>>>(m, n, k,
        reinterpret_cast<float*>(A), lda, 
        reinterpret_cast<float*>(B), ldb, 
        reinterpret_cast<float*>(C), ldc);
    cudaStreamSynchronize(0);
}

template<typename T>
struct abs_minus {
    typedef T first_argument_type; 
    typedef T second_argument_type; 
    typedef T result_type;
 
    __thrust_exec_check_disable__
    __host__ __device__ T operator()(const T &lhs, const T &rhs) const {
        return (lhs > rhs) ? lhs - rhs : rhs - lhs;
    }
 }; // end minus

void Verify(float* x, float* y, int count) {
    float init = 0;
    abs_minus<float> binary_op1;
    thrust::maximum<float> binary_op2;

    auto result = thrust::inner_product(thrust::device, 
        x, x + count, y, init, binary_op1, binary_op2);
    if (result > 1e-6) {
        std::cerr << "error: " << result << std::endl;
    }
    else {
        std::cout << "PASSED" << std::endl;
    }
}