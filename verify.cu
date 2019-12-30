#include "macro.h"
#include "helper.h"
#include "verify.h"
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <cuda_fp16.h>
#include <iostream>
#include <map>
#include <cstdlib>
#include <cstdio>
#include <cassert>

void* AllocAlphaScale(cudaDataType_t dtype)
{
    void* ptr = nullptr;
    ptr = malloc(Dtype2Size(dtype));
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
        default:
            assert(false);
    }
    return ptr;
}

template <typename data_t>
__global__ void InitMatrixKernal(void* dev_ptr, int w, int h, int ld) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    auto ptr = reinterpret_cast<data_t*>(dev_ptr);
    if (x < ld && y < h) {
        ptr[y * ld + x] = (x < w) ? (threadIdx.y * blockDim.x + threadIdx.x) : 0;
    }
}

void InitMatrix(void* ptr, int w, int h, int ld, cudaDataType_t dtype) 
{
    dim3 block(8, 8);
    dim3 grid;
    grid.x = (ld + block.x - 1) / block.x;
    grid.y = ( h + block.y - 1) / block.y;

    switch (dtype) {

        case CUDA_R_8I:
            InitMatrixKernal<char><<<grid, block>>>(ptr, w, h, ld);
            break;
        case CUDA_R_16F:
            InitMatrixKernal<half><<<grid, block>>>(ptr, w, h, ld);
            break;
        case CUDA_R_32F:
            InitMatrixKernal<float><<<grid, block>>>(ptr, w, h, ld);
            break;
        case CUDA_R_64F:
            InitMatrixKernal<double><<<grid, block>>>(ptr, w, h, ld);
        case CUDA_C_8I:
            grid.x = (2 * ld + block.x - 1) / block.x;
            InitMatrixKernal<char><<<grid, block>>>(ptr, w, h, ld);
            break;
        case CUDA_C_32F:
            grid.x = (2 * ld + block.x - 1) / block.x;
            InitMatrixKernal<float><<<grid, block>>>(ptr, w, h, ld);
            break;
        case CUDA_C_64F:
            grid.x = (2 * ld + block.x - 1) / block.x;
            InitMatrixKernal<double><<<grid, block>>>(ptr, w, h, ld);
            break;
        default:
            assert(false);
    }
    RUNTIME_API_CALL(cudaStreamSynchronize(0));
}

template <typename data_t>
__global__ void NaiveMatrixTransposeKernel(
    int w, int h,
    const void* src_ptr, void* dst_ptr)
{
    auto src = reinterpret_cast<const data_t*>(src_ptr);
    auto dst = reinterpret_cast<data_t*>(dst_ptr);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        dst[ x * h + y ] = src[ y * w + x ];
    }
}

void NaiveMatrixTranspose(
    int w, int h,
    const void* src, void* dst,
    cudaDataType_t dtype)
{

    dim3 block(8, 8);
    dim3 grid;
    grid.x = (w + block.x - 1) / block.x;
    grid.y = (h + block.y - 1) / block.y;

    switch (dtype) {
        case CUDA_R_8I:
            NaiveMatrixTransposeKernel<char><<<grid, block>>>(w, h, src, dst);
            break;
        case CUDA_R_16F:
        case CUDA_C_8I:
            NaiveMatrixTransposeKernel<half><<<grid, block>>>(w, h, src, dst);
            break;
        case CUDA_R_32I:
        case CUDA_R_32F:
            NaiveMatrixTransposeKernel<int><<<grid, block>>>(w, h, src, dst);
            break;
        case CUDA_R_64F:
        case CUDA_C_32F:
            NaiveMatrixTransposeKernel<double><<<grid, block>>>(w, h, src, dst);
            break;
        case CUDA_C_64F:
            NaiveMatrixTransposeKernel<double2><<<grid, block>>>(w, h, src, dst);
            break;
        default:
            assert(false);
    }
    RUNTIME_API_CALL(cudaStreamSynchronize(0));
}

template <typename src_t, typename acc_t, typename dst_t>
__global__ void NaiveGemmKernelNN(
    int m, int n, int k,
    const void* A_ptr, int lda,
    const void* B_ptr, int ldb,
    void* C_ptr, int ldc) 
{
    auto A = reinterpret_cast<const src_t*>(A_ptr);
    auto B = reinterpret_cast<const src_t*>(B_ptr);
    auto C = reinterpret_cast<dst_t*>(C_ptr);

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
    const void* A, int lda,
    const void* B, int ldb,
    void* C, int ldc,
    int gemm_type) 
{

    dim3 block(8, 8);
    dim3 grid;
    grid.x = (m + block.x - 1) / block.x;
    grid.y = (n + block.y - 1) / block.y;
    switch (gemm_type) {
        case 0: // CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F
            NaiveGemmKernelNN<half, float, half><<<grid, block>>>(m, n, k,
                A, lda, B, ldb, C, ldc);
            break;
        case 1: // CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I
            NaiveGemmKernelNN<char, int, int><<<grid, block>>>(m, n, k,
                A, lda, B, ldb, C, ldc);
            break;
        case 2: // CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F
            NaiveGemmKernelNN<half, float, half><<<grid, block>>>(m, n, k,
                A, lda, B, ldb, C, ldc);
            break;
        case 3: // CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F
            NaiveGemmKernelNN<char, float, float><<<grid, block>>>(m, n, k,
                A, lda, B, ldb, C, ldc);
            break;
        case 4: // CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F
            NaiveGemmKernelNN<half, float, float><<<grid, block>>>(m, n, k,
                A, lda, B, ldb, C, ldc);
            break;
        case 5: // CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F
            NaiveGemmKernelNN<float, float, float><<<grid, block>>>(m, n, k,
                A, lda, B, ldb, C, ldc);
            break;
        case 6: // CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F
            NaiveGemmKernelNN<double, double, double><<<grid, block>>>(m, n, k,
                A, lda, B, ldb, C, ldc);
            break;
        case 7: // CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F
        case 8: // CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F
        case 9: // CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F
        default:
            assert(false);
    }
    RUNTIME_API_CALL(cudaStreamSynchronize(0));
}

int GetGemmTypeId(cudaDataType_t compute_type,
    cudaDataType_t src_type, cudaDataType_t dst_type) 
{
    switch (compute_type) {
        case CUDA_R_16F: return 0;
        case CUDA_R_32I: return 1;
        case CUDA_R_32F:
            switch (src_type) {
                case CUDA_R_16F: return (dst_type == CUDA_R_16F) ? 2 : 4;
                case CUDA_R_8I: return 3;
                case CUDA_R_32F: return 5;
                default: assert(false);
            }
        case CUDA_R_64F: return 6;
        case CUDA_C_32F: return (src_type == CUDA_C_8I) ? 7 : 8;
        case CUDA_C_64F: return 9;
        default: assert(false);
    }
    return -1;
}

void NaiveGemm(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const void* A, cudaDataType_t a_type, int lda,
    const void* B, cudaDataType_t b_type, int ldb,
    void* C, cudaDataType_t c_type, int ldc,
    cudaDataType_t compute_type) 
{
    int src_dtype_size = Dtype2Size(a_type);
    void* dev_A = (void*)A;
    int trans_lda = lda;
    if (transa == CUBLAS_OP_T) {
        RUNTIME_API_CALL(cudaMalloc(&dev_A, m * lda * src_dtype_size));
        NaiveMatrixTranspose(lda, m, A, dev_A, a_type);
        trans_lda = m;
    }

    void* dev_B = (void*)B;
    int trans_ldb = ldb;
    if (transb == CUBLAS_OP_T) {
        RUNTIME_API_CALL(cudaMalloc(&dev_B, k * ldb * src_dtype_size));
        NaiveMatrixTranspose(ldb, k, B, dev_B, b_type);
        trans_ldb = k;
    }

    auto gemm_type = GetGemmTypeId(compute_type, a_type, c_type);
    NaiveGemmNN(m, n, k, dev_A, trans_lda, dev_B, trans_ldb, C, ldc, gemm_type);
    if (dev_A != A) RUNTIME_API_CALL(cudaFree(dev_A));
    if (dev_B != B) RUNTIME_API_CALL(cudaFree(dev_B));
}

template<typename T>
struct AbsMinus {
    __thrust_exec_check_disable__
    __host__ __device__ T operator()(const T &lhs, const T &rhs) const {
        return (lhs > rhs) ? lhs - rhs : rhs - lhs;
    }
};

template <typename T>
bool VerifyT(const void* x_ptr, const void* y_ptr, int count) {
    auto x = reinterpret_cast<const T*>(x_ptr);
    auto y = reinterpret_cast<const T*>(y_ptr);

    T init = 0;
    thrust::maximum<T> binary_op1;
    AbsMinus<T> binary_op2;

    auto result = thrust::inner_product(thrust::device, 
        x, x + count, y, init, binary_op1, binary_op2);

    if (static_cast<double>(result) > 1e-6) {
        //std::cerr << "error: " << result << std::endl;
        return false;
    }
    else {
        //std::cout << "PASSED" << std::endl;
        return true;
    }
}

std::ostream& operator<<(std::ostream& os, const half& x) {
    os << __half2float(x);
    return os;
}

bool Verify(const void* x, const void* y, int count, cudaDataType_t dtype) {
    switch (dtype) {
        case CUDA_R_16F:
            return VerifyT<half>(x, y, count);
        case CUDA_R_32I:
            return VerifyT<int>(x, y, count);
        case CUDA_R_32F:
            return VerifyT<float>(x, y, count);
        case CUDA_R_64F:
            return VerifyT<double>(x, y, count);
        case CUDA_C_32F:
            return VerifyT<float>(x, y, 2 * count);
        case CUDA_C_64F:
            return VerifyT<double>(x, y, 2 * count);
        default:
            assert(false);
    }
    return false;
}

template <typename data_t>
void PrintMatrixT(const void* ptr, int w, int h, int ld)
{
    auto dev_ptr = reinterpret_cast<const data_t*>(ptr);
    size_t size = ld * h * sizeof(data_t);
    data_t* host_ptr = (data_t*)malloc(size);
    RUNTIME_API_CALL(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < ld; ++x) {
            std::cout << +host_ptr[y * ld + x] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "\n" << std::endl;
    free(host_ptr);
}

template <>
void PrintMatrixT<half>(const void* ptr, int w, int h, int ld)
{
    auto dev_ptr = reinterpret_cast<const half*>(ptr);
    size_t size = ld * h * sizeof(half);
    half* host_ptr = (half*)malloc(size);
    RUNTIME_API_CALL(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < ld; ++x) {
            std::cout << host_ptr[y * ld + x] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "\n" << std::endl;
    free(host_ptr);
}

void PrintMatrix(const void* dev_ptr, int w, int h,
    int ld, cudaDataType_t dtype)
{
    switch (dtype) {
        case CUDA_R_8I:
            PrintMatrixT<char>(dev_ptr, w, h, ld);
            break;
        case CUDA_R_16F:
            PrintMatrixT<half>(dev_ptr, w, h, ld);
            break;
        case CUDA_R_32I:
            PrintMatrixT<int>(dev_ptr, w, h, ld);
            break;
        case CUDA_R_32F:
            PrintMatrixT<float>(dev_ptr, w, h, ld);
            break;
        case CUDA_R_64F:
            PrintMatrixT<double>(dev_ptr, w, h, ld);
            break;
        case CUDA_C_8I:
            PrintMatrixT<char>(dev_ptr, 2 * w, h, 2 * ld);
            break;
        case CUDA_C_32F:
            PrintMatrixT<float>(dev_ptr, 2 * w, h, 2 * ld);
            break;
        case CUDA_C_64F:
            PrintMatrixT<double>(dev_ptr, 2 * w, h, 2 * ld);
            break;
        default:
            assert(false);
    }
}
