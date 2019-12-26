# cuGEMM prof #
A simple tool to profile performance of multiple combinations of GEMM of cuBLAS

## requirement ##
* CUDA
* cuBLAS

## build ##
```sh
$ make
```

## run ##
```sh
$ ./gemm_prof --help
$ ./gemm_prof -m 128 -n 64 -k 1024 --type 5,6 -l 10
```

## available options ##
```sh
Usage:
  ./gemm_perf [OPTION...]

  -m, arg         m dimension (default: 32)
  -n, arg         n dimension (default: 32)
  -k, arg         k dimension (default: 32)
  -d, arg         device ID (default: 0)
  -l, arg         loop (default: 1)
      --ta        set A to CUBLAS_OP_T, else CUBLAS_OP_N
      --tb        set B to CUBLAS_OP_T, else CUBLAS_OP_N
      --type arg  slect combination of types (default: 5)
      --help      print help

available combination of types:
ID, ComputeType, Atype,      Btype,      Ctype
0,  {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F}
1,  {CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I}
2,  {CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F}
3,  {CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F}
4,  {CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F}
5,  {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F}
6,  {CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F}
7,  {CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F}
8,  {CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F}

```
