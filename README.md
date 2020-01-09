# cuGEMM prof #
A simple tool to profile multiple combinations of GEMM based on cuBLAS

## requirement ##
* CUDA 10.1+
* cuBLAS 10.1+
* CUDA Compute Capability 6.0+

## build ##
```sh
$ git submodule init && git submodule update
$ make
```

## run ##
```sh
$ ./cuGemmProf --help
$ ./cuGemmProf -m 128 -n 64 -k 1024 --type 5,6 -l 10
$ ./cuGemmProf -m 128 -n 64 -k 1024 --type 5,6 -l 10 --algo 3,5
$ ./cuGemmProf -m 128 -n 64 -k 1024 --type 5,6 -l 10 --algo 3,5 --tensor_algo 11,7
$ ./cuGemmProf -m 128 -n 64 -k 1024 --type 5,6 -l 10 --all_algo
```

## available options ##
```sh
Usage:
  ./cuGemmProf [OPTION...]

  -m, arg                m dimension (default: 32)
  -n, arg                n dimension (default: 32)
  -k, arg                k dimension (default: 32)
  -d, arg                device ID (default: 0)
  -l, arg                loop (default: 1)
      --ta               set A to CUBLAS_OP_T, else CUBLAS_OP_N
      --tb               set B to CUBLAS_OP_T, else CUBLAS_OP_N
      --type arg         slect combination of types (default: 5)
      --algo arg         assgin algorithm ID (0~23)
      --tensor_algo arg  assgin TensorOp algorithm ID (0~15)
      --all_algo         run all algorithms
  -w, --workspace arg    workspace size, unit: MiB (default: 0)
  -g, --debug            dump matrices if verification is failed
  -h, --help             print help

available combination of types:
ID, ComputeType, A,      B,      C
0,  {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F}
1,  {CUDA_R_32I, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I}
2,  {CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F}
3,  {CUDA_R_32F, CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32F}
4,  {CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F}
5,  {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F}
6,  {CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F}
7,  {CUDA_C_32F, CUDA_C_8I,  CUDA_C_8I,  CUDA_C_32F}
8,  {CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F}
9,  {CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F}
```
