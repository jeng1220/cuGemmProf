all:
	g++ -O3 -std=c++11 -I./cxxopts/include -I/usr/local/cuda/include ./gemm_perf.cpp -L/usr/local/cuda/lib64 -lcublas -lcudart -o gemm_perf

.PHONY: clean
clean:
	rm -rf gemm_perf
