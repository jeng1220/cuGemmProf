all:
	g++ -g -O0 -I./cxxopts/include ./gemm_perf.cpp -lcublas -lcudart -o gemm_perf

.PHONY: clean
clean:
	rm -rf gemm_perf
