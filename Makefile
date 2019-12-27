all:
	g++ -O3 -std=c++11 -I./cxxopts/include -I/usr/local/cuda/include ./cuGemmProf.cpp -L/usr/local/cuda/lib64 -lcublasLt -lcublas -lcudart -o cuGemmProf

.PHONY: clean
clean:
	rm -rf cuGemmProf
