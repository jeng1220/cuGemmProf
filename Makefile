all:
	nvcc -O3 -c -I/usr/local/cuda/include ./verify.cu -o verify.o
	g++ -O3 -c -std=c++11 -I./cxxopts/include -I/usr/local/cuda/include ./cuGemmProf.cpp -o cuGemmProf.o
	g++ -O3 -std=c++11 ./cuGemmProf.o verify.o -L/usr/local/cuda/lib64 -lcublasLt -lcublas -lcudart -o cuGemmProf

.PHONY: clean
clean:
	rm -rf cuGemmProf
