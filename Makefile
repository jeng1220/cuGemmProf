CC:=nvcc
exe:=cuGemmProf
obj:=cuGemmProf.o cublasGemmEx.o cublasLtMatMul.o verify.o helper.o
inc:=-I./cxxopts/include -I/usr/local/cuda/include
lib:=-L/usr/local/cuda/lib64 -lcublasLt -lcublas -lcudart
flags:=-O0 -g -std=c++11 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53

all:$(obj)
	$(CC) $(lib) $(obj) -o $(exe)
%.o:%.cpp
	$(CC) -c $(flags) $(inc) $^ -o $@
%.o:%.cu
	$(CC) -c $(flags) $(inc)  $^ -o $@

.PHONY: clean
clean:
	rm -rf cuGemmProf *.obj *.o *.exe *.pdb
