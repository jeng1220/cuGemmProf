CC:=nvcc
exe:=cuGemmProf
obj:=cuGemmProf.o verify.o
inc:=-I./cxxopts/include -I/usr/local/cuda/include
lib:=-L/usr/local/cuda/lib64 -lcublasLt -lcublas -lcudart
flags:=-O3 -std=c++11

all:$(obj)
	$(CC) $(lib) $(obj) -o $(exe)
%.o:%.cpp
	$(CC) -c $(flags) $(inc) $^ -o $@
%.o:%.cu
	$(CC) -c $(flags) $(inc)  $^ -o $@

.PHONY: clean
clean:
	rm -rf cuGemmProf *.obj *.o *.exe *.pdb
