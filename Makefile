CC = g++
NVCC = nvcc
CFLAGS = -O3 -msse3 -ffast-math -fomit-frame-pointer -Wall -fopenmp
NVCC_FLAGS = -O3
INDENT = astyle

INDENT_OPTS=-n --style=allman --indent-classes --indent-switches --indent-cases --indent-namespaces --indent-labels -Y --min-conditional-indent=2 --max-instatement-indent=40 --pad-oper --unpad-paren --mode=c --add-brackets --break-closing-brackets --align-pointer=type

all: PQ_predict_cpu PQ_predict_gpu

PQ_predict_cpu: PQ_predict_cpu.cpp
	$(CC) $(CFLAGS) -o $@ PQ_predict_cpu.cpp

PQ_predict_gpu: PQ_predict_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ PQ_predict_gpu.cu

indent:
	$(INDENT) $(INDENT_OPTS) *.cu

clean:
	rm PQ_predict_cpu PQ_predict_gpu
