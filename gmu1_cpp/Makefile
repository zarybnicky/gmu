CC := mpicxx
CXX := nvcc
LDFLAGS := -Xlinker -L$(OPENCL_PATH)/lib -L$(CUDA_PATH)/lib -lcudart -lOpenCL -fopenmp
CXXFLAGS := -m64

ifeq ($(dbg),1)
  CXXFLAGS += -g -G
else
  CXXFLAGS += -lineinfo
endif

all: main

main: main.o oclHelper.o

clean:
	rm -f main main.o oclHelper.o
