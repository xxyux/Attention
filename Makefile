NVCC ?= nvcc
NVCCFLAGS = -O3


all: headers attention

headers:  utils.cuh attention.cuh kernel.cuh

attention: attention.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_80 -code=sm_80 attention.cu -o ./bin/attention


.PHONY : clean

clean:
	rm -f ./bin/attention 