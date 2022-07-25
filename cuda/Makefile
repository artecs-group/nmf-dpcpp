OPTIMIZE    = yes
DEBUG       = no
REAL        = simple # {simple, double}

# Program flags
PREAL = REAL

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
NVCC = nvcc
CFLAGS = -D$(PREAL) -I/usr/local/cuda/include -lcuda -lcublas
CLINK = -L/usr/local/cuda/lib64 -lcudart

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

nmf: main.o ./kernels/kernels.o
	$(NVCC) -o nmf main.o ./kernels/kernels.o $(CFLAGS) $(CLINK)

main.o: ./kernels/kernels.o main.cu
	$(NVCC) $(CFLAGS) main.cu -c -o main.o

./kernels/kernels.o: ./kernels/kernels.cu ./kernels/kernels.cuh
	$(NVCC) $(CFLAGS) ./kernels/kernels.cu -c -o ./kernels/kernels.o


run: nmf
	./nmf ..data/V.bin 5000 38 2 1 10

.PHONY: clean
clean:
	rm -f nmf \
  solution-NMFLeeSeung_* \
  *.o \
  ./kernels/kernels.o
