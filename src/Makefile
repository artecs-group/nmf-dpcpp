OPTIMIZE    = yes
DEBUG       = no
REAL        = simple # {simple, double}

# Program flags
PREAL = REAL

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CXX = g++
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
	$(CXX) -o nmf main.o ./kernels/kernels.o $(CFLAGS) $(CLINK)

main.o: ./kernels/kernels.o main.cpp
	$(CXX) -fpermissive $(CFLAGS) main.cpp -c -o main.o

./kernels/kernels.o: ./kernels/kernels.cu ./kernels/kernels.h
	$(NVCC) $(CFLAGS) ./kernels/kernels.cu -c -o ./kernels/kernels.o


run: nmf
	./nmf V.bin 5000 38 2 1 10

.PHONY: clean
clean:
	rm -f nmf \
  solution-NMFLeeSeung_* \
  *.o \
  ./kernels/kernels.o
