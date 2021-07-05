OPTIMIZE    = yes
DEBUG       = no
REAL        = simple # {simple, double}

# Program flags
PREAL = REAL_S

MKL_LDFLAGS = -I${MKLROOT}/include -qmkl -fsycl-device-code-split=per_kernel
CLINK = -fiopenmp -fopenmp-targets=spir64  -fsycl  -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lsycl -lOpenCL -lstdc++ -lpthread -lm -ldl

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = dpcpp
CFLAGS = -D$(PREAL)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

nmf: main.o ./queue_data/queue_data.o ./kernels/kernels.o
	$(CC) $(CFLAGS) main.o ./queue_data/queue_data.o ./kernels/kernels.o -o nmf $(MKL_LDFLAGS)

main.o: ./kernels/kernels.o ./queue_data/queue_data.o main.cpp
	$(CC) $(CFLAGS) main.cpp -c -o main.o $(MKL_LDFLAGS)

./queue_data/queue_data.o: ./queue_data/queue_data.cpp ./queue_data/queue_data.hpp
	$(CC) $(CFLAGS) ./queue_data/queue_data.cpp -c -o ./queue_data/queue_data.o $(MKL_LDFLAGS)

./kernels/kernels.o: ./kernels/kernels.cpp ./kernels/kernels.hpp
	$(CC) $(CFLAGS) ./kernels/kernels.cpp -c -o ./kernels/kernels.o $(MKL_LDFLAGS)

run: nmf
	./nmf V.bin 5000 38 2 1 20

.PHONY: clean
clean:
	rm -f nmf \
  solution-NMFLeeSeung_* \
  *.o \
  ./queue_data/queue_data.o \
  ./kernels/kernels.o