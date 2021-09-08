OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu # {cpu, gpu}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
GPU = GPU_DEVICE

# Program flags
PDEVICE = $(GPU)
PREAL = REAL_S

MKL_LDFLAGS = -I${MKLROOT}/include -qmkl -DMKL_ILP64

ifeq ($(DEVICE),cpu)
  PDEVICE = CPU_DEVICE
endif

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = dpcpp
CLINK = -fiopenmp -fopenmp-targets=spir64  -fsycl  -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lsycl -lOpenCL -lstdc++ -lpthread -lm -ldl
CFLAGS = -D$(PDEVICE) -D$(PREAL) -fiopenmp -fopenmp-targets=spir64

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

main: main.o
	$(CC) $(CFLAGS) $(CLINK) main.o -o main $(MKL_LDFLAGS)

main.o: ./main.cpp ./common.h
	$(CC) $(CFLAGS) main.cpp -c -o main.o $(MKL_LDFLAGS)

run: main
	./main V.bin 5000 38 2 1 20

.PHONY: clean
clean:
	rm -f main solution-NMFLeeSeung_* *.o ./nmf/*.o