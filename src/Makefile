OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu # {cpu, gpu}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
GPU = GPU_DEVICE

# Program flags
PDEVICE = $(GPU)
PREAL = REAL_S

MKL_LDFLAGS = -I${MKLROOT}/include -qmkl

ifeq ($(DEVICE),cpu)
  PDEVICE = CPU_DEVICE
endif

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = dpcpp
CLINK = -fsycl-device-code-split=per_kernel -DMKL_ILP64 ${MKLROOT}/lib/intel64/libmkl_sycl.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lsycl -lOpenCL -lpthread -ldl -lm
CFLAGS = -D$(PDEVICE) -D$(PREAL) -fiopenmp -fopenmp-targets=spir64

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

main: main.o ./nmf/nmf.o
	$(CC) $(CFLAGS) $(CLINK) main.o ./nmf/nmf.o -o main $(MKL_LDFLAGS)

main.o: ./nmf/nmf.o ./main.cpp 
	$(CC) $(CFLAGS) main.cpp -c -o main.o $(MKL_LDFLAGS)

./nmf/nmf.o: ./common.h ./nmf/nmf.cpp
	$(CC) $(CFLAGS) ./nmf/nmf.cpp -c -o ./nmf/nmf.o $(MKL_LDFLAGS)

run: main
	./main V.bin 2000 62 4 100 20

.PHONY: clean
clean:
	rm -f main solution-NMFLeeSeung_* *.o ./nmf/*.o