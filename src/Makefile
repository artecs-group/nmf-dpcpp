OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu # {cpu, gpu}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
GPU = GPU_DEVICE

# Program flags
PDEVICE = $(CPU)
PREAL = REAL_S

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = icpx
CFLAGS = -D$(PDEVICE) -D$(PREAL) -I"${MKLROOT}/include" -qmkl

CLINK = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

ifeq ($(DEVICE),gpu)
  PDEVICE = $(GPU)
  CLINK = -fiopenmp -fopenmp-targets=spir64 -fsycl -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_sycl.a ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lsycl -lOpenCL -lstdc++ -lpthread -lm -ldl
  CFLAGS += -fiopenmp -fopenmp-targets=spir64
else
  CFLAGS += -xhost
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

main: main.o
	$(CC) $(CFLAGS) $(CLINK) main.o -o main 

main.o: ./main.cpp ./common.h
	$(CC) $(CFLAGS) main.cpp -c -o main.o

run: main
	./main V.bin 5000 38 2 1 20

.PHONY: clean
clean:
	rm -f main solution-NMFLeeSeung_* *.o ./nmf/*.o
