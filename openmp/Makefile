OPTIMIZE    = yes
DEBUG       = no
DEVICE      = igpu # {cpu, igpu, nvidia}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
IGPU = INTEL_GPU_DEVICE
NVIDIA = NVIDIA_GPU_DEVICE

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

ifeq ($(DEVICE),igpu)
  PDEVICE = $(IGPU)
  CLINK = -fiopenmp -fopenmp-targets=spir64 -fsycl -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_sycl.a ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lsycl -lOpenCL -lstdc++ -lpthread -lm -ldl
  CFLAGS += -DGPU_DEVICE -fiopenmp -fopenmp-targets=spir64
else ifeq ($(DEVICE),nvidia)
  PDEVICE = $(NVIDIA)
  CC = nvc++
  CFLAGS = -DGPU_DEVICE -D$(PDEVICE) -D$(PREAL) -I/usr/local/cuda/include -lcuda -lcublas -fopenmp -mp=gpu
  CLINK  = -L/usr/local/cuda/lib64 -lcudart
else
  CFLAGS += -xhost # set: -march=core-avx2 for AMD processor
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

nmf: main.o
	$(CC) $(CFLAGS) $(CLINK) main.o -o nmf 

main.o: ./main.cpp ./common.h
	$(CC) $(CFLAGS) main.cpp -c -o main.o

run: nmf
	./nmf ../data/V.bin 5000 38 2 1 20

.PHONY: clean
clean:
	rm -f nmf solution-NMFLeeSeung_* *.o ./nmf/*.o
