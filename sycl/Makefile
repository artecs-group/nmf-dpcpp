OPTIMIZE    = yes
DEBUG       = no
DEVICE      = cpu # {cpu, igpu, nvidia}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
IGPU = INTEL_GPU_DEVICE
NVIDIA = NVIDIA_DEVICE

# Program flags
PDEVICE = $(CPU)
PREAL = REAL_S

ifeq ($(DEVICE), igpu)
  PDEVICE = $(IGPU)
else ifeq ($(DEVICE), nvidia)
  PDEVICE = $(NVIDIA)
endif

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = dpcpp
CFLAGS = -D$(PDEVICE) -D$(PREAL) -qmkl -DMKL_ILP64 -I"${MKLROOT}/include"
CLINK = -fsycl-device-code-split=per_kernel ${MKLROOT}/lib/intel64/libmkl_sycl.a -Wl,-export-dynamic -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L${TBBROOT}/lib/intel64/gcc4.8 -ltbb -lsycl -lOpenCL -lpthread -lm -ldl

ifeq ($(PDEVICE), $(NVIDIA))
  CC = clang++
  CFLAGS = -D$(PDEVICE) -D$(PREAL) -w -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I${ONEMKL}/include
  CLINK = -fsycl-device-code-split=per_kernel -L${ONEMKL}/lib -lonemkl
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

nmf: main.o ./kernels/kernels.o
	$(CC) $(CFLAGS) $(CLINK) main.o ./kernels/kernels.o -o nmf

main.o: ./kernels/kernels.o main.cpp common.hpp
	$(CC) $(CFLAGS) main.cpp -c -o main.o

./kernels/kernels.o: ./kernels/kernels.cpp ./kernels/kernels.hpp
	$(CC) $(CFLAGS) ./kernels/kernels.cpp -c -o ./kernels/kernels.o


run: nmf
	./nmf ../data/V.bin 5000 38 2 1 20

.PHONY: clean
clean:
	rm -f nmf \
  solution-NMFLeeSeung_* \
  *.o \
  ./kernels/kernels.o
