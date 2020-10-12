OPTIMIZE    = yes
DEBUG       = no
DEVICE      = cpu # {cpu, nvidia, igpu}
KERNEL      = bare # {bare, blas, cublas}


# Program flags
PDEVICE = CPU_DEVICE
PKERNEL = BARE_KERNEL

ifeq ($(DEVICE), nvidia)
  PDEVICE = NVIDIA_DEVICE
else ifeq ($(DEVICE), igpu)
  PDEVICE = INTEL_IGPU_DEVICE
endif

ifeq ($(KERNEL), blas)
  PKERNEL = BLAS_KERNEL
else ifeq ($(KERNEL), cublas)
  ifeq ($(DEVICE), nvidia)
    PKERNEL = CUBLAS_KERNEL
  endif
endif


# Standard flags
CC = clang++
CFLAGS := -std=c++11 -Wall -fsycl -D$(PDEVICE) -D$(PKERNEL)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g  -DOUTPUT
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


nmf: nmf_LeeSeung.o ./kernels/bare_kernel/bare_kernel.o ./kernels/common.o
	$(CC) $(CFLAGS) nmf_LeeSeung.o \
    ./kernels/bare_kernel/bare_kernel.o \
    ./kernels/common.o \
    -o nmf

nmf_LeeSeung.o: ./kernels/bare_kernel/bare_kernel.o ./kernels/common.o nmf_LeeSeung.cpp
	$(CC) $(CFLAGS) nmf_LeeSeung.cpp -c -o nmf_LeeSeung.o

./kernels/bare_kernel/bare_kernel.o: ./kernels/bare_kernel/bare_kernel.cpp \
  ./kernels/bare_kernel/bare_kernel.h \
  ./kernels/common.o
	$(CC) $(CFLAGS) ./kernels/bare_kernel/bare_kernel.cpp -c \
    -o ./kernels/bare_kernel/bare_kernel.o

./kernels/common.o: ./kernels/common.cpp ./kernels/common.h
	$(CC) $(CFLAGS) ./kernels/common.cpp -c \
    -o ./kernels/common.o


run: nmf
	./nmf V.bin 500 128 4 100 20

.PHONY: clean
clean:
	rm -f *.o *~ nmf *.linkinfo solution-NMFLeeSeung_*