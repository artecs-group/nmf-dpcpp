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
CFLAGS := -std=c++11 -Wall -fsycl -fsycl-unnamed-lambda -D$(PDEVICE) -D$(PKERNEL)

# Flags for nvidia gpu
ifneq (,$(findstring nvidia, $(DEVICE)))
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda-sycldevice
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g  -DOUTPUT
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


nmf: nmf_LeeSeung.o ./kernels/bare_kernel/bare_kernel.o ./kernels/common.o
	$(CC) $(CFLAGS) -o nmf nmf_LeeSeung.o ./kernels/bare_kernel/bare_kernel.o ./kernels/common.o

nmf_LeeSeung.o: ./kernels/bare_kernel/bare_kernel.o ./kernels/common.o nmf_LeeSeung.cpp
	$(CC) $(CFLAGS) -c nmf_LeeSeung.cpp

./kernels/bare_kernel/bare_kernel.o: ./kernels/bare_kernel/bare_kernel.cpp \
  ./kernels/bare_kernel/bare_kernel.h \
  ./kernels/common.o
	$(CC) $(CFLAGS) -c ./kernels/bare_kernel/bare_kernel.cpp

./kernels/common.o: ./kernels/common.cpp ./kernels/common.h
	$(CC) $(CFLAGS) -c ./kernels/common.cpp



run: nmf
	./nmf V.bin 500 128 4 100 20

.PHONY: clean
clean:
	rm -f *.o *~ nmf *.linkinfo