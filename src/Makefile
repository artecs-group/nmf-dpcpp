OPTIMIZE    = yes
DEBUG       = no
DEVICE      = cpu # {cpu, nvidia, igpu}
KERNEL      = bare # {bare, blas}
REAL        = yes

# Program flags
PDEVICE = CPU_DEVICE
PKERNEL = BARE_KERNEL
PREAL = REAL

ifeq ($(DEVICE), nvidia)
  PDEVICE = NVIDIA_DEVICE
else ifeq ($(DEVICE), igpu)
  PDEVICE = INTEL_IGPU_DEVICE
endif

ifeq ($(KERNEL), blas)
  PKERNEL = BLAS_KERNEL
endif

ifeq ($(REAL), no)
  PREAL = NON_REAL
endif

# Standard flags
CC = clang++
CFLAGS := -std=c++11 -Wall -fsycl –L$ONEMKL/lib –lonemkl -D$(PDEVICE) -D$(PKERNEL) -D$(PREAL)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g  -DOUTPUT
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


nmf: nmf_LeeSeung.o \
  ./kernels/bare_kernel/bare_kernel.o \
  ./kernels/blas_kernel/blas_kernel.o \
  ./kernels/common.o
	$(CC) $(CFLAGS) nmf_LeeSeung.o \
    ./kernels/bare_kernel/bare_kernel.o \
    ./kernels/blas_kernel/blas_kernel.o \
    ./kernels/common.o \
    -o nmf

nmf_LeeSeung.o: ./kernels/bare_kernel/bare_kernel.o \
  ./kernels/blas_kernel/blas_kernel.o \
  ./kernels/common.o \
  nmf_LeeSeung.cpp
	$(CC) $(CFLAGS) nmf_LeeSeung.cpp -c -o nmf_LeeSeung.o

./kernels/blas_kernel/blas_kernel.o: ./kernels/blas_kernel/blas_kernel.cpp \
  ./kernels/blas_kernel/blas_kernel.h \
  ./kernels/common.o
	$(CC) $(CFLAGS) ./kernels/blas_kernel/blas_kernel.cpp -c \
    -o ./kernels/blas_kernel/blas_kernel.o

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
	rm -f nmf solution-NMFLeeSeung_* \
  *.o \
  ./kernels/*.o \
  ./kernels/bare_kernel/*.o \
  ./kernels/blas_kernel/*.o