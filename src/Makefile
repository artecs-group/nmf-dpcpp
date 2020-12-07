OPTIMIZE    = yes
DEBUG       = no
DEVICE      = cpu # {cpu, nvidia, igpu}
KERNEL      = bare # {bare, blas}
REAL        = yes

CPU = CPU_DEVICE
IGPU = INTEL_IGPU_DEVICE
NVIDIA = NVIDIA_DEVICE

# Program flags
PDEVICE = $(CPU)
PKERNEL = BARE_KERNEL
PREAL = REAL

MKL_CXXFLAGS = -I$(MKLROOT)/include -DMKL_ILP64
MKL_LDFLAGS = ${MKLROOT}/lib/intel64/libmkl_sycl.a -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lOpenCL -ldl

ifeq ($(DEVICE),nvidia)
  PDEVICE = NVIDIA_DEVICE
else ifeq ($(DEVICE), igpu)
  PDEVICE = $(IGPU)
endif

ifeq ($(KERNEL),blas)
  PKERNEL = BLAS_KERNEL
endif

ifeq ($(REAL),no)
  PREAL = NON_REAL
endif

# Default compiler and flags
CC = clang++

ifeq ($(CC),dpcpp)
  CFLAGS = -D$(PDEVICE) -D$(PKERNEL) -D$(PREAL)
else
  CFLAGS := -std=c++11 -Wall -fsycl -D$(PDEVICE) -D$(PKERNEL) -D$(PREAL)
endif

# Intel DPC++ Compiler
ifeq ($(CC), dpcpp)
  CC = dpcpp

  # nvidia gpu is not supported in dpcpp
  ifeq ($(PDEVICE), $(NVIDIA))
    PDEVICE = $(IGPU)
  endif
  CFLAGS = -D$(PDEVICE) -D$(PKERNEL) -D$(PREAL)
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


nmf: nmf_LeeSeung.o \
  ./kernels/bare_kernel/bare_kernel.o \
  ./kernels/blas_kernel/blas_kernel.o \
  ./kernels/common.o
	$(CC) $(MKL_CXXFLAGS) $(CFLAGS) nmf_LeeSeung.o \
    ./kernels/bare_kernel/bare_kernel.o \
    ./kernels/blas_kernel/blas_kernel.o \
    ./kernels/common.o \
    -o nmf $(MKL_LDFLAGS)

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