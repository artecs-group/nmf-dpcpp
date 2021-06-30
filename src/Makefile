OPTIMIZE    = yes
DEBUG       = no
KERNEL      = bare # {bare, blas}
REAL        = simple # {simple, double}

# Program flags
PKERNEL = BARE_KERNEL
PREAL = REAL_S

MKL_LDFLAGS = -I${MKLROOT}/include -qmkl -fsycl-device-code-split=per_kernel

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

ifeq ($(KERNEL),blas)
  PKERNEL = BLAS_KERNEL
endif

# Default compiler and flags
CC = dpcpp
CFLAGS = -D$(PREAL) -D$(PKERNEL)

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
  queue_data.o \
  ./kernels/common.o
	$(CC) $(CFLAGS) nmf_LeeSeung.o \
    ./kernels/bare_kernel/bare_kernel.o \
    ./kernels/blas_kernel/blas_kernel.o \
    queue_data.o \
    ./kernels/common.o \
    -o nmf $(MKL_LDFLAGS)

nmf_LeeSeung.o: ./kernels/bare_kernel/bare_kernel.o \
  ./kernels/blas_kernel/blas_kernel.o \
  ./kernels/common.o \
  queue_data.o \
  nmf_LeeSeung.cpp
	$(CC) $(CFLAGS) nmf_LeeSeung.cpp -c -o nmf_LeeSeung.o $(MKL_LDFLAGS)

queue_data.o: ./kernels/common.o
	$(CC) $(CFLAGS) queue_data.cpp -c -o queue_data.o $(MKL_LDFLAGS)

./kernels/blas_kernel/blas_kernel.o: ./kernels/blas_kernel/blas_kernel.cpp \
  ./kernels/blas_kernel/blas_kernel.h \
  ./kernels/common.o
	$(CC) $(CFLAGS) ./kernels/blas_kernel/blas_kernel.cpp -c \
    -o ./kernels/blas_kernel/blas_kernel.o $(MKL_LDFLAGS)

./kernels/bare_kernel/bare_kernel.o: ./kernels/bare_kernel/bare_kernel.cpp \
  ./kernels/bare_kernel/bare_kernel.h \
  ./kernels/common.o
	$(CC) $(CFLAGS) ./kernels/bare_kernel/bare_kernel.cpp -c \
    -o ./kernels/bare_kernel/bare_kernel.o $(MKL_LDFLAGS)

./kernels/common.o: ./kernels/common.cpp ./kernels/common.h
	$(CC) $(CFLAGS) ./kernels/common.cpp -c \
    -o ./kernels/common.o $(MKL_LDFLAGS)


run: nmf
	./nmf V.bin 5000 38 2 1 20

.PHONY: clean
clean:
	rm -f nmf solution-NMFLeeSeung_* \
  *.o \
  ./kernels/*.o \
  ./kernels/bare_kernel/*.o \
  ./kernels/blas_kernel/*.o