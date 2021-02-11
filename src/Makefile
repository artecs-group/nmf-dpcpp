OPTIMIZE    = yes
DEBUG       = no
REAL        = simple # {simple, double}

PREAL = REAL_S
VAR_N = 500
VAR_M = 100
VAR_K = 4

MKL_LDFLAGS = -I${MKLROOT}/include -mkl -fsycl-device-code-split=per_kernel

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = dpcpp

ifeq ($(CC),dpcpp)
  CFLAGS = -D$(PREAL) -DVAR_N=$(VAR_N) -DVAR_M=$(VAR_M) -DVAR_K=$(VAR_K)
else
  CFLAGS := -std=c++11 -Wall -fsycl -D$(PREAL) -DVAR_N=$(VAR_N) -DVAR_M=$(VAR_M) -DVAR_K=$(VAR_K)
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
	$(CC) $(CFLAGS) nmf_LeeSeung.o \
    ./kernels/bare_kernel/bare_kernel.o \
    ./kernels/blas_kernel/blas_kernel.o \
    ./kernels/common.o \
    -o nmf $(MKL_LDFLAGS)

nmf_LeeSeung.o: ./kernels/bare_kernel/bare_kernel.o \
  ./kernels/blas_kernel/blas_kernel.o \
  ./kernels/common.o \
  nmf_LeeSeung.cpp
	$(CC) $(CFLAGS) nmf_LeeSeung.cpp -c -o nmf_LeeSeung.o $(MKL_LDFLAGS)

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
	./nmf V.bin 100 20

.PHONY: clean
clean:
	rm -f nmf solution-NMFLeeSeung_* \
  *.o \
  ./kernels/*.o \
  ./kernels/bare_kernel/*.o \
  ./kernels/blas_kernel/*.o