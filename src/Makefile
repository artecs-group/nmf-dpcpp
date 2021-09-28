OPTIMIZE    = yes
DEBUG       = no
DEVICE      = cpu # {cpu, nvidia, igpu}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
IGPU = INTEL_IGPU_DEVICE
NVIDIA = NVIDIA_DEVICE

# Program flags
PDEVICE = $(CPU)
PREAL = REAL_S

MKL_LDFLAGS = -I${MKLROOT}/include -qmkl -fsycl-device-code-split=per_kernel

ifeq ($(DEVICE),nvidia)
  PDEVICE = NVIDIA_DEVICE
else ifeq ($(DEVICE), igpu)
  PDEVICE = $(IGPU)
endif

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = dpcpp

ifeq ($(CC),dpcpp)
  CFLAGS = -D$(PDEVICE) -D$(PREAL)
else
  CFLAGS := -std=c++11 -Wall -fsycl -D$(PDEVICE) -D$(PREAL)
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
	$(CC) $(CFLAGS) main.o ./kernels/kernels.o -o nmf $(MKL_LDFLAGS)

main.o: ./kernels/kernels.o main.cpp common.hpp
	$(CC) $(CFLAGS) main.cpp -c -o main.o $(MKL_LDFLAGS)

./kernels/kernels.o: ./kernels/kernels.cpp ./kernels/kernels.hpp
	$(CC) $(CFLAGS) ./kernels/kernels.cpp -c -o ./kernels/kernels.o $(MKL_LDFLAGS)


run: nmf
	./nmf V.bin 5000 38 2 1 20

.PHONY: clean
clean:
	rm -f nmf \
  solution-NMFLeeSeung_* \
  *.o \
  ./kernels/kernels.o