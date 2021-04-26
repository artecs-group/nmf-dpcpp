OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu # {cpu, gpu}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
GPU = GPU_DEVICE

# Program flags
PDEVICE = $(GPU)
PREAL = REAL_S

MKL_LDFLAGS = -I${MKLROOT}/include -qmkl

ifeq ($(DEVICE),cpu)
  PDEVICE = CPU_DEVICE
endif

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = dpcpp
CFLAGS = -D$(PDEVICE) -D$(PREAL) -fiopenmp -fopenmp-targets=spir64

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

main: main.o ./nmf/nmf.o
	$(CC) $(CFLAGS) main.o ./nmf/nmf.o -o main $(MKL_LDFLAGS)

main.o: ./nmf/nmf.o ./main.cpp 
	$(CC) $(CFLAGS) main.cpp -c -o main.o $(MKL_LDFLAGS)

./nmf/nmf.o: ./common.h ./nmf/nmf.cpp
	$(CC) $(CFLAGS) ./nmf/nmf.cpp -c -o ./nmf/nmf.o $(MKL_LDFLAGS)

run: main
	./main V.bin 2000 62 4 100 20

.PHONY: clean
clean:
	rm -f main solution-NMFLeeSeung_* *.o ./nmf/*.o