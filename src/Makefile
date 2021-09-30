OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu # {cpu, gpu}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
GPU = GPU_DEVICE

# Program flags
PDEVICE = $(GPU)
PREAL = REAL_S

ifeq ($(DEVICE),cpu)
  PDEVICE = CPU_DEVICE
endif

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = icpx
CLINK = -fiopenmp -fopenmp-targets=spir64 -fsycl -qmkl
CFLAGS = -D$(PDEVICE) -D$(PREAL)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

main: main.o
	$(CC) $(CFLAGS) $(CLINK) main.o -o main

main.o: ./main.cpp ./common.h
	$(CC) $(CFLAGS) $(CLINK) main.cpp -c -o main.o

run: main
	./main V.bin 5000 38 2 1 20

.PHONY: clean
clean:
	rm -f main solution-NMFLeeSeung_* *.o ./nmf/*.o