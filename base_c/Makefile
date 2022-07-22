DEBUG       = no
REAL        = simple # {simple, double}

# Program flags
PREAL = REAL

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = icpx
CFLAGS = -lm -D${PREAL} -DATLAS -O2 -no-vec -qmkl -I"${MKLROOT}/include"
CLINK = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L${TBBROOT}/lib/intel64/gcc4.8 -ltbb -lstdc++ -lpthread -lm -ldl

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

nmf:
	$(CC) $(CLINK) -o nmf main.cpp $(CFLAGS)

run: nmf
	./nmf ..data/V.bin 5000 38 2 1 10

.PHONY: clean
clean:
	rm -f nmf \
  solution-NMFLeeSeung_* \
  *.o
