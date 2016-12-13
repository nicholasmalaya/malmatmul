IDIR = ../MatRoutines
CC=gcc
#CC=hcc
CFLAGS= -O3 -g -I$(IDIR)
LIBS= -lm

VPATH = ../MatRoutines

_DEPS = Mat.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))
 
_OBJ = Mat.o fom_timer.o  matmul.o malmatmul.o
OBJ = $(patsubst %,./%,$(_OBJ)) 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

malmatmul: $(OBJ)
	hcc -o $@ $^ $(CFLAGS)

.PHONY: clean $(IDIR)

subdirs:
	for dir in $(IDIR); do \
		$(MAKE) -C $$dir; \
	done


clean:
	rm malmatmul *.o *~

