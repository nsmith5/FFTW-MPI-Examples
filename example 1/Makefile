CC = mpicc
CFLAGS= -Wall
LIBS= -lfftw3_mpi -lfftw3 -lm

example: example.c example.o
	$(CC) example.o $(CFLAGS) $(LIBS) -o example

example.o: example.c
	$(CC) -c example.c $(CFLAGS) -o example.o

.PHONY: clean

clean:
	rm example.o
	rm example
