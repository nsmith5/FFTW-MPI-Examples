CC = mpicc
CFLAGS = -Wall
LIBS = -lfftw3_mpi -lfftw3 -lm

modela: modela.c modela.o
	$(CC) modela.o $(CFLAGS) $(LIBS) -o modela

modela.o: modela.c
	$(CC) -c modela.c $(CFLAGS) -o modela.o 

.PHONY: clean

clean:
	rm modela.o
	rm modela
