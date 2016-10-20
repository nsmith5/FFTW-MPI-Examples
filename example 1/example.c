#include <mpi.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 2048

int main (int argv, char **argc)
{
	int rank, size;
	double *in; 	
	fftw_complex *out;
	double t1, t2;
	fftw_plan plan;
	ptrdiff_t alloc_local, local_n0, local_0_start;

	MPI_Init (&argv, &argc);
	fftw_mpi_init ();
	
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	
	alloc_local = fftw_mpi_local_size_2d (N, N/2 + 1, 
                                          MPI_COMM_WORLD, 
                                          &local_n0,  
                                          &local_0_start);
	
	in = fftw_alloc_real (2 * alloc_local);
	out = fftw_alloc_complex (alloc_local);

	plan = fftw_mpi_plan_dft_r2c_2d (N, N, in, out, 
									 MPI_COMM_WORLD, 
                                     FFTW_MEASURE);

	// Initialize input with some numbers	
	for (int i = 0; i < local_n0; i++)
		for (int j = 0; j < N; j++)
			in[i*2*(N/2 + 1) + j] = (double)(i + j);

	// Start the clock
	MPI_Barrier (MPI_COMM_WORLD);
	t1 = MPI_Wtime ();

	// Do a fourier transform
	fftw_execute(plan);

	// Stop the clock
	MPI_Barrier (MPI_COMM_WORLD);
	t2 = MPI_Wtime ();

	// Print out how long it took in seconds
	if (rank == 0) printf("Loop time is %gs with %d procs\n", t2-t1, size);

	// Clean up and get out
	fftw_free (in);
	fftw_free (out);
	fftw_destroy_plan (plan);
	MPI_Finalize ();
	return 0;
}
