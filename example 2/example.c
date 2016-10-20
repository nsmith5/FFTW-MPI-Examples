#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "mpi.h"
#include "complex.h"

#include "fftw3-mpi.h"

#define Nx 512
#define Ny 512 

#define Dt 0.1

int epsout(double *in,int mpi_rank,int mpi_size,int alloc_local,int local_size,int epstag);
int msgout(double *in,int mpi_rank,int mpi_size,int local_size,int epstag);
int set_init_field(double *in,ptrdiff_t local_start);

int main(int argc, char **argv){
	double *r_p;
	double pi,tmp1;
	fftw_complex *k_p;
	fftw_plan plan1f,plan1b,plan_tp;
	ptrdiff_t local_size,local_start,alloc_local;
	ptrdiff_t local_size_t,local_start_t,alloc_local_t;
	int mpi_rank,mpi_size,time;

	int i,j,k,l;

	pi=4.0*atan(1.0);

	MPI_Init(&argc, &argv);
        fftw_mpi_init();

	MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);

	if(mpi_rank==0){printf("MPI initialized\n");}
	//get local data size
	alloc_local = fftw_mpi_local_size_2d(Nx, Ny/2+1, 
                                         MPI_COMM_WORLD,
                                         &local_size, 
                                         &local_start);
	
	alloc_local_t = fftw_mpi_local_size_2d_transposed(Nx, Ny/2+1, 
                                                      MPI_COMM_WORLD,
                                                      &local_size, 
                                                      &local_start,
                                                      &local_size_t,
                                                      &local_start_t);
	
	//real data is twice the size of complex data
    r_p = (double*)fftw_malloc(2*(alloc_local)*sizeof(double));
	k_p = (fftw_complex*)fftw_malloc(alloc_local*sizeof(fftw_complex));

	if(mpi_rank==0){
                printf("Memory allocated: alloc_local=%d, local_size=%d alloc_local_t=%d, local_size_t=%d\n",alloc_local,local_size,alloc_local_t,local_size_t);
        }

	//!!!always create plan first before initialize input
	plan1f = fftw_mpi_plan_dft_r2c_2d(Nx,Ny,r_p,k_p,MPI_COMM_WORLD,FFTW_MEASURE);
	plan1b = fftw_mpi_plan_dft_c2r_2d(Nx,Ny,k_p,r_p,MPI_COMM_WORLD,FFTW_MEASURE);

	if(mpi_rank==0){
		printf("plan created\n");
	}

	//initialize real space data
	for(i=0;i<local_size;i++){
		for(j=0;j<Ny;j++){
			//test for simple diffusion
			if(pow(i+local_size*mpi_rank-32,2.0)+pow(j-32,2.0)<25){ 
				r_p[i*2*(Ny/2+1)+j]=1;
			}else{
				r_p[i*2*(Ny/2+1)+j]=0;
			}
		}
	}
	if(mpi_rank==0){
		printf("array initialized\n");
	}

	// epsout(r_p,mpi_rank,mpi_size,alloc_local,local_size,0);	

	if(mpi_rank==0){
		printf("starting plan...\n");
	}

	// Start the clock
	MPI_Barrier (MPI_COMM_WORLD);
	double t1 = MPI_Wtime ();
	
	for(time=0;time<1000;time++){
		fftw_execute(plan1f);//forward transform
		//apply equation in fourier space for all frequencies
		for(k=0;k<local_size;k++){
			for(l=0;l<Ny/2+1;l++){
				tmp1=k_p[k*(Ny/2+1)+l];
				//////////////////////////////////////////
				//Nyquist frequency is based on local data
				//use k and k-local_size
				//////////////////////////////////////////
				if((k+mpi_rank*local_size)<Nx/2) k_p[k*(Ny/2+1)+l]=-(pow(2.0*pi*l/Ny,2)+pow(2.0*pi*(k+mpi_rank*local_size)/Nx,2))*tmp1*Dt+tmp1;
                                if((k+mpi_rank*local_size)>=Nx/2) k_p[k*(Ny/2+1)+l]=-(pow(2.0*pi*l/Ny,2)+pow(2.0*pi*(k+mpi_rank*local_size-Nx)/Nx,2))*tmp1*Dt+tmp1;
			}
		}
		
		fftw_execute(plan1b);
	
		for(i=0;i<local_size;i++){
			for(j=0;j<Ny;j++){
				r_p[i*2*(Ny/2+1)+j]/=(Nx*Ny);
	              		//normalization factor for FFT
		        }
		}
	}
	
	// Stop the clock 	
	MPI_Barrier (MPI_COMM_WORLD);
	double t2 = MPI_Wtime ();
	
	// Print out time
	if (mpi_rank == 0) printf("Loop took %gs with %d procs\n", t2-t1, mpi_size);

	//epsout(r_p,mpi_rank,mpi_size,alloc_local,local_size,1);
	
	fftw_destroy_plan(plan1f);
	fftw_destroy_plan(plan1b);
	fftw_free(r_p);
	fftw_free(k_p);
	fftw_mpi_cleanup();
	MPI_Finalize();
	return 0;
}

int epsout(double *in,int mpi_rank,int mpi_size,int alloc_local,int local_size,int epstag)
{
	FILE *file1;
	int i,j,k;
	char eps_name[80];
	double *field_all,*recv_buf;

        MPI_Status tmp_sta;

        if(mpi_rank==0){
                sprintf(eps_name,"d_mpi.%d.dat",epstag);
                file1=fopen(eps_name,"w");
		if(file1==NULL){
			printf("cannot create file %s\n",eps_name);
			return 0;
		}
                field_all=(double*)malloc(Nx*2*(Ny/2+1)*sizeof(double));
		recv_buf=(double*)malloc(2*alloc_local*sizeof(double));
                for(k=0;k<mpi_size;k++){
                        if(k>0){
                                MPI_Recv(recv_buf,alloc_local*2,MPI_DOUBLE,k,0,MPI_COMM_WORLD,&tmp_sta);
                        }else{
                                memcpy(recv_buf,in,alloc_local*2*sizeof(double));
                        }
                        for(i=0;i<local_size;i++){
                                for(j=0;j<Ny;j++){
                                        field_all[(i+local_size*k)*2*(Ny/2+1)+j]=recv_buf[i*2*(Ny/2+1)+j];
                                }
                        }
        	}
       		for(i=0;i<Nx;i++){
                	for(j=0;j<Ny;j++){
                       		fprintf(file1,"%d %d %f\n",i,j,field_all[i*2*(Ny/2+1)+j]);
                	}
                	fprintf(file1,"\n");
        	}
        	free(field_all);
		free(recv_buf);
        	fclose(file1);
        }
        else
    	{
        	MPI_Send(in,alloc_local*2,MPI_DOUBLE,0,0,MPI_COMM_WORLD);

    }

	return 1;

}
