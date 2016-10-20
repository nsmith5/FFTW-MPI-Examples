//mpicc modela.c -lfftw3_mpi -lfftw3 -o modela
#include <fenv.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

int main(int argc,char **argv) { //initialize variables
    MPI_Status status;
    int myid,np,procs,Lx,Ly,Lxh,Lx2,i,j,ij,jg,pc;
    int n,no,nend,nout,neng,ntype;
    ptrdiff_t alL,lLy,lst;
    double ksq,fx,fy,ch,Sf,kx,ky,lop,lexp,dx,dy,dt,rado,rad;
    double Qf,qxo,qyo,x,y;
    double eng,enl,qx,qy,amp;
    double *p,*g,*kgf,*kpf;
    char run[4],runo[4];
    FILE *fin,*fout;
    fftw_complex *kp,*kg;
    fftw_plan planF; fftw_plan planB; fftw_plan planFg;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    fftw_mpi_init();
    
    fin=fopen("modela.in","r"); //open file modela.in and read input data from file
    fscanf(fin,"%s",run);
    
    fscanf(fin,"%i %i",&Lx,&Ly);
    fscanf(fin,"%lf %lf %lf",&dx,&dy,&dt);
    fscanf(fin,"%i %i %i %lf",&nend,&nout,&neng,&rado);
    fscanf(fin,"%i %i",&ntype,&no);
    fclose(fin);
    if(myid==0){
        printf("dx, dy= %f %f \n", dx,dy);
        printf("nend, nout, neng, rado=%i %i %i %f\n",nend,nout,neng,rado);
        printf("run = %s \n",run);

    }
    
    Lxh=Lx/2+1; Lx2=Lx+2; Sf=1./(Lx*Ly); //calculate various things
    fy=2.*acos(-1.0)/(dy*Ly); fx=2.*acos(-1.0)/(Lx*dx);
  
    alL=fftw_mpi_local_size_2d // set up arrays stuff for FTs
    (Ly,Lx/2+1,MPI_COMM_WORLD,&lLy,&lst);
    p=(double*) fftw_malloc(sizeof(double)*2*alL);
    g=(double*) fftw_malloc(sizeof(double)*2*alL);
    kp=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*alL);
    kg=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*alL);
    kpf=(double*) fftw_malloc(sizeof(double)*alL);
    kgf=(double*) fftw_malloc(sizeof(double)*alL);
    
    planF=fftw_mpi_plan_dft_r2c_2d
    (Ly,Lx,p,kp,MPI_COMM_WORLD,FFTW_MEASURE);
    planFg=fftw_mpi_plan_dft_r2c_2d
    (Ly,Lx,g,kg,MPI_COMM_WORLD,FFTW_MEASURE);
    planB=fftw_mpi_plan_dft_c2r_2d
    (Ly,Lx,kp,p,MPI_COMM_WORLD,FFTW_MEASURE);
    
    for(j=0;j<lLy;j++) { //calculate linear (lop) and non-linear operators
        jg=j+lst;
        if(jg<Ly/2){ky=jg*fy;} else{ky=(jg-Ly)*fy;}
        for (i=0; i<Lxh; i++) {ij=i+Lxh*j;
            kx=i*fx; ksq=kx*kx+ky*ky;   //ksq=k^2
            lop=1.0-ksq;
            lexp=exp(lop*dt); kpf[ij]=lexp*Sf; //operator for linear term
            if(lop==0.0){kgf[ij]=-1.0*dt*Sf;
            } else {kgf[ij]=(lexp-1.0)/lop*Sf;}
        }
        
    }

    if(ntype==3){ //initial condition for random fluctuations
        amp=0.0;
        srand(time(NULL)+myid);
        for(j=0;j<lLy;j++){
            jg=j+lst;
            for(i=0;i<Lx;i++){ij=i+Lx2*j;
                p[ij]=0.1*(0.5-(rand() &1000)/1000.);
                amp=amp+p[ij];
        }
    }
        printf("average p= %f\n",np*amp*Sf);
    }
    printf("run,runo=%s %s \n",run,runo);
   
	
	// Grab the start time
	double t1 = MPI_Wtime ();
	 
	/*  << Main Loop Block >> */
	for(n=1;n<nend+1;n++)
	{
       
		for(j=0;j<lLy;j++)
		{
			for(i=0;i<Lx;i++)
			{
				ij=i+j*Lx2;
            	g[ij]=-p[ij]*p[ij]*p[ij];
			}
        }

        MPI_Barrier(MPI_COMM_WORLD);
        fftw_execute(planF); fftw_execute(planFg); //FT p and g
        for(j=0;j<lLy;j++){
            for(i=0;i<Lxh;i++) {ij=i+Lxh*j;
                kp[ij]=kpf[ij]*kp[ij]+kgf[ij]*kg[ij];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        fftw_execute(planB);
		
		/*    << I/O Block >>
        if(n%nout==0)
		{
			// every nout times steps print output
            if(myid==0) printf("output data at n=%i\n",n);
            char filename[BUFSIZ];
            sprintf(filename,"%s%d.dat",run,n);
            for(pc=0;pc<np;pc++)
			{
                MPI_Barrier(MPI_COMM_WORLD);
                if(myid==pc)
				{
                    if(myid==pc)
					{
                        if(myid==0) fout=fopen(filename, "w");
                        else fout = fopen(filename, "a");
                        for(j=0;j<lLy;j++)
						{
							jg=j+lst;
                            for(i=0;i<Lx;i++)
							{
                                ij=i+Lx2*j;
                                fprintf(fout,"%f %f %f \n", i*dx,jg*dy,p[ij]);
                            }
                    	}
                        fclose(fout);
                	}
            	}
            }
		}
		*/

		/*    << Energy Calculation Block >>
        if(n%neng==0)
		{ 
            char filename[BUFSIZ]; 		//energy and store in file "run".eng
            FILE *fout;
            
			for(j=0;j<lLy;j++)
				for(i=0;i<Lx;i++)
					ij=i+j*Lx2;g[ij]=p[ij];
            
			enl=0.0;
            MPI_Barrier(MPI_COMM_WORLD);
            fftw_execute(planF);
            
			for(j=0;j<lLy;j++)
			{
				jg=j+lst;
                jg=j+lst;
                if(jg<Ly/2) ky = jg * fy; 
				else ky = (jg - Ly)*fy;
                for(i=0;i<Lxh;i++)
				{ 
						ij = i + Lxh*j;
                        kx = i*fx; 
						ksq = kx*kx + ky*ky;
                        kp[ij]=0.5*kp[ij]*ksq;
                }
			}
          	MPI_Barrier(MPI_COMM_WORLD);
            fftw_execute(planB);
            for(j=0; j<lLy; j++) 
            {	
				for(i=0;i<Lx;i++) 
				{
					ij=i+Lx2*j;
                    enl=enl-0.5*g[ij]*g[ij]+0.25*g[ij]*g[ij]*g[ij]*g[ij]+Sf*p[ij]*g[ij];
                }
            }
			
			for(j=0;j<lLy;j++)
			{
				for(i=0;i<Lx;i++)
				{
					ij=i+j*Lx2;
					p[ij]=g[ij];
				}
			}
           	if(myid==0) 
		 	{
            	eng=enl;
               	for(pc=1; pc<np; pc++)
				{
                	MPI_Recv(&enl,1,MPI_LONG_DOUBLE,pc,
                             0,MPI_COMM_WORLD,&status);
                    eng=eng+enl;
              	}
                sprintf(filename,"%s.eng",run);
                if(n==neng) fout = fopen(filename,"w");
                else fout=fopen(filename,"a");
                fprintf(fout,"%f %e\n",n*dt,eng*Sf);
                fclose(fout);
            } 
			else 
			{
            	MPI_Send(&enl,1,MPI_LONG_DOUBLE,
                         0,0,MPI_COMM_WORLD);
                    
            }
        }
		*/
    }
	
	MPI_Barrier (MPI_COMM_WORLD);
	// Grab the end time
	double t2 = MPI_Wtime ();
	if (myid == 0)
	{
		printf("Loop time for %d procs is %gs\n", np, t2-t1);
   	}
	 
    MPI_Barrier(MPI_COMM_WORLD);
    fftw_destroy_plan(planF); fftw_destroy_plan(planB); fftw_destroy_plan(planFg);
    MPI_Finalize();
    }
