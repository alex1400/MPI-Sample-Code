/***********************

Conway Game of Life

************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int bwidth, bheight, nsteps;
long int i, j, n, im, ip, jm, jp, ni, nj, nsum, last, next, prev, mynum, rr;
long int isum;
int myrank, size;
int rows;
int *board, *old, *new, *count, *disp;
float x;
struct timeval start;
struct timeval end;
double rtime, max, *all_rtime;
MPI_Request rqst[4];
MPI_Status stat[4];


//Left-Right Boundary exchange
void L_R_Exchange(int n)
{	
	for (j = 0; j <= n+1; j++)
		{
			old[j*nj] = old[j*nj + bheight];
			old[j*nj + bheight + 1] = old[j*nj + 1];
		}
}

/*Send and Receive the first and last rows*/
void Snd_Rcv_Boundaries(int n)
{
	MPI_Isend(&old[nj + 1], bheight, MPI_INT, prev, 1, MPI_COMM_WORLD, &rqst[0]);
	MPI_Irecv(&old[1], bheight, MPI_INT, prev, 0, MPI_COMM_WORLD, &rqst[1]);
	MPI_Isend(&old[n*nj + 1], bheight, MPI_INT, next, 0, MPI_COMM_WORLD, &rqst[2]);
	MPI_Irecv(&old[(n + 1)*nj + 1], bheight, MPI_INT, next, 1, MPI_COMM_WORLD, &rqst[3]);
	MPI_Waitall(4, rqst, stat);
}

// update board for step n
void doTimeStep(int n)
{
	/* Determine the boundary process in the logical overlay*/
	if (myrank == 0)
	{
		prev = size - 1;
		if (size == 1)
			next = 0;
		else
			next = 1;
	}
	else if (myrank == size - 1)
	{
		prev = myrank - 1;
		next = 0;
	}
	else
	{
		prev = myrank - 1;
		next = myrank + 1;
	}
 
	// send and receive the fisrt and last rows to boundary process
	if (myrank == 0)
	{
		Snd_Rcv_Boundaries(rr);
		
		/* Left-Right boundary conditions */
		L_R_Exchange(rr);

		/* corner top boundary conditions */
		old[0] = old[bheight];
		old[bheight + 1] = old[1];
		
		/* corner bottom boundary conditions */
		if (size == 1)
		{
			old[nj*(bwidth + 1)] = old[(nj*(bwidth + 1)) + bheight];
			old[(nj*(bwidth + 1)) + bheight + 1] = old[nj * (bwidth + 1) + 1];
		}
	}

	else if (myrank == size - 1)
	{
		Snd_Rcv_Boundaries(last);

		/* Left-Right boundary conditions */
		L_R_Exchange(last);
		
	   /* corner boundary conditions */
		old[nj*(last + 1)] = old[(nj*(last + 1)) + bheight];
		old[(nj*(last + 1)) + bheight + 1] = old[nj * (last + 1) + 1];
	}
	else
	{
		Snd_Rcv_Boundaries(rr);
		
		/* Left-Right boundary conditions */
		L_R_Exchange(rr);
	}
	//Update borad
	#ifdef _OPENMP
	#pragma omp parallel for private(j,i,im,ip,jp,nsum)
	#endif
	for (i = 1; i <= rr; i++)
	{
		for ( j = 1; j <= bheight; j++)
		{
			im = i - 1;
			ip = i + 1;
			jm = j - 1;
			jp = j + 1;

			nsum = old[im*nj + jp] + old[i*nj + jp] + old[ip*nj + jp]
				+ old[im*nj + j] + old[ip*nj + j]
				+ old[im*nj + jm] + old[i*nj + jm] + old[ip*nj + jm];

			switch (nsum)
			{
				// a new organism is born
			case 3:
				new[i*nj + j] = 1;
				break;
				// nothing happens
			case 2:
				new[i*nj + j] = old[i*nj + j];
				break;
				// the oranism, if any, dies
			default:
				new[i*nj + j] = 0;
			}
		}
	}
	
	/* copy new state into old state */
	#ifdef _OPENMP
	#pragma omp parallel for private(j,i)
	#endif
	for (i = 1; i <= rr; i++)
	{
		for (j = 1; j <= bheight; j++)
		{
			old[i*nj + j] = new[i*nj + j];
		}
	}
	
}

/***************  Main Function *****************************
*************************************************************/
int main(int argc, char *argv[])
{
	// MPI Initialization
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	/* Get Parameters */

	if (argc != 4)
	{
		fprintf(stderr,
			"Usage: %s board_width board_height steps_count\n",
			argv[0]);
		MPI_Finalize();
		exit(1);
	}

	bwidth = atoi(argv[1]);
	bheight = atoi(argv[2]);
	nsteps = atoi(argv[3]);
	/* allocate arrays */
	nj = bheight + 2; 	/* add 2 for left and right ghost cells */
	
	//Get the number of nodes 
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	//Determine the number of lines which should be sent to each node.
	rr = (int)ceil(bwidth / size* 1.0); //rr stands for "real row"
	rows = rr + 2; //rows shows the number of allocated rows in each node
	
	// last shows the number of allocated rows to the last node
	last = bwidth - (rr*(size - 1));  

	if (myrank==size-1) // The last node in the overlay network probably receive a different number of rows.
	{
		old = (int *)malloc((last+2) * nj *sizeof(int)); 
		new = (int *)malloc((last+2) * nj *sizeof(int));
	} 
	else
	{
	    old = (int *)malloc(rows * nj *sizeof(int)); 
		new = (int *)malloc(rows * nj *sizeof(int));
	}

	// Root node make the board
	if (myrank == 0)
	{
		board = (int *)malloc(bwidth* nj *sizeof(int));
		/*  initialize board */
		for (i = 0; i < bwidth; i++)
		{
			
			for (j = 1; j <= bheight; j++)
			{
				x = rand() / ((float)RAND_MAX + 1);
				if (x<0.5)
				{
					board[i*nj + j] = 0;
				}
				else
				{
					board[i*nj + j] = 1;
				}
			}
		}
		// Initializing the the necessary parameters to use in MPI_Scatterv function
	}
		disp = (int *)malloc(size*sizeof(int)); // Displacement in the borad
		count = (int *)malloc(size*sizeof(int)); // The number of elements 
		for (i = 0; i < size - 1; i++)
		{
			count[i] = rr*nj;
			disp[i] = i*rr*nj;
		}
		count[i] = last * nj;
		disp[i] = i*rr*nj;
		mynum = count[myrank];
	
    MPI_Scatterv(board, count, disp, MPI_INT, &old[nj], mynum, MPI_INT, 0, MPI_COMM_WORLD);
	
	//********************speecial computation*****************/
	
	if (gettimeofday(&start, 0) != 0)
		{
			fprintf(stderr, "could not do timing\n");
			MPI_Finalize();
			exit(1);
		}
	
	
	for (n = 0; n<nsteps; n++)
		{
			doTimeStep(n);
		}
	
	if (gettimeofday(&end, 0) != 0) 
		{
			fprintf(stderr, "could not do timing\n");
			MPI_Finalize();
			exit(1);
		}
		
		
		// compute running time
		rtime = (end.tv_sec + (end.tv_usec / 1000000.0)) -
			(start.tv_sec + (start.tv_usec / 1000000.0));
		

	
	//Finding maximum execution time among the nodes in the MPI_COMM_WORLD
	if(myrank==0){
		all_rtime = (double *)malloc(size *sizeof(double)); 
	}
	
	MPI_Gather(&rtime,1,MPI_DOUBLE, all_rtime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(myrank==0)
	{
		max=all_rtime[0];
		for(i=0; i<size; i++)
		{	
			if(all_rtime[i]>max)
				max=all_rtime[i];
		}
	}
	
		
	MPI_Gatherv(&old[nj], mynum, MPI_INT, board, count, disp, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(myrank==0){
	/*  Iterations are done; sum the number of live cells */
		isum = 0;
		for (i = 0; i < bwidth; i++)
			for (j = 1; j <= bheight; j++)
				isum = isum + board[(i*nj) + j];
					
		printf("Number of live cells = %ld\n", isum);
		fprintf(stderr, "Game of Life took %10.3f seconds\n", max);
		free(board);
	}
	//Free the obtained memeory
	free(disp);
	free(count);
	free(old);
	free(new);
	
	MPI_Finalize();
	return 0;
}
