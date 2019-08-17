/***********************

Conway Game of Life

************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int bwidth, bheight, nsteps;
int i, j, n, im, ip, jm, jp, ni, nj, nsum, isum;
int **old, **new;
float x;
FILE *out;
struct timeval start;
struct timeval end;
double rtime;

// update board for step n
void doTimeStep(int n){
	/* corner boundary conditions */
	old[0][0] = old[bwidth][bheight];
	old[0][bheight + 1] = old[bwidth][1];
	old[bwidth + 1][bheight + 1] = old[1][1];
	old[bwidth + 1][0] = old[1][bheight];

	/* left-right boundary conditions */
	for (i = 1; i <= bwidth; i++){
		old[i][0] = old[i][bheight];
		old[i][bheight + 1] = old[i][1];
	}

	/* top-bottom boundary conditions */
	for (j = 1; j <= bheight; j++){
		old[0][j] = old[bwidth][j];
		old[bwidth + 1][j] = old[1][j];
	}

	// update board
	for (i = 1; i <= bwidth; i++){
		for (j = 1; j <= bheight; j++){
			im = i - 1;
			ip = i + 1;
			jm = j - 1;
			jp = j + 1;

			nsum = old[im][jp] + old[i][jp] + old[ip][jp]
				+ old[im][j] + old[ip][j]
				+ old[im][jm] + old[i][jm] + old[ip][jm];

			switch (nsum){
			// a new organism is born
			case 3:
				new[i][j] = 1;
				break;
			// nothing happens
			case 2:
				new[i][j] = old[i][j];
				break;
			// the oranism, if any, dies
			default:
				new[i][j] = 0;
			}
		}
	}

	/* copy new state into old state */
	for (i = 1; i <= bwidth; i++){
		for (j = 1; j <= bheight; j++){
			old[i][j] = new[i][j];
		}
	}
}

int main(int argc, char *argv[]) {
	/* Get Parameters */
	if (argc != 4) {
		fprintf(stderr,
			"Usage: %s board_width board_height steps_count\n",
			argv[0]);
		exit(1);
	}
	bwidth = atoi(argv[1]);
	bheight = atoi(argv[2]);
	nsteps = atoi(argv[3]);

  /* allocate arrays */
  ni = bwidth + 2;  /* add 2 for left and right ghost cells */
  nj = bheight + 2;
  old = malloc(ni*sizeof(int*));
  new = malloc(ni*sizeof(int*));

  for(i=0; i<ni; i++){
    old[i] = malloc(nj*sizeof(int));
    new[i] = malloc(nj*sizeof(int));
  }

	/*  initialize board */
  for(i=1; i<=bwidth; i++){
    for(j=1; j<=bheight; j++){
      x = rand()/((float)RAND_MAX + 1);
      if(x<0.5){
	old[i][j] = 0;
      } else {
	old[i][j] = 1;
      }
    }
  }

	if(gettimeofday(&start, 0) != 0) {
		fprintf(stderr, "could not do timing\n");
		exit(1);
	}

  /*  time steps */
  for(n=0; n<nsteps; n++){
	  doTimeStep(n);
  }

	if(gettimeofday(&end, 0) != 0) {
		fprintf(stderr, "could not do timing\n");
		exit(1);
	}

	// compute running time
	rtime = (end.tv_sec + (end.tv_usec / 1000000.0)) -
		(start.tv_sec + (start.tv_usec / 1000000.0));

  /*  Iterations are done; sum the number of live cells */
  out=fopen("SEQ.txt","wb");
  isum = 0;
  for(i=1; i<=bwidth; i++){
    for(j=1; j<=bheight; j++){
      isum = isum + new[i][j];
	  putc(new[i][j],out);
    }
  }
  fclose(out);
  printf("Number of live cells = %d\n", isum);
	fprintf(stderr, "Game of Life took %10.3f seconds\n", rtime);

  return 0;
}
