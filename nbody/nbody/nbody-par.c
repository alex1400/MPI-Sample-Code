/*
    N-Body simulation code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif


#define GRAVITY     1.1
#define FRICTION    0.01
#define MAXBODIES   10000
#define DELTA_T     (0.025/5000)
#define BOUNCE      -0.9
#define SEED        27102015


struct bodyType {
    double x[2];        /* Old and new X-axis coordinates */
    double y[2];        /* Old and new Y-axis coordinates */
    double xf;          /* force along X-axis */
    double yf;          /* force along Y-axis */
    double xv;          /* velocity along X-axis */
    double yv;          /* velocity along Y-axis */
    double mass;        /* Mass of the body */
    double radius;      /* width (derived from mass) */
};


struct world {
    struct bodyType bodies[MAXBODIES];
    int                 bodyCt;
    int                 old;    // Flips between 0 and 1

    /*  Dimensions of space (very finite, ain't it?) */
    int                 xdim;
    int                 ydim;
};

/*  Macros to hide memory layout
*/
#define X(w, B)        (w)->bodies[B].x[(w)->old]
#define XN(w, B)       (w)->bodies[B].x[(w)->old^1]
#define Y(w, B)        (w)->bodies[B].y[(w)->old]
#define YN(w, B)       (w)->bodies[B].y[(w)->old^1]
#define XF(w, B)       (w)->bodies[B].xf
#define YF(w, B)       (w)->bodies[B].yf
#define XV(w, B)       (w)->bodies[B].xv
#define YV(w, B)       (w)->bodies[B].yv
#define R(w, B)        (w)->bodies[B].radius
#define M(w, B)        (w)->bodies[B].mass

MPI_Datatype mpi_bodytype;


//The function to define the MPI_struct data type
static void Build_Derived_Data(struct world *world) {
  int lengths[8]={2,2,1,1,1,1,1,1};
  const MPI_Aint dis[8] = {
        0,
        2 * sizeof(double),
        4 * sizeof(double),
        5 * sizeof(double),
        6 * sizeof(double),
        7 * sizeof(double),
        8 * sizeof(double),
        9 * sizeof(double)};


  MPI_Datatype types[8] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                            MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
  MPI_Type_create_struct(8, lengths, dis, types, &mpi_bodytype);
  MPI_Type_commit(&mpi_bodytype);
}

//The function is defined to send and Receive different parts of body to each node
static void
Send_Receive_World(struct world *world, int start_myrange, int end_myrange, int position, int size){
  MPI_Request req[size];
  unsigned int i;


  for(i=0;i<size;i++)
  {

    if((i+1) *position> world->bodyCt){
        MPI_Ibcast(&world->bodies[position*i], world->bodyCt - (i*position), mpi_bodytype, i, MPI_COMM_WORLD, &req[i]);
        break;
      }
    else
        MPI_Ibcast(&world->bodies[position*i], position, mpi_bodytype, i, MPI_COMM_WORLD, &req[i]);
  }

  if(i!=size) i++;
  MPI_Waitall(i , req, MPI_STATUSES_IGNORE);


}

static void
clear_forces(struct world *world, int start_myrange, int end_myrange)
{
    int b;

    /* Clear force accumulation variables */
    for (b = start_myrange; b < end_myrange; ++b) {
        YF(world, b) = XF(world, b) = 0;
    }
}

static void
compute_forces(struct world *world, int start_myrange, int end_myrange, int myrank, int size, int position, double* tem_yf, double *tem_xf)
{
    int b, c;

    /* Incrementally accumulate forces from each body pair,
       skipping force of body on itself (c == b)
    */
    //compute  forces
    #ifdef _OPENMP
    #pragma omp parallel for private(b,c) shared (world) reduction(- : tem_yf[:world->bodyCt]) reduction(- :tem_xf[:world->bodyCt])
    #endif
    for (b = start_myrange; b < end_myrange; ++b) {
        for (c = 0; c < world->bodyCt; ++c) {
            if (c >= start_myrange && c <=b) continue;
            double dx = X(world, c) - X(world, b);
            double dy = Y(world, c) - Y(world, b);
            double angle = atan2(dy, dx);
            double dsqr = dx*dx + dy*dy;
            double mindist = R(world, b) + R(world, c);
            double mindsqr = mindist*mindist;
            double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
            double force = M(world, b) * M(world, c) * GRAVITY / forced;
            double xf = force * cos(angle);
            double yf = force * sin(angle);

            /* Slightly sneaky...
               force of b on c is negative of c on b;
            */
            XF(world, b) += xf;
            YF(world, b) += yf;
            //The intermediate array to be deal with race condition in OpenMP
            #ifdef _OPENMP
            tem_yf[c]-=yf;
            tem_xf[c]-=xf;
            #else
            XF(world, c) -= xf;
            YF(world, c) -= yf;
            #endif
        }
    }

    #ifdef _OPENMP
    for (c = start_myrange+1; c < world->bodyCt; ++c) {
        YF(world, c) +=tem_yf[c];
        XF(world, c) +=tem_xf[c];
    }
    #endif

}

static void
compute_velocities_positions(struct world *world, int start_myrange, int end_myrange)
{
    int b;
    #ifdef _OPENMP
  	#pragma omp parallel for private(b)
  	#endif
    for (b = start_myrange; b < end_myrange; ++b) {
        double xv = XV(world, b);
        double yv = YV(world, b);
        double force = sqrt(xv*xv + yv*yv) * FRICTION;
        double angle = atan2(yv, xv);
        double xf = XF(world, b) - (force * cos(angle));
        double yf = YF(world, b) - (force * sin(angle));

        XV(world, b) += (xf / M(world, b)) * DELTA_T;
        YV(world, b) += (yf / M(world, b)) * DELTA_T;


        //compute positions
        double xn = X(world, b) + XV(world, b) * DELTA_T;
        double yn = Y(world, b) + YV(world, b) * DELTA_T;

        /* Bounce off image "walls" */
        if (xn < 0) {
            xn = 0;
            XV(world, b) = -XV(world, b);
        } else if (xn >= world->xdim) {
            xn = world->xdim - 1;
            XV(world, b) = -XV(world, b);
        }
        if (yn < 0) {
            yn = 0;
            YV(world, b) = -YV(world, b);
        } else if (yn >= world->ydim) {
            yn = world->ydim - 1;
            YV(world, b) = -YV(world, b);
        }

        /* Update position */
        XN(world, b) = xn;
        YN(world, b) = yn;

    }

}

/*  Graphic output stuff...
 */

#include <fcntl.h>
#include <sys/mman.h>

struct filemap {
    int            fd;
    off_t          fsize;
    void          *map;
    unsigned char *image;
};


static void
filemap_close(struct filemap *filemap)
{
    if (filemap->fd == -1) {
        return;
    }
    close(filemap->fd);
    if (filemap->map == MAP_FAILED) {
        return;
    }
    munmap(filemap->map, filemap->fsize);
}


static unsigned char *
Eat_Space(unsigned char *p)
{
    while ((*p == ' ') ||
           (*p == '\t') ||
           (*p == '\n') ||
           (*p == '\r') ||
           (*p == '#')) {
        if (*p == '#') {
            while (*(++p) != '\n') {
                // skip until EOL
            }
        }
        ++p;
    }

    return p;
}


static unsigned char *
Get_Number(unsigned char *p, int *n)
{
    p = Eat_Space(p);  /* Eat white space and comments */

    int charval = *p;
    if ((charval < '0') || (charval > '9')) {
        errno = EPROTO;
        return NULL;
    }

    *n = (charval - '0');
    charval = *(++p);
    while ((charval >= '0') && (charval <= '9')) {
        *n *= 10;
        *n += (charval - '0');
        charval = *(++p);
    }

    return p;
}


static int
map_P6(const char *filename, int *xdim, int *ydim, struct filemap *filemap)
{
    /* The following is a fast and sloppy way to
       map a color raw PPM (P6) image file
    */
    int maxval;
    unsigned char *p;

    /* First, open the file... */
    if ((filemap->fd = open(filename, O_RDWR)) < 0) {
        goto ppm_abort;
    }

    /* Read size and map the whole file... */
    filemap->fsize = lseek(filemap->fd, (off_t)0, SEEK_END);
    filemap->map = mmap(0,                      // Put it anywhere
                        filemap->fsize,         // Map the whole file
                        (PROT_READ|PROT_WRITE), // Read/write
                        MAP_SHARED,             // Not just for me
                        filemap->fd,            // The file
                        0);                     // Right from the start
    if (filemap->map == MAP_FAILED) {
        goto ppm_abort;
    }

    /* File should now be mapped; read magic value */
    p = filemap->map;
    if (*(p++) != 'P') goto ppm_abort;
    switch (*(p++)) {
    case '6':
        break;
    default:
        errno = EPROTO;
        goto ppm_abort;
    }

    p = Get_Number(p, xdim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, ydim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, &maxval);         // Get image max value */
    if (p == NULL) goto ppm_abort;

    /* Should be 8-bit binary after one whitespace char... */
    if (maxval > 255) {
        goto ppm_abort;
    }
    if ((*p != ' ') &&
        (*p != '\t') &&
        (*p != '\n') &&
        (*p != '\r')) {
        errno = EPROTO;
        goto ppm_abort;
    }

    /* Here we are... next byte begins the 24-bit data */
    filemap->image = p + 1;

    return 0;

ppm_abort:
    filemap_close(filemap);

    return -1;
}


static inline void
color(const struct world *world, unsigned char *image, int x, int y, int b)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));
    int tint = ((0xfff * (b + 1)) / (world->bodyCt + 2));

    p[0] = (tint & 0xf) << 4;
    p[1] = (tint & 0xf0);
    p[2] = (tint & 0xf00) >> 4;
}

static inline void
black(const struct world *world, unsigned char *image, int x, int y)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));

    p[2] = (p[1] = (p[0] = 0));
}

static void
display(const struct world *world, unsigned char *image)
{
    double i, j;
    int b;

    /* For each pixel */
    for (j = 0; j < world->ydim; ++j) {
        for (i = 0; i < world->xdim; ++i) {
            /* Find the first body covering here */
            for (b = 0; b < world->bodyCt; ++b) {
                double dy = Y(world, b) - j;
                double dx = X(world, b) - i;
                double d = sqrt(dx*dx + dy*dy);

                if (d <= R(world, b)+0.5) {
                    /* This is it */
                    color(world, image, i, j, b);
                    goto colored;
                }
            }

            /* No object -- empty space */
            black(world, image, i, j);

colored:        ;
        }
    }
}

static void
print(struct world *world)
{
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
               X(world, b), Y(world, b), XF(world, b), YF(world, b), XV(world, b), YV(world, b));
    }
}


/*  Main program...
============================================================================
============================================================================
============================================================================
============================================================================
============================================================================
============================================================================
*/

int
main(int argc, char **argv)
{
    unsigned int lastup = 0;
    unsigned int secsup;
    int b;
    int myrank, size;
    int steps;
    int position;
    double rtime, start_MPI_timing, end_MPI_timing;
    struct timeval start;
    struct timeval end;
    struct filemap image_map;
    int start_myrange;
    int end_myrange;
    double *tem_yf;
    double *tem_xf;

    struct world *world = calloc(1, sizeof *world);
    if (world == NULL) {
        fprintf(stderr, "Cannot calloc(world)\n");
        exit(1);
    }

    /* Get Parameters */
    if (argc != 5) {
        fprintf(stderr, "Usage: %s num_bodies secs_per_update ppm_output_file steps\n",
                argv[0]);
        exit(1);
    }
    if ((world->bodyCt = atol(argv[1])) > MAXBODIES ) {
        fprintf(stderr, "Using only %d bodies...\n", MAXBODIES);
        world->bodyCt = MAXBODIES;
    } else if (world->bodyCt < 2) {
        fprintf(stderr, "Using two bodies...\n");
        world->bodyCt = 2;
    }

//These tow intermediate array are defined to deal with the race condition in OpenMP and
//be able to impelement it without using the expensive "critical" keyword
    tem_yf = (double *)calloc(world->bodyCt, sizeof(double));
    tem_xf = (double *)calloc(world->bodyCt, sizeof(double));

    secsup = atoi(argv[2]);
    if (map_P6(argv[3], &world->xdim, &world->ydim, &image_map) == -1) {
        fprintf(stderr, "Cannot read %s: %s\n", argv[3], strerror(errno));
        exit(1);
    }
    steps = atoi(argv[4]);

  // MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Build_Derived_Data(world);
//Define the boundaries between the bodies to distribute between the nodes
    position=ceil(world->bodyCt/(double)size);
    start_myrange=myrank*position;
    end_myrange=start_myrange+position;
    if(end_myrange>world->bodyCt)
      end_myrange=world->bodyCt;

    if(myrank==0)
      fprintf(stderr, "Running N-body with %i bodies and %i steps\n", world->bodyCt, steps);

    /* Initialize simulation data */
    srand(SEED);
    for (b = 0; b < world->bodyCt; ++b) {
        X(world, b) = (rand() % world->xdim);
        Y(world, b) = (rand() % world->ydim);
        R(world, b) = 1 + ((b*b + 1.0) * sqrt(1.0 * ((world->xdim * world->xdim) + (world->ydim * world->ydim)))) /
                (25.0 * (world->bodyCt*world->bodyCt + 1.0));
        M(world, b) = R(world, b) * R(world, b) * R(world, b);
        XV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
        YV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
    }

    if (gettimeofday(&start, 0) != 0) {
        fprintf(stderr, "could not do timing\n");
        MPI_Finalize();
        exit(1);
    }

    start_MPI_timing= MPI_Wtime();

    /* Main Loop */
    while (steps--) {
        clear_forces(world, start_myrange, end_myrange);
        compute_forces(world, start_myrange, end_myrange, myrank, size, position, tem_yf, tem_xf);
        compute_velocities_positions(world, start_myrange, end_myrange);
        /* Flip old & new coordinates */
        world->old ^= 1;
        //Send and receive other parts of world
        Send_Receive_World(world, start_myrange, end_myrange, position, size);
        //The two intermediate array fill with zero for the next ieration.
        #ifdef _OPENMP
        memset(tem_yf,0,world->bodyCt* sizeof(tem_yf[0]));
        memset(tem_xf,0,world->bodyCt* sizeof(tem_xf[0]));
        #endif
        /*Time for a display update?*/
        if (secsup > 0 && (time(0) - lastup) > secsup) {
            display(world, image_map.image);
            msync(image_map.map, image_map.fsize, MS_SYNC); /* Force write */
            lastup = time(0);
        }
    }

    end_MPI_timing = MPI_Wtime();
    rtime=end_MPI_timing - start_MPI_timing;

    if (gettimeofday(&end, 0) != 0) {
        fprintf(stderr, "could not do timing\n");
        MPI_Finalize();
        exit(1);
    }

    if(myrank==0){
      print(world);
      fflush(stderr);
      usleep(20000);
      fprintf(stderr, "N-body took %10.3f seconds\n", rtime);
    }

    filemap_close(&image_map);
    free(tem_xf);
    free(tem_yf);
    free(world);
    MPI_Finalize();
    return 0;
}
