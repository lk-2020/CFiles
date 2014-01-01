/***************************************************************************
 * FILE: mpi_wave.c
 * OTHER FILES: draw_wave.c
 * DESCRIPTION:
 *   MPI Concurrent Wave Equation - C Version
 *   Point-to-Point Communications Example
 *   This program implements the concurrent wave equation described
 *   in Chapter 5 of Fox et al., 1988, Solving Problems on Concurrent
 *   Processors, vol 1.
 *   A vibrating string is decomposed into points.  Each processor is
 *   responsible for updating the amplitude of a number of points over
 *   time. At each iteration, each processor exchanges boundary points with
 *   nearest neighbors.  This version uses low level sends and receives
 *   to exchange boundary points.
 *  AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
 *    Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
 * LAST REVISED: 07/05/05
 ***************************************************************************/
#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MASTER 0
#define TPOINTS 10000
#define PI 3.14159265

int RtoL = 10;
int LtoR = 20;
int OUT1 = 30;
int OUT2 = 40;
int nthreads, id;
void init_master(void);
void init_workers(void);
void init_line(void);
void update (int left, int right);
void output_master(void);
void output_workers(void);
extern void draw_wave(double *);

int	taskid,           /* task ID */
numtasks,             /* number of processes */
nsteps,               /* number of time steps */
npoints,              /* number of points handled by this processor */
first;                /* index of 1st point handled by this processor */
double	etime,                /* elapsed time in seconds */
values[TPOINTS+2],  /* values at time t */
oldval[TPOINTS+2],  /* values at time (t-dt) */
newval[TPOINTS+2];  /* values at time (t+dt) */
double start_time, end_time,avg_time;
MPI_Request	send_request,recv_request;


void init_line(void) {
    int nmin, nleft, npts, i, j, k;
    double x, fac;
    
    /* calculate initial values based on sine curve */
    nmin = TPOINTS/numtasks;
    nleft = TPOINTS%numtasks;
    fac = 2.0 * PI;
    for (i = 0, k = 0; i < numtasks; i++) {
        npts = (i < nleft) ? nmin + 1 : nmin;
        if (taskid == i) {
            first = k + 1;
            npoints = npts;
            printf ("task=%3d  first point=%5d  npoints=%4d\n", taskid,
                    first, npts);
            for (j = 1; j <= npts; j++, k++) {
                x = (double)k/(double)(TPOINTS - 1);
                values[j] = sin (fac * x);
            }
        }
        else k += npts;
    }
    #pragma omp parallel
    {
    #pragma omp for
    for (i = 1; i <= npoints; i++)
        oldval[i] = values[i];
    
    }
    nthreads = omp_get_num_threads();
    id = omp_get_thread_num();
    printf("INITLINE:Thread:%d of %d threads\n",id,nthreads);
}

/*  -------------------------------------------------------------------------
 *  All processes update their points a specified number of times
 *  -------------------------------------------------------------------------*/
void update(int left, int right) {
    int i, j;
    double dtime, c, dx, tau, sqtau;
    MPI_Status status;
    
    dtime = 0.3;
    c = 1.0;
    dx = 1.0;
    tau = (c * dtime / dx);
    sqtau = tau * tau;
    
    /* Update values for each point along string */
    for (i = 1; i <= nsteps; i++) {
        /* Exchange data with "left-hand" neighbor */
        if (first != 1) {
            MPI_Send(&values[1], 1, MPI_DOUBLE, left, RtoL, MPI_COMM_WORLD);//, &send_request);
            MPI_Recv(&values[0], 1, MPI_DOUBLE, left, LtoR, MPI_COMM_WORLD,// &recv_request);
                    &status);
        }
        /* Exchange data with "right-hand" neighbor */
        if (first + npoints -1 != TPOINTS) {
            MPI_Send(&values[npoints], 1, MPI_DOUBLE, right, LtoR, MPI_COMM_WORLD);  //,&send_request);
            MPI_Recv(&values[npoints+1], 1, MPI_DOUBLE, right, RtoL, MPI_COMM_WORLD, //&recv_request);
                      &status);
        }
        /* Update points along line */
        #pragma omp parallel
        {
        #pragma omp for
        for (j = 1; j <= npoints; j++) {
/* Global endpoints */
            if ((first + j - 1 == 1) || (first + j - 1 == TPOINTS))
                newval[j] = 0.0;
            else
            /* Use wave equation to update points */
                newval[j] = (2.0 * values[j]) - oldval[j]
                + (sqtau * (values[j-1] - (2.0 * values[j]) + values[j+1]));
        }
        }
        nthreads = omp_get_num_threads();
        id = omp_get_thread_num();
        printf("UPDATE:Thread:%d of %d threads\n",id,nthreads);
       // #pragma omp parallel private(j)
        #pragma omp for
        for (j = 1; j <= npoints; j++) {
            oldval[j] = values[j];
            values[j] = newval[j];
        }
    }
}


void updateOneNode() {
    int i, j;
    double dtime, c, dx, tau, sqtau;
    MPI_Status status;
    
    dtime = 0.3;
    c = 1.0;
    dx = 1.0;
    tau = (c * dtime / dx);
    sqtau = tau * tau;

/* Update points along line */
#pragma omp parallel   
{
#pragma omp for
        for (j = 1; j <= npoints; j++) {
            /* Global endpoints */
            if ((first + j - 1 == 1) || (first + j - 1 == TPOINTS))
                newval[j] = 0.0;
            else
            /* Use wave equation to update points */
                newval[j] = (2.0 * values[j]) - oldval[j]
                + (sqtau * (values[j-1] - (2.0 * values[j]) + values[j+1]));
        }
    nthreads = omp_get_num_threads();
    id = omp_get_thread_num();
    printf("Thread:%d of %d threads\n",id,nthreads);
#pragma omp for
        for (j = 1; j <= npoints; j++) {
            oldval[j] = values[j];
            values[j] = newval[j];
        }
}
}



/*  ------------------------------------------------------------------------
 *  Master receives results from workers and prints
 *  ------------------------------------------------------------------------ */
void output_master(void) {
    int i, j, source, start, npts, buffer[2];
    double results[TPOINTS];
    MPI_Status status;
    
    /* Store worker's results in results array */
    for (i = 1; i < numtasks; i++) {
        /* Receive first point, number of points and results */
        MPI_Recv(buffer, 2, MPI_INT, i, OUT1, MPI_COMM_WORLD, &status);
        start = buffer[0];
        npts = buffer[1];
        MPI_Recv(&results[start-1], npts, MPI_DOUBLE, i, OUT2,
                 MPI_COMM_WORLD, &status);
    }
    
    /* Store master's results in results array */
#pragma omp parallel
{
#pragma omp for
    for (i = first; i < first + npoints; i++)
        results[i-1] = values[i];
}
    nthreads = omp_get_num_threads();
    id = omp_get_thread_num();
    printf("OUTPUT:Thread:%d of %d threads\n",id,nthreads);
    printf("Click the EXIT button or use CTRL-C to quit\n");
}

/*  -------------------------------------------------------------------------
 *  Workers send the updated values to the master
 *  -------------------------------------------------------------------------*/

void output_workers(void) {
    int buffer[2];
    MPI_Status status;
    
    /* Send first point, number of points and results to master */
    buffer[0] = first;
    buffer[1] = npoints;
    MPI_Send(&buffer, 2, MPI_INT, MASTER, OUT1, MPI_COMM_WORLD);
    MPI_Send(&values[1], npoints, MPI_DOUBLE, MASTER, OUT2, MPI_COMM_WORLD);
}

/*  ------------------------------------------------------------------------
 *  Main program
 *  ------------------------------------------------------------------------ */

int main (int argc, char *argv[])
{
    int left, right, rc;
    nsteps = 10000000;

    /* Initialize MPI */
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    
    if (numtasks < 2)
    {
    start_time = MPI_Wtime();
    init_line();
    updateOneNode();
    output_master();
    end_time = MPI_Wtime();
    MPI_Finalize();
    avg_time= end_time-start_time;
    printf("\n total time = %lf",avg_time);
    return 0;
    }
    
    /* Determine left and right neighbors */
    if (taskid == numtasks-1)
        right = 0;
    else
        right = taskid + 1;
    
    if (taskid == 0)
        left = numtasks - 1;
    else
        left = taskid - 1;
    
    /* Get program parameters and initialize wave values */
    if (taskid == MASTER) {
        printf ("Starting mpi_wave using %d tasks.\n", numtasks);
        printf ("Using %d points on the vibrating string.\n", TPOINTS);
    }
    MPI_Barrier(MPI_COMM_WORLD); 
    start_time = MPI_Wtime();
    init_line();
    
    /* Update values along the line for nstep time steps */
    update(left, right);
    
    /* Master collects results from workers and prints */
    if (taskid == MASTER)
        output_master();
    else
       output_workers();
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    MPI_Finalize();
    
    
    
    avg_time= end_time-start_time;
    
    printf("\n total time = %lf",avg_time);
    
    return 0;
}


