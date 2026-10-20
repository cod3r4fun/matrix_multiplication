/*
**  PROGRAM: Matrix Multiply
**
**  PURPOSE: This is a simple matrix multiply program. 
**           It will compute the product
**
**                C  = A * B
**
**           A and B are set to constant matrices so we
**           can make a quick test of the multiplication.
**
**  HISTORY: Written by Tim Mattson, Nov 1999.
*/
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define AVAL 3.0
#define BVAL 5.0
#define TOL  0.001

int main(int argc, char **argv)
{
	int Ndim, Pdim, Mdim;   /* A[N][P], B[P][M], C[N][M] */
	int i,j,k;
	double *A, *B, *C, cval, err, errsq;
      double dN, mflops;
	double start_time, run_time;
    int rank, size;

    if (argc < 2) {
        printf("Usage: %s <matrix_order>\n", argv[0]);
        return -1;
    }

	int order = atoi(argv[1]);
	Ndim = order;
	Pdim = order;
	Mdim = order;

	/* Initialize matrices */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    B = (double *)malloc(Pdim*Mdim*sizeof(double));
    if (rank == 0) {
        A = (double *)malloc(Ndim*Pdim*sizeof(double));
        C = (double *)malloc(Ndim*Mdim*sizeof(double));

        for (i=0; i<Ndim; i++)
            for (j=0; j<Pdim; j++)
                *(A+(i*Ndim+j)) = AVAL;

        for (i=0; i<Pdim; i++)
            for (j=0; j<Mdim; j++)
                *(B+(i*Pdim+j)) = BVAL;

        for (i=0; i<Ndim; i++)
            for (j=0; j<Mdim; j++)
                *(C+(i*Ndim+j)) = 0.0;
    }

    start_time = MPI_Wtime();
    /* =========================== AREA OF INTEREST ============================ */
#define START_ROW(r, N, P) ((r) * ((N) / (P)) + ((r) < ((N) % (P)) ? (r) : ((N) % (P))))

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));

    int offset = 0;
    for (i = 0; i < size; i++) {
        int rows_for_i = (Ndim / size) + (i < (Ndim % size) ? 1 : 0);
        sendcounts[i] = rows_for_i * Mdim; 
        displs[i]     = offset;            
        offset       += sendcounts[i];
    }

    int my_count = sendcounts[rank];
    int my_rows  = my_count / Mdim;

    double *local_A = (rank == 0) ? A : (double *)malloc(my_count * sizeof(double));
    double *local_C = (rank == 0) ? C : (double *)calloc(my_count, sizeof(double));

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 (rank == 0) ? MPI_IN_PLACE : local_A, my_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Bcast(B, Pdim * Mdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (i = 0; i < my_rows; i++) {
        for (k = 0; k < Pdim; k++) {
            double a_ik = local_A[i * Pdim + k];
            for (j = 0; j < Mdim; j++) {
                local_C[i * Mdim + j] += a_ik * B[k * Mdim + j];
            }
        }
    }

    MPI_Gatherv((rank == 0) ? MPI_IN_PLACE : local_C, my_count, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank != 0) {
        free(local_A);
        free(local_C);
    }
    free(sendcounts);
    free(displs);    

    /* ========================= END AREA OF INTEREST ========================== */
	run_time = MPI_Wtime() - start_time;
 
	if (rank == 0) {
        double max_time;
        MPI_Reduce(&run_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        printf("Order %d multiplication in %f seconds\n", order, run_time);

        dN = (double)order;
        mflops = 2.0 * dN * dN * dN / (1.0e6 * run_time);
        printf("Order %d multiplication at %f mflops\n", order, mflops);
        // Check the result
        cval = Pdim * AVAL * BVAL;
        errsq = 0.0;
        for (i = 0; i < Ndim; i++)
            for (j = 0; j < Mdim; j++) {
                err = C[i * Mdim + j] - cval;
                errsq += err * err;
            }

        if (errsq > TOL)
            printf("Errors in multiplication: %f\n", errsq);
        else
            printf("Hey, it worked!\n");

        free(A);
        free(C);
    } else {
        MPI_Reduce(&run_time, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    free(B);

    MPI_Finalize();
}