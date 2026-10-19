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

    if (rank != 0) {
        A = (double *)malloc(Ndim * Pdim * sizeof(double));
        C = (double *)calloc(Ndim * Mdim, sizeof(double));
    }

    MPI_Bcast(A, Ndim * Pdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, Pdim * Mdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int starting_row_pos = START_ROW(rank, Ndim, size);
    int ending_row_pos   = START_ROW(rank + 1, Ndim, size);
    int rows_accumulated = ending_row_pos - starting_row_pos;


    for (i = starting_row_pos; i < ending_row_pos; i++) {
        for (k = 0; k < Pdim; k++) {
            double a_ik = *(A + (i * Pdim + k));
            for (j = 0; j < Mdim; j++) {
                *(C + (i * Mdim + j)) += a_ik * *(B + (k * Mdim + j));
            }
        }
    }

    for (int step = 1; step < size; step *= 2) {
        if (rank % (2 * step) == 0) {
            int src = rank + step;
            if (src < size) {
                int subtree_end_rank = (src + step < size) ? (src + step) : size;
                int recv_start_row   = START_ROW(src, Ndim, size);
                int recv_rows        = START_ROW(subtree_end_rank, Ndim, size) - recv_start_row;

                MPI_Recv(C + (recv_start_row * Mdim), recv_rows * Mdim, MPI_DOUBLE,
                         src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                rows_accumulated += recv_rows;
            }
        }
        else {
            int dst = rank - step;
            MPI_Send(C + (starting_row_pos * Mdim), rows_accumulated * Mdim, MPI_DOUBLE,
                        dst, 0, MPI_COMM_WORLD);
            break;
        }
    }

    if (rank != 0) {
        free(A);
        free(C);
    }
    

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