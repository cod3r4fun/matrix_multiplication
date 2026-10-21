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
 {
        const int MC_TILE = 128;
        const int KC_TILE = 256;
        const int NC_TILE = 128;
        const int T_TILE  = 32;

        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);
        int Pr = dims[0];
        int Pc = dims[1];

        int my_row = rank / Pc;
        int my_col = rank % Pc;

        MPI_Comm row_comm, col_comm;
        MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);
        MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);

        int *row_counts   = (int *)malloc((Pr + Pc) * 3 * sizeof(int));
        int *row_displs   = row_counts + Pr;
        int *A_Counts     = row_displs + Pr;
        int *A_Displs     = A_Counts   + Pr;
        int *col_counts   = A_Displs   + Pr;
        int *col_displs   = col_counts + Pc;

        int *BT_Counts    = (int *)malloc(Pc * 2 * sizeof(int));
        int *BT_Displs    = BT_Counts + Pc;

        int r_off = 0;
        for (i = 0; i < Pr; i++) {
            row_counts[i] = (Ndim / Pr) + (i < (Ndim % Pr) ? 1 : 0);
            row_displs[i] = r_off;
            A_Counts[i]   = row_counts[i] * Pdim;
            A_Displs[i]   = r_off * Pdim;
            r_off        += row_counts[i];
        }

        int c_off = 0;
        for (j = 0; j < Pc; j++) {
            col_counts[j] = (Mdim / Pc) + (j < (Mdim % Pc) ? 1 : 0);
            col_displs[j] = c_off;
            BT_Counts[j]  = col_counts[j] * Pdim;
            BT_Displs[j]  = c_off * Pdim;
            c_off        += col_counts[j];
        }

        int my_rows = row_counts[my_row];
        int my_cols = col_counts[my_col];

        if (rank == 0) {
            for (int ii = 0; ii < Pdim; ii += T_TILE) {
                int imax = (ii + T_TILE < Pdim) ? (ii + T_TILE) : Pdim;
                for (i = ii; i < imax; i++) {
                    for (j = i + 1; j < imax; j++) {
                        double tmp      = B[i * Mdim + j];
                        B[i * Mdim + j] = B[j * Mdim + i];
                        B[j * Mdim + i] = tmp;
                    }
                }
                for (int jj = ii + T_TILE; jj < Mdim; jj += T_TILE) {
                    int jmax = (jj + T_TILE < Mdim) ? (jj + T_TILE) : Mdim;
                    for (i = ii; i < imax; i++) {
                        for (j = jj; j < jmax; j++) {
                            double tmp      = B[i * Mdim + j];
                            B[i * Mdim + j] = B[j * Mdim + i];
                            B[j * Mdim + i] = tmp;
                        }
                    }
                }
            }
        }

        double *local_A = (rank == 0)
            ? A
            : (double *)malloc(((size_t)my_rows * Pdim > 0 ? (size_t)my_rows * Pdim : 1) * sizeof(double));
        double *local_BT = B; 
        int ldc = (my_col == 0) ? Mdim : my_cols;
        double *local_C = NULL;
        if (rank == 0) {
            local_C = C;
        } else if (my_col == 0) {
            size_t elems = (size_t)my_rows * Mdim;
            local_C = (double *)calloc(elems > 0 ? elems : 1, sizeof(double));
        } else {
            size_t elems = (size_t)my_rows * my_cols;
            local_C = (double *)calloc(elems > 0 ? elems : 1, sizeof(double));
        }

        if (my_col == 0) {
            MPI_Scatterv(A, A_Counts, A_Displs, MPI_DOUBLE,
                         (my_row == 0) ? MPI_IN_PLACE : local_A,
                         my_rows * Pdim, MPI_DOUBLE, 0, col_comm);
        }
        MPI_Bcast(local_A, my_rows * Pdim, MPI_DOUBLE, 0, row_comm);

        if (my_row == 0) {
            MPI_Scatterv(B, BT_Counts, BT_Displs, MPI_DOUBLE,
                         (my_col == 0) ? MPI_IN_PLACE : local_BT,
                         my_cols * Pdim, MPI_DOUBLE, 0, row_comm);
        }
        MPI_Bcast(local_BT, my_cols * Pdim, MPI_DOUBLE, 0, col_comm);

        MPI_Request *row_reqs = NULL;
        MPI_Datatype *col_types = NULL;
        int active_reqs = 0;

        if (my_col == 0 && Pc > 1 && my_rows > 0) {
            row_reqs  = (MPI_Request *)malloc((Pc - 1) * sizeof(MPI_Request));
            col_types = (MPI_Datatype *)malloc((Pc - 1) * sizeof(MPI_Datatype));
            for (int c = 1; c < Pc; c++) {
                if (col_counts[c] > 0) {
                    MPI_Type_vector(my_rows, col_counts[c], Mdim, MPI_DOUBLE, &col_types[active_reqs]);
                    MPI_Type_commit(&col_types[active_reqs]);
                    MPI_Irecv(local_C + col_displs[c], 1, col_types[active_reqs],
                              c, 0, row_comm, &row_reqs[active_reqs]);
                    active_reqs++;
                }
            }
        }

        for (int ii = 0; ii < my_rows; ii += MC_TILE) {
            int imax  = (ii + MC_TILE < my_rows) ? (ii + MC_TILE) : my_rows;
            int i_vec = ii + ((imax - ii) & ~3);

            for (int kk = 0; kk < Pdim; kk += KC_TILE) {
                int kmax  = (kk + KC_TILE < Pdim) ? (kk + KC_TILE) : Pdim;
                int k_vec = kk + ((kmax - kk) & ~3);

                for (int jj = 0; jj < my_cols; jj += NC_TILE) {
                    int jmax  = (jj + NC_TILE < my_cols) ? (jj + NC_TILE) : my_cols;
                    int j_vec = jj + ((jmax - jj) & ~3);

                    for (i = ii; i < i_vec; i += 4) {
                        const double * __restrict__ a0_ptr = local_A + (size_t)(i + 0) * Pdim;
                        const double * __restrict__ a1_ptr = local_A + (size_t)(i + 1) * Pdim;
                        const double * __restrict__ a2_ptr = local_A + (size_t)(i + 2) * Pdim;
                        const double * __restrict__ a3_ptr = local_A + (size_t)(i + 3) * Pdim;

                        double * __restrict__ c0_ptr = local_C + (size_t)(i + 0) * ldc;
                        double * __restrict__ c1_ptr = local_C + (size_t)(i + 1) * ldc;
                        double * __restrict__ c2_ptr = local_C + (size_t)(i + 2) * ldc;
                        double * __restrict__ c3_ptr = local_C + (size_t)(i + 3) * ldc;

                        for (j = jj; j < j_vec; j += 4) {
                            const double * __restrict__ b0_ptr = local_BT + (size_t)(j + 0) * Pdim;
                            const double * __restrict__ b1_ptr = local_BT + (size_t)(j + 1) * Pdim;
                            const double * __restrict__ b2_ptr = local_BT + (size_t)(j + 2) * Pdim;
                            const double * __restrict__ b3_ptr = local_BT + (size_t)(j + 3) * Pdim;

                            double c00 = c0_ptr[j + 0], c01 = c0_ptr[j + 1], c02 = c0_ptr[j + 2], c03 = c0_ptr[j + 3];
                            double c10 = c1_ptr[j + 0], c11 = c1_ptr[j + 1], c12 = c1_ptr[j + 2], c13 = c1_ptr[j + 3];
                            double c20 = c2_ptr[j + 0], c21 = c2_ptr[j + 1], c22 = c2_ptr[j + 2], c23 = c2_ptr[j + 3];
                            double c30 = c3_ptr[j + 0], c31 = c3_ptr[j + 1], c32 = c3_ptr[j + 2], c33 = c3_ptr[j + 3];

                            for (k = kk; k < k_vec; k += 4) {
                                double a0 = a0_ptr[k], a1 = a1_ptr[k], a2 = a2_ptr[k], a3 = a3_ptr[k];
                                double b0 = b0_ptr[k], b1 = b1_ptr[k], b2 = b2_ptr[k], b3 = b3_ptr[k];
                                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;

                                a0 = a0_ptr[k + 1]; a1 = a1_ptr[k + 1]; a2 = a2_ptr[k + 1]; a3 = a3_ptr[k + 1];
                                b0 = b0_ptr[k + 1]; b1 = b1_ptr[k + 1]; b2 = b2_ptr[k + 1]; b3 = b3_ptr[k + 1];
                                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;

                                a0 = a0_ptr[k + 2]; a1 = a1_ptr[k + 2]; a2 = a2_ptr[k + 2]; a3 = a3_ptr[k + 2];
                                b0 = b0_ptr[k + 2]; b1 = b1_ptr[k + 2]; b2 = b2_ptr[k + 2]; b3 = b3_ptr[k + 2];
                                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;

                                a0 = a0_ptr[k + 3]; a1 = a1_ptr[k + 3]; a2 = a2_ptr[k + 3]; a3 = a3_ptr[k + 3];
                                b0 = b0_ptr[k + 3]; b1 = b1_ptr[k + 3]; b2 = b2_ptr[k + 3]; b3 = b3_ptr[k + 3];
                                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
                            }

                            for (; k < kmax; k++) {
                                double a0 = a0_ptr[k], a1 = a1_ptr[k], a2 = a2_ptr[k], a3 = a3_ptr[k];
                                double b0 = b0_ptr[k], b1 = b1_ptr[k], b2 = b2_ptr[k], b3 = b3_ptr[k];
                                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
                            }

                            c0_ptr[j + 0] = c00; c0_ptr[j + 1] = c01; c0_ptr[j + 2] = c02; c0_ptr[j + 3] = c03;
                            c1_ptr[j + 0] = c10; c1_ptr[j + 1] = c11; c1_ptr[j + 2] = c12; c1_ptr[j + 3] = c13;
                            c2_ptr[j + 0] = c20; c2_ptr[j + 1] = c21; c2_ptr[j + 2] = c22; c2_ptr[j + 3] = c23;
                            c3_ptr[j + 0] = c30; c3_ptr[j + 1] = c31; c3_ptr[j + 2] = c32; c3_ptr[j + 3] = c33;
                        }

                        for (; j < jmax; j++) {
                            const double * __restrict__ b0_ptr = local_BT + (size_t)j * Pdim;
                            double c00 = c0_ptr[j], c10 = c1_ptr[j], c20 = c2_ptr[j], c30 = c3_ptr[j];
                            for (k = kk; k < k_vec; k += 4) {
                                double b0 = b0_ptr[k], b1 = b0_ptr[k + 1], b2 = b0_ptr[k + 2], b3 = b0_ptr[k + 3];
                                c00 += a0_ptr[k] * b0 + a0_ptr[k + 1] * b1 + a0_ptr[k + 2] * b2 + a0_ptr[k + 3] * b3;
                                c10 += a1_ptr[k] * b0 + a1_ptr[k + 1] * b1 + a1_ptr[k + 2] * b2 + a1_ptr[k + 3] * b3;
                                c20 += a2_ptr[k] * b0 + a2_ptr[k + 1] * b1 + a2_ptr[k + 2] * b2 + a2_ptr[k + 3] * b3;
                                c30 += a3_ptr[k] * b0 + a3_ptr[k + 1] * b1 + a3_ptr[k + 2] * b2 + a3_ptr[k + 3] * b3;
                            }
                            for (; k < kmax; k++) {
                                double b0 = b0_ptr[k];
                                c00 += a0_ptr[k] * b0;
                                c10 += a1_ptr[k] * b0;
                                c20 += a2_ptr[k] * b0;
                                c30 += a3_ptr[k] * b0;
                            }
                            c0_ptr[j] = c00; c1_ptr[j] = c10; c2_ptr[j] = c20; c3_ptr[j] = c30;
                        }
                    }

                    for (i = i_vec; i < imax; i++) {
                        const double * __restrict__ a0_ptr = local_A + (size_t)i * Pdim;
                        double * __restrict__ c0_ptr       = local_C + (size_t)i * ldc;
                        for (j = jj; j < jmax; j++) {
                            const double * __restrict__ b0_ptr = local_BT + (size_t)j * Pdim;
                            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                            for (k = kk; k < k_vec; k += 4) {
                                s0 += a0_ptr[k + 0] * b0_ptr[k + 0];
                                s1 += a0_ptr[k + 1] * b0_ptr[k + 1];
                                s2 += a0_ptr[k + 2] * b0_ptr[k + 2];
                                s3 += a0_ptr[k + 3] * b0_ptr[k + 3];
                            }
                            double sum = (s0 + s1) + (s2 + s3);
                            for (; k < kmax; k++) {
                                sum += a0_ptr[k] * b0_ptr[k];
                            }
                            c0_ptr[j] += sum;
                        }
                    }
                }
            }
        }

        if (my_col == 0) {
            if (active_reqs > 0) {
                MPI_Waitall(active_reqs, row_reqs, MPI_STATUSES_IGNORE);
                for (int r = 0; r < active_reqs; r++) {
                    MPI_Type_free(&col_types[r]);
                }
            }
            if (row_reqs)  free(row_reqs);
            if (col_types) free(col_types);
        } else if (my_rows > 0 && my_cols > 0) {
            MPI_Send(local_C, my_rows * my_cols, MPI_DOUBLE, 0, 0, row_comm);
        }

        if (my_col == 0) {
            MPI_Gatherv((my_row == 0) ? MPI_IN_PLACE : local_C,
                        my_rows * Mdim, MPI_DOUBLE,
                        C, A_Counts, A_Displs, MPI_DOUBLE,
                        0, col_comm);
        }

        if (rank != 0) {
            free(local_A);
            free(local_C);
        }
        free(row_counts);
        free(BT_Counts);
        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
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