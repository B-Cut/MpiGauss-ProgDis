#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_matrix(double *mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) {
            printf("%8.3f ", mat[i * (n + 1) + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void generate_random_matrix(double *mat, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) {
            mat[i * (n + 1) + j] = (rand() % 200 - 100) / 10.0; // Random values between -10.0 and 10.0
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size, n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    n = atoi(argv[1]);

    // Allocate memory for the matrix
    double *matrix = NULL;
    if (rank == 0) {
        matrix = malloc(n * (n + 1) * sizeof(double));
        generate_random_matrix(matrix, n);
        printf("Initial Matrix:\n");
        print_matrix(matrix, n);
    }

    // Calculate row distribution
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rows_per_proc = n / size;
    int extra_rows = n % size;

    for (int i = 0; i < size; i++) {
        sendcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * (n + 1);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    int local_rows = sendcounts[rank] / (n + 1);
    double *local_matrix = malloc(local_rows * (n + 1) * sizeof(double));
    MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE,
                 local_matrix, sendcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    double *row_buffer = malloc((n + 1) * sizeof(double));

    for (int k = 0; k < n; k++) {
        int owner = k / rows_per_proc + (k % rows_per_proc < extra_rows ? 0 : -1);

        // Broadcast pivot row
        if (rank == owner) {
            int local_k = k - displs[owner];
            for (int j = 0; j <= n; j++) {
                row_buffer[j] = local_matrix[local_k * (n + 1) + j];
            }
        }
        MPI_Bcast(row_buffer, n + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Perform elimination on local rows
        for (int i = 0; i < local_rows; i++) {
            int global_row = displs[rank] / (n + 1) + i;
            if (global_row > k) {
                double factor = local_matrix[i * (n + 1) + k] / row_buffer[k];
                for (int j = k; j <= n; j++) {
                    local_matrix[i * (n + 1) + j] -= factor * row_buffer[j];
                }
            }
        }
    }

    MPI_Gatherv(local_matrix, sendcounts[rank], MPI_DOUBLE,
                matrix, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix after Gaussian elimination:\n");
        print_matrix(matrix, n);
        free(matrix);
    }

    free(local_matrix);
    free(row_buffer);
    free(sendcounts);
    free(displs);
    MPI_Finalize();
    return 0;
}
