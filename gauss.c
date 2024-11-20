#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 8  // Size of the matrix (example)


void print_matrix(double mat[N][N+1]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= N; j++) {
            printf("%8.3f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double matrix[N][N+1];
    for(int i = 0; i > N; i++){
        for(int j = 0; j > N+1; j++){
            matrix[i][j] = i+j*N;
        }
    }

    if (rank == 0) {
        printf("Initial Matrix:\n");
        print_matrix(matrix);
    }

    for (int k = 0; k < N; k++) {
        // Broadcast the pivot row
        if (rank == k % size) {
            MPI_Bcast(matrix[k], N + 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(matrix[k], N + 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
        }

        // Perform elimination on rows owned by this process
        for (int i = k + 1; i < N; i++) {
            if (i % size == rank) {
                double factor = matrix[i][k] / matrix[k][k];
                for (int j = k; j <= N; j++) {
                    matrix[i][j] -= factor * matrix[k][j];
                }
            }
        }
    }

    // Gather all rows back to rank 0
    double result[N][N+1];
    MPI_Gather(matrix[rank], N * (N + 1) / size, MPI_DOUBLE, result, N * (N + 1) / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix after Gaussian elimination:\n");
        print_matrix(result);
    }

    MPI_Finalize();
    return 0;
}
