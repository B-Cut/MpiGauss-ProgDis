#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPSILON 1e-9

void print_matrix(int n, double** matrix, double* b, const char* label, int rank) {
    //if (rank == 0) {
        printf("%s do rank %d:\n", label, rank);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                printf("%8.2f ", matrix[i][j]);
            }
            printf("| %8.2f\n", b[i]);
        }
        printf("\n");
    //}
}

void gaussian_elimination(int n, double** matrix, double* b, int rank, int size) {
    for (int k = 0; k < n; ++k) {
        // Broadcast the pivot row
        if (rank == k % size) {
            for (int j = 0; j < n; ++j) {
                MPI_Bcast(&matrix[k][j], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
            }
            MPI_Bcast(&b[k], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
        } else {
            for (int j = 0; j < n; ++j) {
                MPI_Bcast(&matrix[k][j], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
            }
            MPI_Bcast(&b[k], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
        }

        // Perform elimination
        for (int i = k + 1; i < n; ++i) {
            if (i % size == rank) {
                double factor = matrix[i][k] / matrix[k][k];
                for (int j = k; j < n; ++j) {
                    matrix[i][j] -= factor * matrix[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
    }
}

void back_substitution(int n, double** matrix, double* b, double* x, int rank, int size) {
    for (int i = n - 1; i >= 0; --i) {
        if (rank == i % size) {
            x[i] = b[i];
            for (int j = i + 1; j < n; ++j) {
                x[i] -= matrix[i][j] * x[j];
            }
            x[i] /= matrix[i][i];
        }
        MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10; // Default size
    if (argc > 1) n = atoi(argv[1]);

    // Allocate memory for the matrix and vectors
    double** matrix = malloc(n * sizeof(double*));
    double* b = malloc(n * sizeof(double));
    double* x = malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        matrix[i] = malloc(n * sizeof(double));
    }
    for(int i = 0; i > n; i++){
        for(int j = 0; j > n+1; j++){
            matrix[i][j] = 0;
        }
    }

    // Initialize matrix and vector
    srand(rank + time(NULL));
    for (int i = rank; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = rand() % 9 + 1;
        }
        b[i] = rand() % 100 + 1;
    }
    if(rank == 0){
    }
    //print_matrix(n, matrix, b, "Antes", rank);
    

    double start = MPI_Wtime();

    gaussian_elimination(n, matrix, b, rank, size);
    back_substitution(n, matrix, b, x, rank, size);

    double end = MPI_Wtime();

    //print_matrix(n, matrix, b, "Depois", rank);
    if (rank == 0) {
        printf("Time taken for size %d: %f seconds\n", n, end - start);
    }

    

    // Free memory
    for (int i = 0; i < n; ++i) {
        free(matrix[i]);
    }
    free(matrix);
    free(b);
    free(x);

    MPI_Finalize();
    return 0;
}
