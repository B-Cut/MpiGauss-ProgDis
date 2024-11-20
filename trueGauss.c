#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void print_matrix(int n, double** matrix, double* y, const char* label, int rank) {
    //if (rank == 0) {
        printf("%s do rank %d:\n", label, rank);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                printf("%8.2f ", matrix[i][j]);
            }
            printf("| %8.2f\n", y[i]);
        }
        printf("\n");
    //}
}



int main(int argc, char** argv){
    int rank, size;

    int n = 10;
    if(argc > 1) n = atoi(argv[1]);

    double **matrix = (double**)malloc(sizeof(double*)*n);
    double *y = (double*)malloc(sizeof(double)*n);
    double *res = (double*)malloc(sizeof(double)*n);
    
    for(int i = 0; i < n; i++){
        matrix[i] = (double*)malloc(sizeof(double)*n);
    }

    srand(1);
    for (int i = rank; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = rand() % 100 + 1;
        }
        y[i] = rand() % 100 + 1;
    }
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //print_matrix(n, matrix, y, "Teste", rank);

    MPI_Finalize();
    return 0;
}