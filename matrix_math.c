#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Matrix multiplication: C = A * B
void matrix_multiply(float* A, float* B, float* C, int m, int n, int p) {
    // A is m x n, B is n x p, C is m x p
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// Apply sigmoid activation to each element of a matrix
void apply_sigmoid(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = sigmoid(matrix[i]);
    }
}

// Initialize matrix with random values
void init_random_matrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Values between -1 and 1
    }
}

// Print a portion of a matrix for verification
void print_matrix_sample(float* matrix, int rows, int cols, const char* name) {
    printf("%s (showing first 3x3 or less):\n", name);
    for (int i = 0; i < fmin(rows, 3); i++) {
        for (int j = 0; j < fmin(cols, 3); j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {
    // Matrix dimensions
    int m = 1000;  // Rows of A
    int n = 500;   // Columns of A, Rows of B
    int p = 800;   // Columns of B
    
    // Parse command line arguments if provided
    if (argc > 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        p = atoi(argv[3]);
    }
    
    printf("Matrix dimensions: A(%d,%d) x B(%d,%d) = C(%d,%d)\n", m, n, n, p, m, p);
    
    // Allocate matrices
    float* A = (float*)malloc(m * n * sizeof(float));
    float* B = (float*)malloc(n * p * sizeof(float));
    float* C = (float*)malloc(m * p * sizeof(float));
    
    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Seed random number generator
    srand(42);  // Fixed seed for reproducibility
    
    // Initialize matrices with random values
    init_random_matrix(A, m * n);
    init_random_matrix(B, n * p);
    
    // Measure time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Perform matrix multiplication
    matrix_multiply(A, B, C, m, n, p);
    
    // Apply sigmoid activation function
    apply_sigmoid(C, m * p);
    
    // End time measurement
    gettimeofday(&end, NULL);
    
    // Calculate execution time
    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
    
    // Print results sample and execution time
    print_matrix_sample(A, m, n, "Matrix A");
    print_matrix_sample(B, n, p, "Matrix B");
    print_matrix_sample(C, m, p, "Result C (after sigmoid)");
    printf("Execution time: %f seconds\n", time_taken);
    
    // Calculate and print FLOPS (Floating-point Operations Per Second)
    // For matrix multiplication: 2*m*n*p floating-point operations
    // For sigmoid: 5*m*p (approximately, considering exp and division)
    double flops = (2.0 * m * n * p) + (5.0 * m * p);
    printf("FLOPS: %.2f GFLOPS\n", flops / time_taken / 1e9);
    
    // Free memory
    free(A);
    free(B);
    free(C);
    
    return 0;
}