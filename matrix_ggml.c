#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "ggml.h"

// Print a portion of a tensor for verification
void print_tensor_sample(struct ggml_tensor* tensor, const char* name) {
    int rows = tensor->ne[1];
    int cols = tensor->ne[0];
    
    printf("%s (showing first 3x3 or less):\n", name);
    for (int i = 0; i < fmin(rows, 3); i++) {
        for (int j = 0; j < fmin(cols, 3); j++) {
            float* data = ggml_get_data_f32(tensor);
            float val = data[i * cols + j];
            printf("%f ", val);
        }
        printf("\n");
    }
    printf("\n");
}

// Custom sigmoid implementation
void apply_sigmoid(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
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
    
    // Seed random number generator
    srand(42);  // Fixed seed for reproducibility
    
    // Initialize GGML
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024 * 1024,  // 1 GB
        .mem_buffer = NULL,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }
    
    // Create tensors
    struct ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, m);
    struct ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p, n);
    struct ggml_tensor* C = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p, m);
    
    // Initialize tensors with random values
    float* A_data = ggml_get_data_f32(A);
    float* B_data = ggml_get_data_f32(B);
    float* C_data = ggml_get_data_f32(C);
    
    for (int i = 0; i < m * n; i++) {
        A_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    for (int i = 0; i < n * p; i++) {
        B_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Measure time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Perform matrix multiplication manually
    // C = A * B where A is m x n and B is n x p
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A_data[i * n + k] * B_data[k * p + j];
            }
            C_data[i * p + j] = sum;
        }
    }
    
    // Apply sigmoid activation function
    apply_sigmoid(C_data, m * p);
    
    // End time measurement
    gettimeofday(&end, NULL);
    
    // Calculate execution time
    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
    
    // Print results sample and execution time
    print_tensor_sample(A, "Matrix A");
    print_tensor_sample(B, "Matrix B");
    print_tensor_sample(C, "Result C (after sigmoid)");
    printf("Execution time: %f seconds\n", time_taken);
    
    // Calculate and print FLOPS (Floating-point Operations Per Second)
    double flops = (2.0 * m * n * p) + (5.0 * m * p);
    printf("FLOPS: %.2f GFLOPS\n", flops / time_taken / 1e9);
    
    // Free GGML context
    ggml_free(ctx);
    
    return 0;
}