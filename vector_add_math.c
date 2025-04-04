#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// Function to add two vectors
void vector_add(float* a, float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

int main(void) {
    // Create two vectors
    float a[2] = {1.0f, 2.0f};
    float b[2] = {2.0f, 1.0f};
    float c[2] = {0.0f, 0.0f};
    
    // Measure time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Perform vector addition
    vector_add(a, b, c, 2);
    
    // End time measurement
    gettimeofday(&end, NULL);
    
    // Calculate execution time
    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
    
    // Print results
    printf("Result of [1,2] + [2,1]:\n");
    printf("[%f, %f]\n", c[0], c[1]);
    printf("Execution time: %f seconds\n", time_taken);
    
    return 0;
}