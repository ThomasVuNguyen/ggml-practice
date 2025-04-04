#include <stdio.h>
#include "ggml.h"

int main(void) {
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,  // 16 MB
        .mem_buffer = NULL,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return 1;
    }
    
    // Create two 1D tensors
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    
    // Set values for tensor a: [1, 2]
    float * a_data = ggml_get_data_f32(a);
    a_data[0] = 1.0f;
    a_data[1] = 2.0f;
    
    // Set values for tensor b: [2, 1]
    float * b_data = ggml_get_data_f32(b);
    b_data[0] = 2.0f;
    b_data[1] = 1.0f;
    
    // Perform addition: c = a + b
    struct ggml_tensor * c = ggml_add(ctx, a, b);
    
    // Manually compute the results without the graph API
    float * c_data = ggml_get_data_f32(c);
    c_data[0] = a_data[0] + b_data[0];  // 1 + 2 = 3
    c_data[1] = a_data[1] + b_data[1];  // 2 + 1 = 3
    
    // Print the results
    printf("Result of [1,2] + [2,1]:\n");
    printf("[%f, %f]\n", c_data[0], c_data[1]);
    
    // Free the context
    ggml_free(ctx);
    
    return 0;
}