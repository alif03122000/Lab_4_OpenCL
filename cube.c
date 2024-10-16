#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <CL/cl.h>

#define DATA_SIZE (128)

// Kernel source code for cubing each element
const char *KernelSource = "\n" \
"__kernel void cube(                                                          \n" \
"   __global int* input,                                                 \n" \
"   __global int* output,                                                \n" \
"   const unsigned int count)                                            \n" \
"{                                                                       \n" \
"   int i = get_global_id(0);                                            \n" \
"   if (i < count)                                                       \n" \
"       output[i] = input[i] * input[i] * input[i];                      \n" \
"}                                                                       \n" \
"\n";


// Function to print device info
void print_device_info(cl_device_id device_id) {
    char device_name[1024];
    char vendor[1024];
    cl_uint max_compute_units;
    cl_uint max_work_item_dimensions;
    size_t max_work_item_sizes[3];
    size_t max_work_group_size;
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    
    // Get device name
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    
    // Get vendor name
    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    
    // Get maximum compute units
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
    
    // Get maximum work item dimensions
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);
    
    // Get maximum work item sizes
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, NULL);
    
    // Get maximum work group size
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    
    // Get global memory size
    clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    
    // Get local memory size
    clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
    
    // Print the device information
    printf("Device Name: %s\n", device_name);
    printf("Vendor: %s\n", vendor);
    printf("Max Compute Units: %u\n", max_compute_units);
    printf("Max Work Item Dimensions: %u\n", max_work_item_dimensions);
    printf("Max Work Item Sizes: %zu / %zu / %zu\n", max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);
    printf("Max Work Group Size: %zu\n", max_work_group_size);
    printf("Global Memory Size: %lu MB\n", global_mem_size / (1024 * 1024));
    printf("Local Memory Size: %lu KB\n", local_mem_size / 1024);
}

int main(int argc, char** argv)
{
    int err;                           // error code returned from api calls
    int data[DATA_SIZE];               // original data set given to device
    int results[DATA_SIZE];            // results returned from device
    unsigned int correct;              // number of correct results returned

    size_t global;                     // global domain size for our calculation
    size_t local;                      // local domain size for our calculation

    cl_device_id device_id;            // compute device id 
    cl_context context;                // compute context
    cl_command_queue commands;         // compute command queue
    cl_program program;                // compute program
    cl_kernel kernel;                  // compute kernel
    
    cl_mem input;                      // device memory used for the input array
    cl_mem output;                     // device memory used for the output array

    // Fill the data array with random integer values
    unsigned int count = DATA_SIZE;
    for (int i = 0; i < count; i++) {
        data[i] = rand() % 100;  // Fill with random values between 0 and 99
    }

    // Connect to a compute device (GPU)
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    print_device_info(device_id);

    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program
    kernel = clCreateKernel(program, "cube", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * count, NULL, NULL);
    if (!input || !output) {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(int) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1D input data set
    global = count;
    // err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command queue to finish
    clFinish(commands);

    // Read back the results from the device
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(int) * count, results, 0, NULL, NULL);  
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    // Validate the results
    correct = 0;
    for (int i = 0; i < count; i++) {
        if (results[i] == data[i] * data[i] * data[i]) {
            correct++;
        }
        printf("%d -> %d\n", data[i], results[i]);
    }

    // Print a summary
    printf("Computed '%d/%d' correct cubic values!\n", correct, count);

    // Cleanup
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
