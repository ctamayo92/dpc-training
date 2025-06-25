#include <sycl/sycl.hpp>

#include <vector>
#include <iostream>
#include <random>

#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;
using namespace std;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
#if _DEBUG
            std::cout << "Failure" << std::end
#endif
            std::terminate();
        }
    }
};

std::string device_type_to_string(sycl::info::device_type type) {
    switch (type) {
        case sycl::info::device_type::cpu:
            return "CPU";
        case sycl::info::device_type::gpu:
            return "GPU";
        case sycl::info::device_type::accelerator:
            return "Accelerator";
        case sycl::info::device_type::host:
            return "Host";
        default:
            return "Unknown";
    }
}

// Function to perform matrix-matrix multiplication
// void matrixMultiply(
//     const std::vector<std::vector<int>>& A,
//     const std::vector<std::vector<int>>& B,
//     std::vector<std::vector<int>>& C) {

//     // Get the dimensions of the matrices
//     size_t rowsA = A.size();
//     size_t colsA = A[0].size();
//     size_t rowsB = B.size();
//     size_t colsB = B[0].size();

//     // Check if the matrices can be multiplied
//     if (colsA != rowsB) {
//         std::cerr << "Error: Number of columns in A must be equal to number of rows in B." << std::endl;
//         return;
//     }

//     // Resize the result matrix C to the correct dimensions
//     C.resize(rowsA, std::vector<int>(colsB, 0));

//     // Perform matrix multiplication
//     for (size_t i = 0; i < rowsA; ++i) {
//         for (size_t j = 0; j < colsB; ++j) {
//             for (size_t k = 0; k < colsA; ++k) {
//                 C[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }

//     std::cout << "Matrix multiplication completed." << std::endl;
// }

void printMatrix(const int* matrix, int N, string matrixName){
    std::cout << "------ " << "Matrix " << matrixName << " ---" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << matrix[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-------------------" << std::endl;
}

int main() {
    // Create device selector for the device of your interest.
#if FPGA_EMULATOR
    // Intel extension: FPGA emulator selector on systems without FPGA card.
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
    // Intel extension: FPGA simulator selector on systems without FPGA card.
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    // Intel extension: FPGA selector on systems with FPGA card.
    auto selector = sycl::ext::intel::fpga_selector_v;
#else
    // The default device selector will select the most performant device.
    auto selector = default_selector_v;
    // auto selector = cpu_selector_v;
#endif


    auto devices = sycl::device::get_devices();

    std::cout << "Available devices:" << std::endl;
    for (const auto& device : devices) {
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "  Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
        std::cout << "  Type: " << device_type_to_string( device.get_info<sycl::info::device::device_type>() ) << std::endl;
        // std::cout << "  Max Compute Units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
        // std::cout << "  Global Memory Size: " << device.get_info<sycl::info::device::global_mem_size>() << " bytes" << std::endl;
        // std::cout << "  Local Memory Size: " << device.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;
        // std::cout << std::endl;
    }

    // Create a SYCL queue to submit work
    queue q(selector, exception_handler);

    // Print out the device information used for the kernel code.
    cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";

    // Define matrix dimensions
    const size_t N = 10000; // Assuming square matrices for simplicity
    cout << "Matrix size: " << N << "x" << N << "\n";

    // Create a 2D vector to represent the matrix
    int* A = malloc_shared<int>(N*N, q);
    int* B = malloc_shared<int>(N*N, q);
    int* C = malloc_shared<int>(N*N, q);

    // Set up random number generation
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(0, 100); // Define the range

    // TODO: Parallelize this!!
    auto initStart = std::chrono::high_resolution_clock::now();
    // Initialize the matrix with random integers
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[i*N + j] = distr(gen); // Generate random number and assign to matrix
            B[i*N + j] = distr(gen); // Generate random number and assign to matrix
        }
    }
    auto initEnd = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> initDuration = initEnd - initStart;
    std::cout << "Time taken for matrix initialization: " << initDuration.count() << " seconds" << std::endl;

    // Print matrices A and B before kernel execution
    // printMatrix(A, N, "A");
    // printMatrix(B, N, "B");

    // Start timing
    auto operationStart = std::chrono::high_resolution_clock::now();

    // Submit a command group to the queue
    q.submit([&](handler& h) {
        // Define the kernel
        h.parallel_for(
            range<2>(N, N), [=](id<2> idx) {
                size_t i = idx[0];
                size_t j = idx[1];
                int sum = 0;
                for (size_t k = 0; k < N; ++k) {
                    sum += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = sum;
            }
        );
    });

    // Wait for the queue to finish processing
    q.wait();

    // End timing
    auto operationEnd = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = operationEnd - operationStart;
    std::cout << "Time taken for matrix multiplication: " << duration.count() << " seconds" << std::endl;

    // printMatrix(C, N, "C");

    // Free the allocated memory
    free(A, q);
    free(B, q);
    free(C, q);

    return 0;
}