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

	// Potential Optimizations:
	// 1. split per worker
	// 2. use device memory 

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