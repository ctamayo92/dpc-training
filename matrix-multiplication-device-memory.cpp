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
#endif

    // Create a SYCL queue to submit work
    queue q(selector, exception_handler);
    auto dev = q.get_device();
    auto ctxt = q.get_context();

    // Print out the device information used for the kernel code.
    cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";

    // Define matrix dimensions
    const size_t N = 10000; // Assuming square matrices for simplicity
    cout << "Matrix size: " << N << "x" << N << "\n";

    // Create a 2D vector to represent the matrix
    int* A_h = (int*) malloc_host(N*N * sizeof(int), ctxt);
    int* B_h = (int*) malloc_host(N*N * sizeof(int), ctxt);
    int* C_h = (int*) malloc_host(N*N * sizeof(int), ctxt);

    int* A_d = (int*) malloc_device(N*N * sizeof(int), dev, ctxt);
    int* B_d = (int*) malloc_device(N*N * sizeof(int), dev, ctxt);
    int* C_d = (int*) malloc_device(N*N * sizeof(int), dev, ctxt);

    // Set up random number generation
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(0, 100); // Define the range

    auto initStart = std::chrono::high_resolution_clock::now();
    // Initialize the matrix with random integers
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A_h[i*N + j] = distr(gen); // Generate random number and assign to matrix
            B_h[i*N + j] = distr(gen); // Generate random number and assign to matrix
        }
    }
    auto initEnd = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> initDuration = initEnd - initStart;
    std::cout << "Time taken for matrix initialization: " << initDuration.count() << " seconds" << std::endl;

    // Print matrices A and B before kernel execution
    // printMatrix(A_h, N, "A_h");
    // printMatrix(B_h, N, "B_h");

    // Start timing
    auto operationStart = std::chrono::high_resolution_clock::now();

    std::cout << "Copying input matrices from host to device." << std::endl;
    q.submit([&](handler& h) {
        h.memcpy(A_d, A_h, N*N*sizeof(int));
    });
    q.wait();

    q.submit([&](handler& h) {
        h.memcpy(B_d, B_h, N*N*sizeof(int));
    });
    q.wait();

    // Submit a command group to the queue
    q.submit([&](handler& h) {
        // Define the kernel
        h.parallel_for(
            range<2>(N, N), [=](id<2> idx) {
                size_t i = idx[0];
                size_t j = idx[1];
                int sum = 0;
                for (size_t k = 0; k < N; ++k) {
                    sum += A_d[i*N + k] * B_d[k*N + j];
                }
                C_d[i*N + j] = sum;
            }
        );
    });

    // Wait for the queue to finish processing
    q.wait();

    std::cout << "Copying result matrix from device to host." << std::endl;
    q.submit([&](handler& h) {
        h.memcpy(C_h, C_d, N*N*sizeof(int));
    });
    q.wait();

    // End timing
    auto operationEnd = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = operationEnd - operationStart;
    std::cout << "Time taken for matrix multiplication: " << duration.count() << " seconds" << std::endl;

    // printMatrix(C_h, N, "C_h");

    // Free the allocated memory
    free(A_d, q);
    free(B_d, q);
    free(C_d, q);
    free(A_h, q);
    free(B_h, q);
    free(C_h, q);

    return 0;
}