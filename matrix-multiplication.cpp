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
// static auto exception_handler = [](sycl::exception_list e_list) {
//     for (std::exception_ptr const& e : e_list) {
//         try {
//             std::rethrow_exception(e);
//         }
//         catch (std::exception const& e) {
// #if _DEBUG
//             std::cout << "Failure" << std::end
// #endif
//             std::terminate();
//         }
//     }
// };

static auto getSelector(){
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
    return selector;
}

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

void printMatrix(std::vector<int> matrix, int N, string matrixName){
    std::cout << "------ " << "Matrix " << matrixName << " ---" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << matrix[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-------------------" << std::endl;
}

void resetMatrix(std::vector<int> matrix, int N){
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            matrix[i*N + j] = 0;
        }
    }
}

void runBasicMult(
    queue q,
    std::vector<int> a_matrix,
    std::vector<int> b_matrix,
    std::vector<int> c_matrix,
    int N){

    std::cout << "Using basic buffers ... ";

    buffer<int, 2> a_buf (a_matrix.data(), range<2>(N, N));
    buffer<int, 2> b_buf (b_matrix.data(), range<2>(N, N));
    buffer<int, 2> c_buf (c_matrix.data(), range<2>(N, N));

    // Start timing
    auto operationStart = std::chrono::high_resolution_clock::now();

    q.submit([&](handler& h) {
        // Get accessors to the buffers
        auto a_acc = a_buf.get_access<access::mode::read>(h);
        auto b_acc = b_buf.get_access<access::mode::read>(h);
        auto c_acc = c_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<2>(N, N), [=](id<2> idx) {
            size_t i = idx[0];
            size_t j = idx[1];
            int sum = 0;

            for (size_t k = 0; k < N; ++k) {
                sum += a_acc[i][k] * b_acc[k][j];
            }

            c_acc[i][j] = sum;
        });
    });

    // Wait for the queue to finish processing
    q.wait();

    // Copying values back from buffer:
    {
        auto C_acc = c_buf.get_access<access::mode::read>();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                c_matrix[i * N + j] = C_acc[i][j];
            }
        }
    }

    // End timing
    auto operationEnd = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = operationEnd - operationStart;
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;
}

void runSharedMemory(
    queue q,
    std::vector<int> a_matrix,
    std::vector<int> b_matrix,
    std::vector<int> c_matrix,
    int N){

    std::cout << "Using shared memory ... ";
    // Create a 2D vector to represent the matrix
    int* a_shared = malloc_shared<int>(N*N, q);
    int* b_shared = malloc_shared<int>(N*N, q);
    int* c_shared = malloc_shared<int>(N*N, q);

    std::memcpy(a_shared, a_matrix.data(), N*N * sizeof(int));
    std::memcpy(b_shared, b_matrix.data(), N*N * sizeof(int));

    // Start timing
    auto operationStart = std::chrono::high_resolution_clock::now();

    q.submit([&](handler& h) {
        // Define the kernel
        h.parallel_for(
            range<2>(N, N), [=](id<2> idx) {
                size_t i = idx[0];
                size_t j = idx[1];
                int sum = 0;
                for (size_t k = 0; k < N; ++k) {
                    sum += a_shared[i*N + k] * b_shared[k*N + j];
                }
                c_shared[i*N + j] = sum;
            }
        );
    });

    // Wait for the queue to finish processing
    q.wait();

    std::memcpy(c_matrix.data(), c_shared, N*N * sizeof(int));

    // End timing
    auto operationEnd = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = operationEnd - operationStart;
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

    // Free the allocated memory
    free(a_shared, q);
    free(b_shared, q);
    free(c_shared, q);
}

void runDeviceMemory(
    queue q,
    std::vector<int> a_matrix,
    std::vector<int> b_matrix,
    std::vector<int> c_matrix,
    int N){

    auto dev = q.get_device();
    auto ctxt = q.get_context();

    std::cout << "Using device memory ... ";

    // Creating pointers in both host and device
    int* A_h = (int*) malloc_host(N*N * sizeof(int), ctxt);
    int* B_h = (int*) malloc_host(N*N * sizeof(int), ctxt);
    int* C_h = (int*) malloc_host(N*N * sizeof(int), ctxt);

    int* A_d = (int*) malloc_device(N*N * sizeof(int), dev, ctxt);
    int* B_d = (int*) malloc_device(N*N * sizeof(int), dev, ctxt);
    int* C_d = (int*) malloc_device(N*N * sizeof(int), dev, ctxt);

    // Copying input matrixes into pointers in host memory:
    std::memcpy(A_h, a_matrix.data(), N*N * sizeof(int));
    std::memcpy(B_h, b_matrix.data(), N*N * sizeof(int));

    // Copying input to device memory:
    q.submit([&](handler& h) {
        h.memcpy(A_d, A_h, N*N*sizeof(int));
    });
    q.wait();

    q.submit([&](handler& h) {
        h.memcpy(B_d, B_h, N*N*sizeof(int));
    });
    q.wait();

    // Start timing
    auto operationStart = std::chrono::high_resolution_clock::now();

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

    // Copying result from device to host memory:
    q.submit([&](handler& h) {
        h.memcpy(C_h, C_d, N*N*sizeof(int));
    });
    q.wait();

    // Copying result to c_matrix:
    std::memcpy(c_matrix.data(), C_h, N*N * sizeof(int));

    // End timing
    auto operationEnd = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = operationEnd - operationStart;
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

    // Free the allocated memory
    free(A_d, q);
    free(B_d, q);
    free(C_d, q);
}

void runNdRange(
    queue q,
    std::vector<int> a_matrix,
    std::vector<int> b_matrix,
    std::vector<int> c_matrix,
    int N,
    int B){

    std::cout << "Using ND-Range with B = " << B << " ... ";

    buffer<int, 2> a_buf (a_matrix.data(), range<2>(N, N));
    buffer<int, 2> b_buf (b_matrix.data(), range<2>(N, N));
    buffer<int, 2> c_buf (c_matrix.data(), range<2>(N, N));

    // Start timing
    auto operationStart = std::chrono::high_resolution_clock::now();

    q.submit([&](handler& h) {
        // Get accessors to the buffers
        auto a_acc = a_buf.get_access<access::mode::read>(h);
        auto b_acc = b_buf.get_access<access::mode::read>(h);
        auto c_acc = c_buf.get_access<access::mode::write>(h);

        // ND-Range Kernel
        range<2> global(N, N);
        range<2> local(B, B);
        h.parallel_for<class matrix_mul>(nd_range<2>(global, local), [=](nd_item<2> item){
            int i = item.get_global_id(0);
            int j = item.get_global_id(1);
            int sum = 0;
            for (int k = 0; k < N; ++k){
                sum += a_acc[i][k] * b_acc[k][j];
            }
            c_acc[i][j] = sum;
        });
    });

    // Wait for the queue to finish processing
    q.wait();

    // Copying values back from buffer:
    {
        auto C_acc = c_buf.get_access<access::mode::read>();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                c_matrix[i * N + j] = C_acc[i][j];
            }
        }
    }

    // End timing
    auto operationEnd = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = operationEnd - operationStart;
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;
}

void runHierarchy(
    queue q,
    std::vector<int> a_matrix,
    std::vector<int> b_matrix,
    std::vector<int> c_matrix,
    int N,
    int B){

    std::cout << "Using Hierarchy with B = " << B << " ... ";

    buffer<int, 2> a_buf (a_matrix.data(), range<2>(N, N));
    buffer<int, 2> b_buf (b_matrix.data(), range<2>(N, N));
    buffer<int, 2> c_buf (c_matrix.data(), range<2>(N, N));

    // Start timing
    auto operationStart = std::chrono::high_resolution_clock::now();

    q.submit([&](handler& h) {
        // Get accessors to the buffers
        auto a_acc = a_buf.get_access<access::mode::read>(h);
        auto b_acc = b_buf.get_access<access::mode::read>(h);
        auto c_acc = c_buf.get_access<access::mode::write>(h);

        // Hierarchical parallelism
        range<2> num_groups(N/B, N/B);
        range<2> group_size(B, B);

        h.parallel_for_work_group<class matrix_mul_h>(num_groups, [=](group<2> grp){
            int ib = grp.get_id(0);
            int jb = grp.get_id(1);
            
            grp.parallel_for_work_item(group_size, [&](h_item<2> item){
                int i = ib*B + item.get_logical_local_id(0);
                int j = jb*B + item.get_logical_local_id(1);

                if (i < N && j < N) { // Ensure indices are within bounds
                    for (int k = 0; k < N; ++k) {
                        c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
                    }
                }
            });
        });
    });

    // Wait for the queue to finish processing
    q.wait();

    // Copying values back from buffer:
    {
        auto C_acc = c_buf.get_access<access::mode::read>();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                c_matrix[i * N + j] = C_acc[i][j];
            }
        }
    }

    // End timing
    auto operationEnd = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = operationEnd - operationStart;
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;
}

int main() {
    auto selector = getSelector();

    // Create a SYCL queue to submit work
    queue q(selector);

    // Print out the device information used for the kernel code.
    cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";

    // Define matrix dimensions
    const size_t N = 1000; // Assuming square matrices for simplicity
    cout << "Matrix size: " << N << "x" << N << "\n";

    // Create a 2D vector to represent the matrix
    std::vector<int> a_matrix(N*N);
    std::vector<int> b_matrix(N*N);
    std::vector<int> c_matrix(N*N);

    // Set up random number generation
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> distr(0, 100); // Define the range

    // Initialize the input matrices with random integers
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            a_matrix[i*N + j] = distr(gen); // Generate random number and assign to matrix
            b_matrix[i*N + j] = distr(gen); // Generate random number and assign to matrix
        }
    }

    // Print matrices A and B before kernel execution
    // printMatrix(a_matrix, N, "A");
    // printMatrix(b_matrix, N, "B");

    // Potential Optimizations:
    // 1. use device memory 
    // 2. Use ND-range parallelism and work-group local memory. ALSO TRY tiled matrix multiply optimization (Fig 4-17)

    runBasicMult(q, a_matrix, b_matrix, c_matrix, N);

    resetMatrix(c_matrix, N);
    runSharedMemory(q, a_matrix, b_matrix, c_matrix, N);

    resetMatrix(c_matrix, N);
    runDeviceMemory(q, a_matrix, b_matrix, c_matrix, N);

    resetMatrix(c_matrix, N);
    runNdRange(q, a_matrix, b_matrix, c_matrix, N, 2);

    resetMatrix(c_matrix, N);
    runNdRange(q, a_matrix, b_matrix, c_matrix, N, 10);

    resetMatrix(c_matrix, N);
    runNdRange(q, a_matrix, b_matrix, c_matrix, N, 20);

    resetMatrix(c_matrix, N);
    runHierarchy(q, a_matrix, b_matrix, c_matrix, N, 2);

    resetMatrix(c_matrix, N);
    runHierarchy(q, a_matrix, b_matrix, c_matrix, N, 10);

    resetMatrix(c_matrix, N);
    runHierarchy(q, a_matrix, b_matrix, c_matrix, N, 20);

    return 0;
}
