#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>

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

void convolution(sycl::queue& q, const cv::Mat& input, cv::Mat& output, const std::vector<float>& filter, size_t filter_size) {
    // Get image dimensions and channels
    size_t width = input.cols;
    size_t height = input.rows;
    size_t channels = input.channels();

    // Allocate shared memory for input, output, and filter
    uchar* input_data = malloc_shared<uchar>(width * height * channels, q);
    uchar* output_data = malloc_shared<uchar>(width * height * channels, q);
    float* filter_data = malloc_shared<float>(filter_size * filter_size, q);

    // Copy data to shared memory
    std::memcpy(input_data, input.data, width * height * channels);
    std::copy(filter.begin(), filter.end(), filter_data);

    // Submit kernel to perform convolution
    q.submit([&](handler& h) {
        h.parallel_for(range<3>(height, width, channels), [=](id<3> idx) {
            size_t y = idx[0];
            size_t x = idx[1];
            size_t c = idx[2];
            float sum = 0.0f;

            // Apply filter
            for (size_t i = 0; i < filter_size; ++i) {
                for (size_t j = 0; j < filter_size; ++j) {
                    int x_offset = x + j - filter_size / 2;
                    int y_offset = y + i - filter_size / 2;

                    // Check boundaries
                    if (x_offset >= 0 && x_offset < width && y_offset >= 0 && y_offset < height) {
                        sum += input_data[(y_offset * width + x_offset) * channels + c] * filter_data[i * filter_size + j];
                    }
                }
            }

            // Clamp the result to valid uchar range
            output_data[(y * width + x) * channels + c] = static_cast<uchar>(std::clamp(sum, 0.0f, 255.0f));
        });
    });

    q.wait();

    // Copy result back to output image
    std::memcpy(output.data, output_data, width * height * channels);

    // Free allocated memory
    free(input_data, q);
    free(output_data, q);
    free(filter_data, q);
}

// Convolution filters:
std::vector<float> get_laplacian_filter() {
    return {
        0, -1,  0,
       -1,  4, -1,
        0, -1,  0
    };
}

std::vector<float> get_gaussian_blur() {
    return {
        1/16.0f, 1/8.0f, 1/16.0f,
        1/8.0f,  1/4.0f, 1/8.0f,
        1/16.0f, 1/8.0f, 1/16.0f
    };
}

std::vector<float> get_average_filter() {
    return {
        1/9.0f, 1/9.0f, 1/9.0f,
        1/9.0f, 1/9.0f, 1/9.0f,
        1/9.0f, 1/9.0f, 1/9.0f
    };
}

std::vector<float> get_sharpening_filter() {
    return {
        0, -1,  0,
       -1,  5, -1,
        0, -1,  0
    };
}

std::vector<float> get_filter(string filter_name) {
    if (filter_name == "laplacian") {
        return get_laplacian_filter();
    } else if (filter_name == "gaussian") {
        return get_gaussian_blur();
    } else if (filter_name == "average") {
        return get_average_filter();
    } else if (filter_name == "sharpening") {
        return get_sharpening_filter();
    } else {
        throw std::runtime_error("Filter name not supported");
    }
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

    // Create a SYCL queue to submit work
    sycl::queue q(selector, exception_handler);

    // Print out the device information used for the kernel code.
    cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";

    // Load the image using OpenCV
    string input_name = "beach";
    // cv::Mat input = cv::imread(input_name + ".jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat input = cv::imread(input_name + ".jpg", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error: Could not open or find the image " << input_name << "!" << std::endl;
        return -1;
    }

    // Create an output image
    cv::Mat output(input.size(), input.type());

    std::vector<string> filter_names = {
        "laplacian",
        "gaussian",
        "average",
        "sharpening"
    };

    for (string filter_name : filter_names){
        std::vector<float> filter = get_filter(filter_name);
        size_t filter_size = 3;

        // Perform convolution
        convolution(q, input, output, filter, filter_size);

        // Save the output image
        string output_name = input_name + "-" + filter_name + "-output.jpg";
        cv::imwrite(output_name, output);

        std::cout << "Convolution completed and saved to " << output_name << "." << std::endl;
    }

    return 0;
}