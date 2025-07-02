#!/bin/bash

source /opt/intel/oneapi/setvars.sh
echo ""
echo "### Running basic matrix multiplication"
dpcpp matrix-multiplication.cpp -o multiplication.bin && ./multiplication.bin

echo ""
echo "### Running matrix multiplication with device memory optimization"
dpcpp matrix-multiplication-device-memory.cpp -o matrix-multiplication-device-memory.bin && ./matrix-multiplication-device-memory.bin

# dpcpp -I/usr/include/opencv4 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc convolution.cpp -o convolution.bin && ./convolution.bin