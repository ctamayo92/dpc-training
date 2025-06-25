#!/bin/bash

source /opt/intel/oneapi/setvars.sh
# dpcpp matrix-multiplication.cpp -o test.bin && ./test.bin

dpcpp -I/usr/include/opencv4 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc convolution.cpp -o convolution.bin

./convolution.bin