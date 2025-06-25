#!/bin/bash

source /opt/intel/oneapi/setvars.sh
dpcpp matrix-multiplication.cpp -o test.bin && ./test.bin