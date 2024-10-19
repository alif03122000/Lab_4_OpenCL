#!/bin/bash
gcc -D CL_TARGET_OPENCL_VERSION=100 cube.c -o cube -lOpenCL && ./cube