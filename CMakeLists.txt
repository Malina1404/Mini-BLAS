cmake_minimum_required(VERSION 3.20)

project(Mini_BLAS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/O2>")

add_executable(Mini_BLAS main.cu)