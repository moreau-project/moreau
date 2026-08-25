cmake_minimum_required(VERSION 3.22)
project(moreau LANGUAGES CXX CUDA)

# Delegate to moreau-cuda package for all CUDA builds
add_subdirectory(packages/moreau-cuda)
