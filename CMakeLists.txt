cmake_minimum_required(VERSION 3.14)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("include")

find_package (Eigen3 5.0.1 REQUIRED NO_MODULE)
find_package(HDF5 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
    
 
add_executable (main src/main.cpp src/layers.cpp src/mnist_loader.cpp src/mlp.cpp)
target_link_libraries (main PRIVATE
    Eigen3::Eigen 
    hdf5::hdf5-static 
    hdf5::hdf5_cpp-static 
    spdlog::spdlog_header_only)

