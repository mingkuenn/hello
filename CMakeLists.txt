cmake_minimum_required(VERSION 3.14)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("include")

find_package (Eigen3 5.0.1 REQUIRED NO_MODULE)
 
add_executable (main src/main.cpp src/layers.cpp src/mnist_loader.cpp)
target_link_libraries (main Eigen3::Eigen)