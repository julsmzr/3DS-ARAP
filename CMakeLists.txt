cmake_minimum_required(VERSION 3.10)
project(InteractiveArap)

set(CMAKE_CXX_STANDARD 17)

# Add Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Create main executable
add_executable(main src/main.cpp)
target_link_libraries(main Eigen3::Eigen)

# Ensure proper C++ settings
set_property(TARGET main PROPERTY CXX_STANDARD 17)