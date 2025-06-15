cmake_minimum_required(VERSION 3.10)
project(InteractiveArap)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

# Create main executable
add_executable(main src/main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")

# Ensure proper C++ settings
set_property(TARGET main PROPERTY CXX_STANDARD 17)