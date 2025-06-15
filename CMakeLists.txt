cmake_minimum_required(VERSION 3.10)
project(InteractiveArap)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

# Create main executable
add_executable(main 
	src/main.cpp
	src/point_transformer_layer.cpp
	src/point_transformer_layer.h
	src/position_encoder.h
	src/point_transformer_block.h
)

# Add src/ to the include directories so headers in src/ are found
target_include_directories(main PRIVATE src)

target_link_libraries(main "${TORCH_LIBRARIES}")

# Ensure proper C++ settings
set_property(TARGET main PROPERTY CXX_STANDARD 17)