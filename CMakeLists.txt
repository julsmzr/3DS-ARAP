cmake_minimum_required(VERSION 3.10)
project(InteractiveArap)

set(CMAKE_CXX_STANDARD 17)

# Fetch Polyscope
include(FetchContent)

FetchContent_Declare(
   polyscope
   GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
   GIT_TAG        master
)

FetchContent_MakeAvailable(polyscope)

# Create main executable
add_executable(main src/main.cpp src/viewer.cpp)

# Ensure proper C++ settings
set_property(TARGET main PROPERTY CXX_STANDARD 17)

# Link Polyscope to the main executable
target_link_libraries(main polyscope)