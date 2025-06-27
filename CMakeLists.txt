cmake_minimum_required(VERSION 3.10)
project(InteractiveArap)

set(CMAKE_CXX_STANDARD 17)

# Fetch Polyscope
include(FetchContent)

FetchContent_Declare(
   polyscope
   GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
   GIT_TAG        v2.4.0
)

# Fetch Eigen
FetchContent_Declare(
   Eigen3
   GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
   GIT_TAG        3.4.0
)

# Fetch happly
FetchContent_Declare(
   happly
   GIT_REPOSITORY https://github.com/nmwsharp/happly.git
   GIT_TAG        master
)

FetchContent_MakeAvailable(polyscope Eigen3)

# For happly, we need to handle it manually since it's header-only without a CMake config
FetchContent_GetProperties(happly)
if(NOT happly_POPULATED)
  FetchContent_Populate(happly)
endif()

# Create main executable
add_executable(main src/main.cpp src/viewer.cpp src/load_mesh.cpp src/solver.cpp)

# Ensure proper C++ settings
set_property(TARGET main PROPERTY CXX_STANDARD 17)

# Include happly directory
target_include_directories(main PRIVATE ${happly_SOURCE_DIR})

# Link Polyscope and Eigen to the main executable
target_link_libraries(main polyscope Eigen3::Eigen)