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

# Fetch ceres
FetchContent_Declare(ceres
  GIT_REPOSITORY  https://ceres-solver.googlesource.com/ceres-solver
  GIT_TAG         2.2.0
)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(PROVIDE_UNINSTALL_TARGET OFF)
set(MINIGLOG ON)

FetchContent_MakeAvailable(polyscope Eigen3 ceres)

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
target_link_libraries(main PRIVATE polyscope Eigen3::Eigen ceres)

# -------- Fetch GoogleTest --------
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

option(BUILD_ARAP_TESTS "Build ARAP tests" ON)
if(BUILD_ARAP_TESTS)
    enable_testing()

    add_executable(test_arap src/test_arap_solver.cpp src/load_mesh.cpp src/solver.cpp)
    target_include_directories(test_arap PRIVATE ${happly_SOURCE_DIR})
    target_link_libraries(test_arap
        PRIVATE
        polyscope
        Eigen3::Eigen
        gtest_main
        ceres
    )

    include(GoogleTest)
    gtest_discover_tests(test_arap)
endif()

# benchmark executable 
add_executable(benchmark src/benchmark.cpp src/viewer.cpp src/load_mesh.cpp src/solver.cpp)

# Ensure proper C++ settings
set_property(TARGET benchmark PROPERTY CXX_STANDARD 17)

# Include happly directory
target_include_directories(benchmark PRIVATE ${happly_SOURCE_DIR})

# Link Polyscope, Eigen and Ceres to the benchmark executable
target_link_libraries(benchmark PRIVATE polyscope Eigen3::Eigen ceres)