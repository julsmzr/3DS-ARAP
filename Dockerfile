FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install essential tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Download and install LibTorch C++ API
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip -O libtorch.zip \
 && unzip libtorch.zip -d /opt/ \
 && rm libtorch.zip

# Set environment variables for CMake to find LibTorch
ENV LIBTORCH_PATH=/opt/libtorch
ENV CMAKE_PREFIX_PATH=${LIBTORCH_PATH}
ENV LD_LIBRARY_PATH=${LIBTORCH_PATH}/lib:$LD_LIBRARY_PATH
