# 3DS Interactive ARAP

### Docker setup
1. Create the docker image: `docker build -t interactive-arap .`
2. Run the docker container: `docker run -it -v $(pwd):/workspace interactive-arap`

### CMake build 
```bash
rm -rf build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. # optimized
#cmake -DCMAKE_BUILD_TYPE=Release .. # for debugging
cmake --build .
cd .. && ./build/main
```
