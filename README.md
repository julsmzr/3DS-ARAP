# 3DS Interactive ARAP

### Docker setup
1. Create the docker image: `docker build -t interactive-arap .`
> For Apple Silicon, try: `docker build --platform=linux/amd64 -t interactive-arap .`
2. Run the docker container: `docker run -it -v $(pwd):/workspace interactive-arap`

### CMake build 
```bash
mkdir -p build && cd build
# Optional: Release build for heavy optimization
cmake -DCMAKE_BUILD_TYPE=Release .. 
cmake --build .
cd .. && ./build/main
```
