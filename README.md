3DS-ARAP

Docker setup
1. Create the docker image: `docker build -t interactive-arap .`
2. Run the docker container: `docker run -it -v $(pwd):/workspace interactive-arap`

CMake build
1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`
5. `./main`