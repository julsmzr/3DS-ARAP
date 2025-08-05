# 3DS Interactive ARAP

We present **3DS Interactive ARAP**, an interactive **As-Rigid-As-Possible (ARAP)** mesh deformation system implemented in C++. The project features three different ARAP solver implementations, provides a flexible and extensible UI, and supports real-time visualizations, benchmarking, and error analysis.


##  Quick Start

### Docker Setup

```bash
# 1. Create the Docker image
docker build -t interactive-arap .

# 2. Run the container
docker run -it -v $(pwd):/workspace interactive-arap
```

> 💡 **Windows Users**: Use  
```bash
docker run -it -e DISPLAY=host.docker.internal:0.0 -v %cd%:/workspace interactive-arap
```
Make sure to install and run something like [XLaunch](https://sourceforge.net/projects/xming/) for GUI support for Windows.

---

### CMake Build

```bash
# Clean previous builds
rm -rf build
mkdir -p build && cd build

# Optional: Use Release build for performance
cmake -DCMAKE_BUILD_TYPE=Release .. 
cmake --build .

# Launch the application
cd .. && ./build/main
```

---

## Executables

The project produces several executables after building:

| Executable        | Description |
|-------------------|-------------|
| `main`            | Launches the main **interactive UI application**. Use this to explore mesh deformation, solver behavior, and visualization for any mesh in the `./Data` folder. |
| `benchmark`       | Runs **automated benchmarks** for a specified solver setup in `./src/benchmark.cpp`. Outputs quantitative results and error metrics to a `.csv` file. |
| `test_arap`| Performs a couple **unit tests** to verify correctness of the ARAP implementations. Useful for validation and debugging. |

---

## References

1. Sorkine, O., & Alexa, M. (2007). *As-Rigid-As-Possible Surface Modeling*. [PDF](https://igl.ethz.ch/projects/ARAP/)
2. Agarwal, S., Mierle, K., & Others. (Ceres Solver). *Ceres Solver: A Nonlinear Least Squares Minimizer*. http://ceres-solver.org/
3. Jacobson, A., et al. *libigl: A simple C++ geometry processing library*. https://libigl.github.io/