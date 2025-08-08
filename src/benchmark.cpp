#include "solver.h"
#include "load_mesh.h"
#include <happly.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <set>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <unordered_set>

std::tuple<double, Eigen::MatrixXd, Eigen::MatrixXi> benchMarkSolver_us(const MeshLoader::Mesh& mesh, 
                        const std::vector<int>& constraintIndices,
                        const std::vector<Eigen::Vector3d>& constraintPoints, 
                        int iterations = 5, 
                        Solver::ARAPImplementation implementation = Solver::PAPER_ARAP,
                        Solver::SolverType solverType = Solver::SolverType::CHOLESKY,
                        Solver::PaperSolverType paperSolverType = Solver::PaperSolverType::PAPER_CHOLESKY) {
    
    Solver::ARAPSolver solver;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
        
    vertices = Eigen::MatrixXd(mesh.vertices.size(), 3);
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        vertices.row(i) = mesh.vertices[i];
    }
    faces = Eigen::MatrixXi(mesh.faces.size(), 3);
    for (size_t i = 0; i < mesh.faces.size(); i++) {
        faces.row(i) = Eigen::Vector3i{mesh.faces[i][0], mesh.faces[i][1], mesh.faces[i][2]};
    }

    solver.setMesh(vertices, faces);
    solver.setArapImplementation(implementation);
    solver.setNumberofIterations(iterations);
    solver.setSolverType(solverType);
    solver.setPaperSolverType(paperSolverType);
    solver.setConstraints(constraintIndices, constraintPoints);

    auto startSampleTime = std::chrono::steady_clock::now();
    solver.solveARAP();
    auto finishedSampleTime = std::chrono::steady_clock::now();

    double dt = std::chrono::duration<double, std::micro>(finishedSampleTime - startSampleTime).count();

    return {dt, solver.getVertices(), solver.getFaces()};
}

std::string getImplementationName(Solver::ARAPImplementation impl) {
    switch (impl) {
        case Solver::PAPER_ARAP: return "PAPER_ARAP";
        case Solver::CERES_ARAP: return "CERES_ARAP";
        default: return "IGL_IMPL";
    }
}

std::string getSolverTypeName(Solver::SolverType type) {
    switch (type) {
        case Solver::SolverType::CHOLESKY: return "CHOLESKY";
        case Solver::SolverType::SPARSE_SCHUR: return "SPARSE_SCHUR";
        case Solver::SolverType::CGNR: return "CGNR";
        default: return "UNKNOWN_SOLVER";
    }
}

std::string getPaperSolverTypeName(Solver::PaperSolverType type) {
    switch (type) {
        case Solver::PaperSolverType::PAPER_CHOLESKY: return "PAPER_CHOLESKY";
        case Solver::PaperSolverType::PAPER_LDLT: return "PAPER_LDLT";
        default: return "UNKNOWN_PAPER_SOLVER";
    }
}

// Helper to extract unique undirected edges from triangle mesh
std::set<std::pair<int, int>> extractEdges(const Eigen::MatrixXi& faces) {
    std::set<std::pair<int, int>> edges;
    for (int i = 0; i < faces.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int v1 = faces(i, j);
            int v2 = faces(i, (j + 1) % 3);
            if (v1 > v2) std::swap(v1, v2);
            edges.emplace(v1, v2);
        }
    }
    return edges;
}

// Computes the relative RMS error of edge lengths between original and deformed meshes
double computeRelativeEdgeLengthError(const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V, const std::set<std::pair<int, int>>& edges) {
    double sumSquaredRelError = 0.0;
    for (const auto& [i, j] : edges) {
        double origLen = (V0.row(i) - V0.row(j)).norm();
        double newLen = (V.row(i) - V.row(j)).norm();
        if (origLen > 1e-8) {
            double relErr = (newLen - origLen) / origLen;
            sumSquaredRelError += relErr * relErr;
        }
    }
    return std::sqrt(sumSquaredRelError / edges.size());
}

void createDirectoryIfNotExists(const std::string& dirPath) {
  namespace fs = std::filesystem;
  try {
    if (!fs::exists(dirPath)) {
      if (fs::create_directories(dirPath)) {
        std::cout << "Created directory: " << dirPath << "\n";
      } else {
        std::cerr << "Failed to create directory: " << dirPath << "\n";
      }
    }
  } catch (const fs::filesystem_error& e) {
    std::cerr << "Filesystem error: " << e.what() << "\n";
  }
}

void writePLY(const std::string& filepath,
              const Eigen::MatrixXd& vertices,
              const Eigen::MatrixXi& faces) {
  try {
    if (vertices.rows() == 0 || faces.rows() == 0) {
      throw std::runtime_error("Invalid mesh data: missing vertices or faces");
    }

    happly::PLYData plyOut;

    // Scale down vertices (inverse of loading)
    const double SCALE = 1000.0;
    std::vector<double> xs, ys, zs;
    xs.reserve(vertices.rows());
    ys.reserve(vertices.rows());
    zs.reserve(vertices.rows());

    for (int i = 0; i < vertices.rows(); ++i) {
      xs.push_back(vertices(i, 0) / SCALE);
      ys.push_back(vertices(i, 1) / SCALE);
      zs.push_back(vertices(i, 2) / SCALE);
    }

    plyOut.addElement("vertex", vertices.rows());
    plyOut.getElement("vertex").addProperty<double>("x", xs);
    plyOut.getElement("vertex").addProperty<double>("y", ys);
    plyOut.getElement("vertex").addProperty<double>("z", zs);

    // Convert Eigen faces to vector<vector<int>>
    std::vector<std::vector<int>> faceList;
    faceList.reserve(faces.rows());

    for (int i = 0; i < faces.rows(); ++i) {
      std::vector<int> face;
      for (int j = 0; j < faces.cols(); ++j) {
        face.push_back(faces(i, j));
      }
      faceList.push_back(face);
    }

    plyOut.addElement("face", faceList.size());
    plyOut.getElement("face").addListProperty<int>("vertex_indices", faceList);

    // Write to file
    std::ofstream outFile(filepath, std::ios::binary);
    plyOut.write(outFile, happly::DataFormat::Binary);
    std::cout << "Saved mesh to " << filepath << "\n";

  } catch (const std::exception& e) {
    std::cerr << "PLY write error: " << e.what() << "\n";
  }
}

int main() {
    // Load mesh
    std::string mesh_name = "cactus_highres";

    // Constraints
    // Cactus example
    std::vector<int> constraintIndices = {354, 611, 612, 609, 608, 594, 300, 292, 290, 288, 291, 315, 595, 306, 304, 584, 617, 358, 615, 591};
    std::vector<Eigen::Vector3d> constraintPoints = {
    {514.819,106.442,559.996}, 
    {551.175,104.154,550.823}, 
    {576.554,104.41,517.175}, 
    {576.298,103.994,486.737}, 
    {568.357,101.746,465.853}, 
    {551.45,104.969,448.676}, 
    {522.309,98.2256,440.738}, 
    {484.765,112.011,448.654}, 
    {462.839,89.6076,481.665}, 
    {460.058,102.087,511.853}, 
    {459.604,104.058,484.115},
    {462.064,89.624,506.189}, 
    {472.301,80.6303,532.403}, 
    {467.741,102.481,533.089}, 
    {483.399,118.989,551.607}, 
    {491.324,95.0022,551.097}, 
    {537.063,81.6816,551.736}, 
    {572.573,82.2856,517.995}, 
    {573.008,81.0372,489.674}, 
    {877.991,793.445,517.462}
    };
    
    /* 
    //Dino example
    std::string mesh_name = "dino";
    std::vector<int> constraintIndices = {
        318, 887, 712, 607, 1196, 1164, 725, 1283, 1699, 1531,
        1280, 967, 783, 681, 274, 1009, 878, 273, 275, 479,
        830, 1412, 1628, 977, 854, 929, 1539, 1620, 1115,
        753, 1298, 968, 1119, 1481, 1200, 798, 790, 913,
        1516, 797, 1129, 1279, 774, 1477, 1483, 569, 711,
        1474, 617, 658, 1062, 557, 616, 568, 1269, 1292,
        1675, 1876, 1576, 1014, 1191, 1555, 726, 882,
        866, 647, 834, 1514, 1307, 1284, 1846, 1750,
        1336, 1225, 965, 1421, 313, 1110, 918, 747, 1359, 
        1510, 242, 1053, 1236, 1228, 776, 1109, 668, 1253};

    std::vector<Eigen::Vector3d> constraintPoints = {
        {-86.7136,-363.316,-48.0333},
        {-114.251,-357.482,-51.9913},
        {-106.982,-360.691,-46.8479},
        {-79.904,-358.756,-48.1925},
        {-73.2094,-343.282,-50.5862},
        {-36.5717,-341.666,-44.4499},
        {-47.608,-363.084,-38.8237},
        {-55.7404,-340.446,-51.0507},
        {-40.6306,-323.995,-57.7216},
        {-76.2248,-332.14,-80.0458},
        {-85.0842,-337.365,-58.3267},
        {-104.833,-350.645,-85.7179},
        {-131.433,-360.055,-86.8409},
        {-116.148,-362.782,-45.0205},
        {-98.0093,-363.529,-46.8853},
        {-85.4869,-347.245,-49.8231},
        {-108.181,-358.173,-78.0957},
        {-86.7932,-354.001,-48.5084},
        {-88.246,-369.679,-47.7132},
        {-110.167,-369.019,-42.5158},
        {-57.7445,-351.687,-42.6125},
        {-61.5921,-329.642,-53.2655},
        {-53.3277,-320.63,-56.4875},
        {-51.2156,-340.925,-48.4944},
        {-33.8864,-355.169,-44.1045},
        {-22.0064,-345.063,-55.5127},
        {-28.7798,-330.709,-52.2243},
        {-20.8889,-322.264,-60.9953},
        {-11.4741,-335.386,-68.7544},
        {-11.4725,-356.948,-67.2934},
        {-3.37315,-338.17,-78.152},
        {-1.74179,-350.299,-78.8662},
        {-0.544113,-349.698,-86.6997},
        {-3.95415,-325.344,-86.7177},
        {-2.10561,-340.987,-93.5959},
        {-2.72208,-351.913,-98.254},
        {-0.73349,-355.343,-83.7826},
        {-6.64719,-349.254,-106.243},
        {-4.70683,-331.899,-97.2088},
        {-9.00015,-356.658,-108.924},
        {-28.5468,-348.803,-120.844},
        {-12.6608,-335.372,-110.419},
        {-21.8522,-353.903,-117.961},
        {-19.72,-325.626,-113.005},
        {-35.887,-333.052,-120.226},
        {-57.488,-358.102,-130.258},
        {-39.7776,-356.015,-124.734},
        {-23.7858,-330.93,-116.441},
        {-43.0454,-342.83,-124.503},
        {-29.0023,-356.525,-121.941},
        {-15.8343,-341.62,-114.442},
        {-54.1109,-362.069,-128.686},
        {-51.6333,-344.648,-128.652},
        {-67.3895,-352.103,-134.791},
        {-46.7529,-327.011,-125.236},
        {-64.3051,-331.515,-126.602},
        {-50.8322,-318.242,-121.516},
        {-38.0321,-316.06,-118.576},
        {-32.2981,-320.819,-117.698},
        {-77.3779,-347.641,-134.714},
        {-63.2279,-339.519,-130.669},
        {-64.4878,-326.999,-123.906},
        {-109.677,-359.936,-99.4251},
        {-88.2262,-351.022,-134.626},
        {-101.027,-358.766,-138.163},
        {-72.6557,-358.224,-137.197},
        {-89.7877,-358.458,-137.9},
        {-71.6887,-328.069,-114.91},
        {-80.4516,-339.835,-104.241},
        {-79.5979,-337.421,-115.403},
        {-68.8871,-320.485,-91.5428},
        {-63.369,-318.784,-100.713},
        {-88.2908,-336.227,-96.2631},
        {-98.9591,-344.287,-94.3267},
        {-100.897,-353.311,-98.4637},
        {-73.3654,-330.858,-101.362},
        {-98.2361,-357.261,-115.776},
        {-89.2106,-345.03,-120.811},
        {-101.544,-357.343,-126.933},
        {-106.631,-355.896,-67.678},
        {-86.0723,-337.727,-68.7238},
        {-71.7758,-327.245,-71.1866},
        {-113.878,-362.189,-65.6129},
        {-98.8138,-347.954,-69.4689},
        {-93.524,-339.739,-81.3666},
        {-95.3585,-344.817,-57.0553},
        {-90.1291,-350.342,-110.462}, 
        {-92.7347,-347.379,-124.751},
        {-109.731,-361.985,-134.642},
        {-235.204,-335.379,141.738}
    };
    */

    /* 
    //Bar example
    std::string mesh_name = "bar2";
    std::vector<int> constraintIndices = {
        66, 67, 68, 69, 70, 54, 53, 52, 51, 50, 71, 55, 56, 72,
        73, 57, 58, 74, 75, 59, 60, 76, 77, 61, 78, 62, 79, 63,
        64, 80, 81, 65, 37 
    };
    std::vector<Eigen::Vector3d> constraintPoints = {
        {1262.33,16193,3.38281},
        {8255.67,16193,3.38281},
        {15249,16193,3.38281},
        {21983.3,16193,3.38281},
        {28459.8,15927.6,0.371826},
        {30052.1,7182.11,2.38281},
        {23299.3,7446.75,-0.396973},
        {15508.6,7978.12,-0.163574},
        {7459.65,7978.12,-0.163574},
        {969.645,8243.75,-0.059082},
        {30179.9,15813.5,-7834.71},
        {30177.1,8765.23,-7859.33},
        {30177.6,8352.25,-15002},
        {30184.3,15206.7,-14764.3},
        {30179.8,16001.2,-23040.8},
        {30182,7916.03,-22978.2},
        {30180,7981.08,-29794.2},
        {30180,14671.2,-29879.1},
        {23572.5,15646.4,-30472.7},
        {23059.2,9463.98,-30474.1},
        {14109.9,9191.88,-30477.7},
        {14848.5,15426.6,-30476.6},
        {7011.4,15470.2,-30477.2},
        {7313.54,8026.65,-30475},
        {610.246,15795.2,-30475.1},
        {937.705,9031.27,-30477},
        {314.741,16101.4,-22668.4},
        {318.612,9810.38,-22656.6},
        {318.992,9524.77,-15834.1},
        {318.608,16578.1,-15571.6},
        {317.076,15540.4,-7089.07},
        {315.044,9661.1,-7375.36},
        {-77755.6,107510,6598.36}
    };

    */

    int maxIterations = 20;
    std::unordered_set<int> saveIterations = {1, 2, 4, 10, 15, 21, 25};
    saveIterations.insert(maxIterations - 1);

    std::stringstream inMeshFileName;
    inMeshFileName << "Data/" << mesh_name << ".ply";
    createDirectoryIfNotExists("benchmark_data/" + mesh_name);

    MeshLoader::Mesh mesh = MeshLoader::loadPLY(inMeshFileName.str());

    // Convert mesh vertices to Eigen
    Eigen::MatrixXd V0(mesh.vertices.size(), 3);
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        V0.row(i) = mesh.vertices[i];
    }

    Eigen::MatrixXi faces(mesh.faces.size(), 3);
    for (size_t i = 0; i < mesh.faces.size(); ++i) {
        faces.row(i) = Eigen::Vector3i{mesh.faces[i][0], mesh.faces[i][1], mesh.faces[i][2]};
    }

    std::set<std::pair<int, int>> edgeSet = extractEdges(faces);

    // Generate reference solution using IGL with high iterations
    int referenceIterations = 100;
    auto [_, targetVertices, targetFaces] = benchMarkSolver_us(mesh, constraintIndices, constraintPoints, 
                                                               referenceIterations, Solver::ARAPImplementation::IGL_ARAP, 
                                                               Solver::SolverType::CHOLESKY, Solver::PaperSolverType::PAPER_CHOLESKY);

    // Save baseline mesh
    std::stringstream baselinePlyFileName;
    baselinePlyFileName << "benchmark_data/" << mesh_name << "/arap_mesh_baseline.ply";
    writePLY(baselinePlyFileName.str(), targetVertices, targetFaces);

    // Define all solver combinations to test
    std::vector<std::tuple<Solver::ARAPImplementation, Solver::SolverType, Solver::PaperSolverType, std::string>> solverConfigs = {
        // PAPER_ARAP with different paper solver types
        {Solver::PAPER_ARAP, Solver::CHOLESKY, Solver::PAPER_CHOLESKY, "PAPER_ARAP_PAPER_CHOLESKY"},
        {Solver::PAPER_ARAP, Solver::CHOLESKY, Solver::PAPER_LDLT, "PAPER_ARAP_PAPER_LDLT"},
        {Solver::PAPER_ARAP, Solver::CHOLESKY, Solver::PAPER_LU, "PAPER_ARAP_PAPER_LU"},
        
        // CERES_ARAP with different ceres solver types
        {Solver::CERES_ARAP, Solver::CHOLESKY, Solver::PAPER_CHOLESKY, "CERES_ARAP_CHOLESKY"},
        {Solver::CERES_ARAP, Solver::SPARSE_SCHUR, Solver::PAPER_CHOLESKY, "CERES_ARAP_SPARSE_SCHUR"},
        //{Solver::CERES_ARAP, Solver::CGNR, Solver::PAPER_CHOLESKY, "CERES_ARAP_CGNR"},
        
        // IGL_ARAP 
        {Solver::IGL_ARAP, Solver::CHOLESKY, Solver::PAPER_CHOLESKY, "IGL_ARAP_CHOLESKY"}
    };

    // Run benchmarks for all solver configurations
    for (const auto& [implementation, solverType, paperSolverType, configName] : solverConfigs) {
        std::cout << "\n=== Running benchmark for: " << configName << " ===" << std::endl;
        
        // Create output files
        std::stringstream fileName;
        fileName << "benchmark_data/" << mesh_name << "/arap_benchmark_" << configName << ".csv";

        std::stringstream plyDirName;
        plyDirName << "benchmark_data/" << mesh_name << "/arap_meshes_" << configName;
        createDirectoryIfNotExists(plyDirName.str());

        std::ofstream outFile(fileName.str());
        if (!outFile.is_open()) {
            std::cerr << "Failed to open output file: " << fileName.str() << std::endl;
            continue;
        }

        outFile << "Iterations,Time(us),VertexChangeNorm,TargetError,EdgeLengthRelRMSError\n";

        Eigen::MatrixXd prevVertices;
        Eigen::MatrixXi prevFaces;

        // Run iterations for this solver configuration
        for (int i = 1; i < maxIterations; ++i) {
            try {
                auto [dt, vertices, faces] = benchMarkSolver_us(mesh, constraintIndices, constraintPoints, 
                                                              i, implementation, solverType, paperSolverType);

                double vertexChangeRMSE = 0.0;
                if (i > 1) {
                    Eigen::MatrixXd diff = vertices - prevVertices;
                    vertexChangeRMSE = std::sqrt((diff.array().square().sum()) / diff.size());
                }

                Eigen::MatrixXd targetDiff = vertices - targetVertices;
                double targetErrorRMSE = std::sqrt((targetDiff.array().square().sum()) / targetDiff.size());

                double edgeLengthRelRMSE = computeRelativeEdgeLengthError(V0, vertices, edgeSet);

                std::cout << configName << " - Iterations: " << i
                          << " | Time: " << dt << " us"
                          << " | ΔV (RMSE): " << vertexChangeRMSE
                          << " | Target RMSE: " << targetErrorRMSE
                          << " | Edge RMSE: " << edgeLengthRelRMSE << std::endl;

                outFile << i << "," << dt << "," << vertexChangeRMSE << "," << targetErrorRMSE << "," << edgeLengthRelRMSE << "\n";

                prevVertices = vertices;
                prevFaces = faces;
                
                if (saveIterations.count(i)) {
                    std::stringstream iterPlyFile;
                    iterPlyFile << plyDirName.str() << "/" << mesh_name << "_i" << i << ".ply";
                    writePLY(iterPlyFile.str(), vertices, faces);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error running " << configName << " at iteration " << i << ": " << e.what() << std::endl;
                break;
            }
        }

        outFile.close();
        std::cout << "Completed benchmark for: " << configName << std::endl;
    }

    std::cout << "\n=== All benchmarks completed! ===" << std::endl;
    return 0;
}