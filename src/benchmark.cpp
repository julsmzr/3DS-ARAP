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
                        Solver::SolverType solverType = Solver::SolverType::CHOLESKY) {
    
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
    solver.setPaperSolverType(Solver::PAPER_CHOLESKY);  // Default to Cholesky for paper solver
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
        default: return "UNKNOWN_SOLVER";
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

    Solver::ARAPImplementation implementation = Solver::CERES_ARAP;
    Solver::SolverType solverType = Solver::SolverType::CHOLESKY;

    // Constraints
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

    int maxIterations = 20;
    std::unordered_set<int> saveIterations = {1, 4, 10};
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

    int referenceIterations = 100;
    auto [_, targetVertices,targetFaces] = benchMarkSolver_us(mesh, constraintIndices, constraintPoints, referenceIterations, Solver::ARAPImplementation::IGL_ARAP, solverType);

    std::stringstream fileName;
    fileName << "benchmark_data/" << mesh_name <<"/arap_benchmark_"
             << getImplementationName(implementation) << "_"
             << getSolverTypeName(solverType) << ".csv";

    std::stringstream plyDirName;
    plyDirName << "benchmark_data/" << mesh_name <<"/arap_meshes_"
             << getImplementationName(implementation) << "_"
             << getSolverTypeName(solverType);
    createDirectoryIfNotExists(plyDirName.str());
    

    std::stringstream baselinePlyFileName;
    baselinePlyFileName << "benchmark_data/" << mesh_name <<"/arap_mesh_baseline.ply";
    writePLY(baselinePlyFileName.str(), targetVertices, targetFaces);

    std::ofstream outFile(fileName.str());
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file!" << std::endl;
        return 1;
    }

    outFile << "Iterations,Time(us),VertexChangeNorm,TargetError,EdgeLengthRelRMSError\n";

    Eigen::MatrixXd prevVertices;
    Eigen::MatrixXi prevFaces;

    for (int i = 1; i < maxIterations; ++i) {
        auto [dt, vertices, faces] = benchMarkSolver_us(mesh, constraintIndices, constraintPoints, i, implementation, solverType);

        double vertexChangeRMSE = 0.0;
        if (i > 1) {
            Eigen::MatrixXd diff = vertices - prevVertices;
            vertexChangeRMSE = std::sqrt((diff.array().square().sum()) / diff.size());
        }

        Eigen::MatrixXd targetDiff = vertices - targetVertices;
        double targetErrorRMSE = std::sqrt((targetDiff.array().square().sum()) / targetDiff.size());

        double edgeLengthRelRMSE = computeRelativeEdgeLengthError(V0, vertices, edgeSet);

        std::cerr << "Iterations: " << i
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
    }

    outFile.close();

    return 0;
}