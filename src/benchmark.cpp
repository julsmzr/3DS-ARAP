#include "solver.h"
#include "load_mesh.h"
#include <iostream>
#include <chrono>
#include <fstream>

std::tuple<double, Eigen::MatrixXd> benchMarkSolver_us(const MeshLoader::Mesh& mesh, 
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

    return {dt, solver.getVertices()};
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

int main() {
    // Load mesh
    MeshLoader::Mesh mesh = MeshLoader::loadPLY("Data/drill_shaft_vrip.ply");

    // Constraint setup
    std::vector<int> constraintIndices = {888, 1530};
    std::vector<Eigen::Vector3d> constraintPoints = {
        {0.204709, 132.755, 15.3112},
        {-0.635985, 137.134, 15.1851}
    };

    Solver::ARAPImplementation implementation = Solver::CERES_ARAP;
    Solver::SolverType solverType = Solver::SolverType::CHOLESKY;

    int referenceIterations = 100;
    auto [_, targetVertices] = benchMarkSolver_us(mesh, constraintIndices, constraintPoints, referenceIterations, Solver::ARAPImplementation::IGL_ARAP, solverType);

    std::stringstream fileName;
    fileName << "benchmark_data/arap_benchmark_"
             << getImplementationName(implementation) << "_"
             << getSolverTypeName(solverType) << ".csv";

    std::ofstream outFile(fileName.str());
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file!" << std::endl;
        return 1;
    }

    outFile << "Iterations,Time(us),VertexChangeNorm,TargetError\n";

    Eigen::MatrixXd prevVertices;

    for (int i = 1; i < 20; i++) {
        auto [dt, vertices] = benchMarkSolver_us(mesh, constraintIndices, constraintPoints, i, implementation, solverType);

        double vertexChange = 0.0;
        if (i > 1) {
            vertexChange = (vertices - prevVertices).norm();
        }

        double targetError = (vertices - targetVertices).norm();

        std::cerr << "Iterations: " << i
                  << " | Time: " << dt << " us"
                  << " | ΔV: " << vertexChange
                  << " | Target Error: " << targetError << std::endl;

        outFile << i << "," << dt << "," << vertexChange << "," << targetError << "\n";

        prevVertices = vertices;
    }

    outFile.close();
    return 0;
}
