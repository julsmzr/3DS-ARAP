#include "solver.h"
#include "load_mesh.h"
#include <iostream>
#include <chrono>

double benchMarkSolver_us(const MeshLoader::Mesh& mesh, 
                        const std::vector<int>& constraintIndices,
                        const std::vector<Eigen::Vector3d>& constraintPoints, 
                        int iterations = 5, 
                        bool paperARAP = true,
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
    solver.setArapImplementation(paperARAP ? Solver::PAPER_ARAP : Solver::CERES_ARAP);
    solver.setNumberofIterations(iterations);
    solver.setSolverType(solverType);
    solver.setPaperSolverType(Solver::PAPER_CHOLESKY);  // Default to Cholesky for paper solver
    solver.setConstraints(constraintIndices, constraintPoints);

    auto startSampleTime = std::chrono::steady_clock::now();
    solver.solveARAP();
    auto finishedSampleTime = std::chrono::steady_clock::now();

    double dt = std::chrono::duration<double, std::micro>(finishedSampleTime - startSampleTime).count();

    return dt;
}

int main() {
    
    MeshLoader::Mesh mesh = MeshLoader::loadPLY("Data/drill_shaft_vrip.ply");
    
    // define a non rigid constraint
    std::vector<int> constraintIndices;
    std::vector<Eigen::Vector3d> constraintPoints;
    constraintIndices.emplace_back(888);
    constraintPoints.emplace_back(Eigen::Vector3d{0.204709, 132.755, 15.3112});    
    
    constraintIndices.emplace_back(1530);
    constraintPoints.emplace_back(Eigen::Vector3d{-0.635985, 137.134, 15.1851});   
    for (int i = 1; i < 20; i++) {
        double dt = benchMarkSolver_us(mesh, constraintIndices, constraintPoints, i);
        std::cerr << "Number of iterations: " << i << " : " << dt << " us" << std::endl;
    }
    
    return 0;
}
