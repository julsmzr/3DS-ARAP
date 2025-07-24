// solver.h
#pragma once

#include <glm/glm.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <set>

#include "ceresSolver.h"

namespace Solver {

// Screen projection utility
Eigen::Vector3d screenToWorld(
    const glm::vec2&   screenCoords,
    const Eigen::Vector3d& planePoint,
    const glm::vec3&   planeNormal);


enum ARAPImplementation {
    PAPER_ARAP,
    CERES_ARAP,
    IGL_ARAP  // IGL implementation (mocked for now)
};

enum SolverType {
    CHOLESKY,
    SPARSE_SCHUR,
    CGNR
};

enum PaperSolverType {
    PAPER_CHOLESKY,
    PAPER_LDLT
};

// ARAP Solver class to manage mesh data and deformation
class ARAPSolver {
public:
    // Initialize with mesh data
    void setMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces);
    
    // Get current mesh data
    const Eigen::MatrixXd& getVertices() const { return vertices_; }
    const Eigen::MatrixXi& getFaces() const { return faces_; }
    
    // Set constraint vertices (selected points)
    void setConstraints(const std::vector<int>& constraintIndices, 
                       const std::vector<Eigen::Vector3d>& constraintPositions);
    
    // Update a single vertex position (for interactive dragging)
    void updateVertex(int vertexIndex, const Eigen::Vector3d& newPosition);

    void setNumberofIterations(int numIter) {
        numberOfIterations = numIter;
    }

    void setArapImplementation(ARAPImplementation impl) {
        arapImplementation = impl;
    }

    void setPaperSolverType(PaperSolverType paperSolver) {
        paperSolverType = paperSolver;
    }

    ceres::LinearSolverType getSolverType() {
        switch(solvertype) {
            case CHOLESKY:
                return ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
            case SPARSE_SCHUR:
                return ceres::LinearSolverType::SPARSE_SCHUR; 
            case CGNR:
                return ceres::LinearSolverType::CGNR;
            default:
                return ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
        }
    }

    void setSolverType(SolverType solverType_) {
        solvertype = solverType_;
    }
    
    // Perform ARAP deformation
    void solveARAP();

    void solveARAPPaper();
    
    //Perform ARAP deformation using Ceres
    void solveARAPCeres();
    
    // Check if mesh is loaded
    bool hasMesh() const { return vertices_.rows() > 0 && faces_.rows() > 0; }

private:
    Eigen::MatrixXd vertices_;  // Nx3 vertex positions
    Eigen::MatrixXi faces_;     // Mx3 face indices
    
    // Constraint data for ARAP
    std::vector<int> constraintIndices_;
    std::vector<Eigen::Vector3d> constraintPositions_;
    
    // ARAP data structures
    //std::vector<std::vector<int>> neighbors_;
    std::vector<std::set<int>> neighbors_;
    std::vector<std::unordered_map<int, float>> weights_;
    bool weightsComputed_ = false;

    int numberOfIterations = 5;
    ARAPImplementation arapImplementation = PAPER_ARAP;
    SolverType solvertype = CHOLESKY;
    PaperSolverType paperSolverType = PAPER_CHOLESKY;
    
    // ARAP implementation
    void computeCotangentWeights();
    Eigen::Matrix3f computeOptimalRotation(int i, const std::vector<Eigen::Vector3f>& p, 
                                          const std::vector<Eigen::Vector3f>& p_prime);
    void buildLaplacianAndRHS(const std::vector<Eigen::Vector3f>& p, 
                             const std::vector<Eigen::Matrix3f>& R,
                             Eigen::SparseMatrix<float>& L, Eigen::MatrixXf& b);
};

} // namespace Solver
