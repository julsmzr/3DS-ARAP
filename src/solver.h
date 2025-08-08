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
    IGL_ARAP 
};

enum SolverType {
    CHOLESKY,
    SPARSE_SCHUR,
    CGNR
};

enum PaperSolverType {
    PAPER_CHOLESKY,
    PAPER_LDLT,
    PAPER_LU
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

    // Set number of iterations
    void setNumberofIterations(int numIter) {
        numberOfIterations = numIter;
    }

    // Set ARAP implementation
    void setArapImplementation(ARAPImplementation impl) {
        arapImplementation = impl;
    }

    // Set solver for the paper implementation
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

    // Set solver for the ceres implementation
    void setSolverType(SolverType solverType_) {
        solvertype = solverType_;
    }
    
    // Perform ARAP deformation for the chosen implementation
    void solveARAP();

    //Perform ARAP deformation according to the original ARAP alternating optimization
    void solveARAPPaper();
    
    //Perform ARAP deformation using ceres as a joint-optimization problem
    void solveARAPCeres();

    //Perform ARAP deformation using libigl
    void solveARAPIgl();
    
    // Check if mesh is loaded
    bool hasMesh() const { return vertices_.rows() > 0 && faces_.rows() > 0; }

private:
    Eigen::MatrixXd vertices_;  // Nx3 vertex positions
    Eigen::MatrixXi faces_;     // Mx3 face indices
    
    // Constraint data for ARAP
    std::vector<int> constraintIndices_;
    std::vector<Eigen::Vector3d> constraintPositions_;
    
    // ---- ARAP data structures ---- 
    std::vector<std::set<int>> neighbors_;
    std::vector<std::unordered_map<int, float>> weights_;
    bool weightsComputed_ = false;

    int numberOfIterations = 5;
    ARAPImplementation arapImplementation = PAPER_ARAP;
    SolverType solvertype = CHOLESKY;
    PaperSolverType paperSolverType = PAPER_CHOLESKY;
    
    // ---- ARAP implementation ----

    //Computes the neighbours for each vertex
    void computeNeighbours();

    //Jointly precomputes the neighbors and cotangent weights for each vertex
    void computeCotangentWeights();

    //Computes the optimal Rotation using the SVD approach as described in the original ARAP Paper
    Eigen::Matrix3f computeOptimalRotation(int i, const std::vector<Eigen::Vector3f>& p, 
                                          const std::vector<Eigen::Vector3f>& p_prime);

    // Builds the the discrete Laplace-Beltrami operator and sets up the right-hand side expression from equation (8) in the original ARAP paper thus formulating the Position step in the ARAP alternating optimization
    void buildLaplacianAndRHS(const std::vector<Eigen::Vector3f>& p, 
                             const std::vector<Eigen::Matrix3f>& R,
                             Eigen::SparseMatrix<float>& L, Eigen::MatrixXf& b);
};

} // namespace Solver
