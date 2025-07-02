// solver.h
#pragma once

#include <glm/glm.hpp>
#include <Eigen/Dense>
#include <vector>

namespace Solver {

// Screen projection utility
Eigen::Vector3d screenToWorld(
    const glm::vec2&   screenCoords,
    const Eigen::Vector3d& planePoint,
    const glm::vec3&   planeNormal);

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
    
    // Perform ARAP deformation (to be implemented later)
    void solveARAP();
    
    // Check if mesh is loaded
    bool hasMesh() const { return vertices_.rows() > 0 && faces_.rows() > 0; }

private:
    Eigen::MatrixXd vertices_;  // Nx3 vertex positions
    Eigen::MatrixXi faces_;     // Mx3 face indices
    
    // Constraint data for ARAP
    std::vector<int> constraintIndices_;
    std::vector<Eigen::Vector3d> constraintPositions_;
};

} // namespace Solver
