#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace MeshLoader {
    
    struct Mesh {
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::vector<int>> faces;
        bool isValid() const { return !vertices.empty() && !faces.empty(); }
    };
    
    // Load a PLY file and return mesh data
    Mesh loadPLY(const std::string& filepath);
    
    // Helper function to display mesh in polyscope
    void displayMesh(const Mesh& mesh, const std::string& name);
    
} // namespace MeshLoader