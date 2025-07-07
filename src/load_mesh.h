#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <polyscope/surface_mesh.h>

namespace MeshLoader {
  struct Mesh {
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::vector<int>> faces;
    bool isValid() const {
      return !vertices.empty() && !faces.empty();
    }
  };

  Mesh loadPLY(const std::string& filepath);
  polyscope::SurfaceMesh* displayMesh(const Mesh& mesh, const std::string& name);

} // namespace MeshLoader