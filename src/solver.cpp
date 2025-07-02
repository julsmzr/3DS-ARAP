#include "solver.h"
#include <polyscope/view.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <iostream>

Eigen::Vector3d Solver::screenToWorld(
    const glm::vec2& screenCoords,
    const Eigen::Vector3d& planePoint,
    const glm::vec3& planeNormal) {

  // Get viewport size
  size_t w = polyscope::view::windowWidth;
  size_t h = polyscope::view::windowHeight;

  // 1) Build NDC coords
  float x_ndc = (screenCoords.x / float(w)) * 2.0f - 1.0f;
  float y_ndc = 1.0f - (screenCoords.y / float(h)) * 2.0f;
  glm::vec4 clip{x_ndc, y_ndc, -1.0f, 1.0f};

  // 2) Invert the projection matrix
  glm::mat4 invProj = glm::inverse(
    polyscope::view::getCameraPerspectiveMatrix()
  );
  glm::vec4 camSpace = invProj * clip;
  camSpace /= camSpace.w;

  // 3) Invert the view matrix
  glm::mat4 invView = glm::inverse(
    polyscope::view::getCameraViewMatrix()
  );
  glm::vec4 worldDir4 = invView * glm::vec4(camSpace.x, camSpace.y, camSpace.z, 0.0f);
  glm::vec3 rayDir = glm::normalize(glm::vec3(worldDir4));

  // 4) Ray‐plane intersection
  glm::vec3 origin = polyscope::view::getCameraWorldPosition();  
  glm::vec3 planeP{ float(planePoint.x()), float(planePoint.y()), float(planePoint.z()) };
  float denom = glm::dot(planeNormal, rayDir);
  if (std::fabs(denom) < 1e-6f) {
    return planePoint;
  }
  float t = glm::dot(planeP - origin, planeNormal) / denom;
  glm::vec3 W = origin + rayDir * t;
  return Eigen::Vector3d{ W.x, W.y, W.z };
}

// ARAPSolver implementation
namespace Solver {

void ARAPSolver::setMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces) {
    vertices_ = vertices;
    faces_ = faces;
    constraintIndices_.clear();
    constraintPositions_.clear();
    
    std::cout << "[Solver] Mesh loaded: " << vertices_.rows() << " vertices, " 
              << faces_.rows() << " faces" << std::endl;
}

void ARAPSolver::setConstraints(const std::vector<int>& constraintIndices, 
                               const std::vector<Eigen::Vector3d>& constraintPositions) {
    if (constraintIndices.size() != constraintPositions.size()) {
        std::cerr << "[Solver] Error: constraint indices and positions size mismatch" << std::endl;
        return;
    }
    
    constraintIndices_ = constraintIndices;
    constraintPositions_ = constraintPositions;
    
    std::cout << "[Solver] Constraints set: " << constraintIndices_.size() << " vertices" << std::endl;
}

void ARAPSolver::updateVertex(int vertexIndex, const Eigen::Vector3d& newPosition) {
    if (vertexIndex < 0 || vertexIndex >= vertices_.rows()) {
        std::cerr << "[Solver] Error: vertex index " << vertexIndex << " out of range" << std::endl;
        return;
    }
    
    vertices_.row(vertexIndex) = newPosition.transpose();
    std::cout << "[Solver] Updated vertex " << vertexIndex << " to " 
              << newPosition.transpose() << std::endl;
}

void ARAPSolver::solveARAP() {
    // TODO: Implement ARAP algorithm here
    // For now, this is a placeholder
    std::cout << "[Solver] ARAP solve called (not yet implemented)" << std::endl;
}

} // namespace Solver
