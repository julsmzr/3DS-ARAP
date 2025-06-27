// solver.h
#pragma once

#include <glm/glm.hpp>
#include <Eigen/Dense>

namespace Solver {
    Eigen::Vector3d screenToWorld(
        const glm::vec2&   screenCoords,
        const Eigen::Vector3d& planePoint,
        const glm::vec3&   planeNormal
    );

} // namespace Solver
