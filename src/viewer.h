#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/view.h>
#include "solver.h"

namespace Window {
  class Viewer {
    public:
      void init();
      void show();
  };

  void startViewer();

// globals
extern polyscope::SurfaceMesh*    currentMesh;
extern std::vector<Eigen::Vector3d> selectedPoints;
extern std::vector<int>           selectedVertexIndices;  // Track vertex indices
extern polyscope::PointCloud*     highlightPoints;
extern Solver::ARAPSolver         solver;  // Solver instance

// deformation‐mode state
extern bool                       deformationModeEnabled;
extern polyscope::CameraParameters lockedCameraParams;

// vertex dragging state
extern int                        draggedVertexIndex;

} // namespace Window
