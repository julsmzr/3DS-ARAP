#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
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

// animation mode state
extern bool                       animationModeEnabled;
extern std::vector<Eigen::MatrixXd> precomputedMeshes;
extern std::atomic<bool>          isPrecomputing;
extern std::atomic<int>           precomputationProgress;
extern std::atomic<bool>          precomputationComplete;
extern bool                       isPlaying;
extern int                        currentFrame;
extern std::thread                precomputationThread;
extern std::mutex                 meshDataMutex;

} // namespace Window
