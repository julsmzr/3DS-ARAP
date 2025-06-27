#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/view.h>
#include "load_mesh.h"

struct ActiveMesh {
    MeshLoader::Mesh data;                  // actual mesh data
    polyscope::SurfaceMesh* view = nullptr; // polyscope handle
};

namespace Window {
  class Viewer {
    public:
      void init();
      void show();
  };

  void startViewer();

  // UI + picking entrypoints
  void setupUI();
  void vertexPickerCallback();
  void clearSelection();

  // globals
  extern std::unique_ptr<ActiveMesh> currentMesh;
  extern std::vector<Eigen::Vector3d> selectedPoints;
  extern polyscope::PointCloud*     highlightPoints;

  // deformation‐mode state
  extern bool                       deformationModeEnabled;
  extern polyscope::CameraParameters lockedCameraParams;

} // namespace Window
