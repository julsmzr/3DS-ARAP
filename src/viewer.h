#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/view.h>

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
extern polyscope::SurfaceMesh*    currentMesh;
extern std::vector<Eigen::Vector3d> selectedPoints;
extern polyscope::PointCloud*     highlightPoints;

// deformation‐mode state
extern bool                       deformationModeEnabled;
extern polyscope::CameraParameters lockedCameraParams;

// vertex dragging state
extern int                        draggedVertexIndex;
extern Eigen::MatrixXd           originalVertices;

} // namespace Window
