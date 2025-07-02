#include "viewer.h"
#include "load_mesh.h"
#include "solver.h"

#include <polyscope/polyscope.h>
#include <polyscope/view.h>
#include <polyscope/point_cloud.h>
#include <polyscope/curve_network.h>
#include <imgui.h>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <glm/glm.hpp>

namespace fs = std::filesystem;
namespace Window {

// globals
polyscope::SurfaceMesh*       currentMesh            = nullptr;
std::vector<Eigen::Vector3d>  selectedPoints;
std::vector<int>              selectedVertexIndices;
polyscope::PointCloud*        highlightPoints        = nullptr;
bool                          deformationModeEnabled = false;
polyscope::CameraParameters   lockedCameraParams;
Solver::ARAPSolver            solver;

// vertex dragging state
int                           draggedVertexIndex     = -1;

// drag‐state
static bool                        isDragging     = false;
static glm::vec2                   dragStartScreen;
static Eigen::Vector3d             dragPlanePoint;
static glm::vec3                   dragPlaneNormal;
static std::vector<Eigen::Vector3d> dragSamples;
static polyscope::CurveNetwork*    dragPath       = nullptr;

// sampling
static constexpr double sampleRate     = 60.0;
static constexpr double sampleInterval = 1.0 / sampleRate;
static auto lastSampleTime = std::chrono::steady_clock::now();

void updateMeshVisualization() {
    if (currentMesh && solver.hasMesh()) {
        currentMesh->updateVertexPositions(solver.getVertices());
    }
}

void updateConstraints() {
    // Update solver with current selection as constraints
    if (!selectedVertexIndices.empty() && !selectedPoints.empty()) {
        solver.setConstraints(selectedVertexIndices, selectedPoints);
    }
}

void clearSelection() {
  selectedPoints.clear();
  selectedVertexIndices.clear();
  if (highlightPoints) {
    polyscope::removeStructure(highlightPoints->name);
    highlightPoints = nullptr;
  }
  dragSamples.clear();
  if (dragPath) {
    polyscope::removeStructure(dragPath->name);
    dragPath = nullptr;
  }
  draggedVertexIndex = -1;
  
  // Clear solver constraints
  if (solver.hasMesh()) {
    solver.setConstraints({}, {});
  }
  
  std::cout << "[Info] Selection cleared\n";
}

void setupUI() {
  if (ImGui::Button("Select Mesh...")) ImGui::OpenPopup("Select Mesh");
  ImGui::SameLine();
  if (currentMesh) {
    if (!deformationModeEnabled) {
      if (ImGui::Button("Clear Selection")) clearSelection();
      ImGui::SameLine();
    } else {
      ImGui::BeginDisabled();
      ImGui::Button("Clear Selection");
      ImGui::EndDisabled();
      ImGui::SameLine();
    }
    const char* modeLabel = deformationModeEnabled
      ? "Disable Deformation Mode"
      : "Enable Deformation Mode";
    if (ImGui::Button(modeLabel)) {
      if (!deformationModeEnabled) {
        lockedCameraParams     = polyscope::view::getCameraParametersForCurrentView();
        deformationModeEnabled = true;
        updateConstraints();
        std::cout << "[Info] Deformation mode ENABLED\n";
      } else {
        deformationModeEnabled = false;
        isDragging            = false;
        std::cout << "[Info] Deformation mode DISABLED\n";
      }
    }
  }

  // Show solver info
  if (solver.hasMesh()) {
    ImGui::Separator();
    ImGui::Text("Solver Status:");
    ImGui::Text("Vertices: %d", (int)solver.getVertices().rows());
    ImGui::Text("Faces: %d", (int)solver.getFaces().rows());
    ImGui::Text("Selected: %d vertices", (int)selectedVertexIndices.size());
    
    if (ImGui::Button("Solve ARAP")) {
      solver.solveARAP();
      updateMeshVisualization();
    }
  }

  if (ImGui::BeginPopup("Select Mesh")) {
    const std::string dataDir = "Data";
    if (fs::is_directory(dataDir)) {
      for (auto& e : fs::directory_iterator(dataDir)) {
        if (e.path().extension() == ".ply") {
          auto fn = e.path().filename().string();
          if (ImGui::Selectable(fn.c_str())) {
            polyscope::removeAllStructures();
            clearSelection();
            deformationModeEnabled = false;
            auto M = MeshLoader::loadPLY(e.path().string());
            if (M.isValid()) {
              currentMesh = MeshLoader::displayMesh(M, e.path().stem().string());
              
              // Convert mesh data to Eigen format and pass to solver
              Eigen::MatrixXd vertices(M.vertices.size(), 3);
              for (size_t i = 0; i < M.vertices.size(); i++) {
                vertices.row(i) = M.vertices[i];
              }
              
              // Convert faces to Eigen format
              std::vector<std::array<int,3>> triangles;
              for (auto& f : M.faces) {
                if (f.size() == 3) {
                  triangles.push_back({f[0], f[1], f[2]});
                } else if (f.size() == 4) {
                  triangles.push_back({f[0], f[1], f[2]});
                  triangles.push_back({f[0], f[2], f[3]});
                } else {
                  for (size_t i = 1; i + 1 < f.size(); i++) {
                    triangles.push_back({f[0], f[i], f[i+1]});
                  }
                }
              }
              
              Eigen::MatrixXi faces(triangles.size(), 3);
              for (size_t i = 0; i < triangles.size(); i++) {
                faces.row(i) = Eigen::Vector3i{triangles[i][0], triangles[i][1], triangles[i][2]};
              }
              
              // Initialize solver with mesh data
              solver.setMesh(vertices, faces);
            }
            ImGui::CloseCurrentPopup();
          }
        }
      }
    } else {
      ImGui::Text("Data directory not found");
    }
    ImGui::EndPopup();
  }
}

void vertexPickerCallback() {
  if (!currentMesh) return;
  ImGuiIO& io = ImGui::GetIO();
  if (io.WantCaptureMouse) return;

  glm::vec2 mpos{ io.MousePos.x, io.MousePos.y };
  size_t w = polyscope::view::windowWidth;
  size_t h = polyscope::view::windowHeight;
  bool inside = mpos.x >= 0 && mpos.x < float(w) && mpos.y >= 0 && mpos.y < float(h);

  auto now = std::chrono::steady_clock::now();

  if (deformationModeEnabled) {
    // lock camera first
    polyscope::view::setViewToCamera(lockedCameraParams);

    // if dragging and cursor left window: end drag
    if (isDragging && !inside) {
      isDragging = false;
      draggedVertexIndex = -1;
      std::cout << "[Deform] end drag (cursor left window)\n";
      return;
    }

    // start drag
    if (ImGui::IsMouseClicked(0) && !isDragging && inside) {
      auto pr = polyscope::pickAtScreenCoords(mpos);
      if (pr.isHit && pr.structure == currentMesh) {
        auto mpr = currentMesh->interpretPickResult(pr);
        if (mpr.elementType == polyscope::MeshElement::VERTEX) {
          isDragging         = true;
          draggedVertexIndex = mpr.index;
          lastSampleTime     = now;
          dragStartScreen    = mpos;
          dragPlanePoint     = { pr.position.x, pr.position.y, pr.position.z };
          dragPlaneNormal    = polyscope::view::getFrontVec();
          dragSamples.clear();
          dragSamples.push_back(dragPlanePoint);
          if (dragPath) {
            polyscope::removeStructure(dragPath->name);
            dragPath = nullptr;
          }
          std::cout << "[Deform] start drag vertex=" << draggedVertexIndex
                    << " screen=(" << mpos.x << "," << mpos.y
                    << ") world=(" << dragPlanePoint.x() << ","
                    << dragPlanePoint.y() << "," << dragPlanePoint.z() << ")\n";
        }
      }
    }

    // sample drag
    if (isDragging && ImGui::IsMouseDown(0) && draggedVertexIndex >= 0) {
      double dt = std::chrono::duration<double>(now - lastSampleTime).count();
      if (dt >= sampleInterval && inside) {
        auto wp = Solver::screenToWorld(mpos, dragPlanePoint, dragPlaneNormal);
        std::cout << "[Deform] drag vertex=" << draggedVertexIndex
                  << " screen=(" << mpos.x << "," << mpos.y
                  << ") world=(" << wp.x() << "," << wp.y() << "," << wp.z() << ")\n";
        lastSampleTime = now;

        // Update the vertex position in solver and mesh
        if (draggedVertexIndex >= 0 && solver.hasMesh()) {
          solver.updateVertex(draggedVertexIndex, wp);
          updateMeshVisualization();
        }

        dragSamples.push_back(wp);
      }
    }

    // end drag
    if (isDragging && !ImGui::IsMouseDown(0)) {
      isDragging = false;
      draggedVertexIndex = -1;
      std::cout << "[Deform] end drag\n";
    }
    return;
  }

  // normal selection mode
  if (ImGui::IsMouseClicked(1)) {
    clearSelection();
    return;
  }
  if (ImGui::IsMouseClicked(0) && inside) {
    auto pr = polyscope::pickAtScreenCoords(mpos);
    if (pr.isHit && pr.structure == currentMesh) {
      auto mpr = currentMesh->interpretPickResult(pr);
      if (mpr.elementType == polyscope::MeshElement::VERTEX) {
        std::cout << "pick: screen=("
                  << pr.screenCoords.x << "," << pr.screenCoords.y
                  << "), depth=" << pr.depth
                  << ", world=("
                  << pr.position.x << "," << pr.position.y << "," << pr.position.z
                  << "), vertex=" << mpr.index << "\n";

        selectedPoints.emplace_back(pr.position.x, pr.position.y, pr.position.z);
        selectedVertexIndices.push_back(mpr.index);
        
        if (highlightPoints) {
          polyscope::removeStructure(highlightPoints->name);
        }
        Eigen::MatrixXd M(selectedPoints.size(), 3);
        for (size_t i = 0; i < selectedPoints.size(); i++) {
          M.row(i) = selectedPoints[i];
        }
        highlightPoints = polyscope::registerPointCloud("vertex highlight", M);
        highlightPoints->setPointRadius(0.004f);
        highlightPoints->setPointColor({1.0f, 0.0f, 0.0f});
      }
    }
  }
}

void Viewer::init() {
  polyscope::init();
  polyscope::options::usePrefsFile = false;
  polyscope::options::programName = "Interactive ARAP Viewer";
  polyscope::options::verbosity   = 1;
  polyscope::state::userCallback  = []() {
    if (deformationModeEnabled) {
      polyscope::view::setViewToCamera(lockedCameraParams);
    }
    setupUI();
    vertexPickerCallback();
  };
}

void Viewer::show() {
  polyscope::show();
}

void startViewer() {
  Viewer v;
  v.init();
  v.show();
}

} // namespace Window
