#include "viewer.h"
#include "load_mesh.h"

#include <polyscope/polyscope.h>
#include <polyscope/view.h>
#include <imgui.h>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
namespace Window {

polyscope::SurfaceMesh*    currentMesh            = nullptr;
std::vector<Eigen::Vector3d> selectedPoints;
polyscope::PointCloud*     highlightPoints        = nullptr;
bool                       deformationModeEnabled = false;
polyscope::CameraParameters lockedCameraParams;

void clearSelection() {
  selectedPoints.clear();
  if (highlightPoints) {
    polyscope::removeStructure(highlightPoints->name);
    highlightPoints = nullptr;
  }
  std::cout << "[Info] Selection cleared\n";
}

void setupUI() {
  if (ImGui::Button("Select Mesh...")) {
    ImGui::OpenPopup("Select Mesh");
  }
  ImGui::SameLine();
  if (currentMesh) {
    if (!deformationModeEnabled) {
      if (ImGui::Button("Clear Selection")) {
        clearSelection();
      }
      ImGui::SameLine();
    } else {
      ImGui::BeginDisabled();
      ImGui::Button("Clear Selection");
      ImGui::EndDisabled();
      ImGui::SameLine();
    }
    const char* label = deformationModeEnabled
      ? "Disable Deformation Mode"
      : "Enable Deformation Mode";
    if (ImGui::Button(label)) {
      if (!deformationModeEnabled) {
        lockedCameraParams = polyscope::view::getCameraParametersForCurrentView();
        deformationModeEnabled = true;
        std::cout << "[Info] Deformation mode ENABLED, camera frozen\n";
      } else {
        deformationModeEnabled = false;
        std::cout << "[Info] Deformation mode DISABLED, camera unlocked\n";
      }
    }
  }

  if (ImGui::BeginPopup("Select Mesh")) {
    const std::string dataDir = "Data";
    if (fs::is_directory(dataDir)) {
      for (auto& entry : fs::directory_iterator(dataDir)) {
        if (entry.path().extension() == ".ply") {
          auto fn = entry.path().filename().string();
          if (ImGui::Selectable(fn.c_str())) {
            polyscope::removeAllStructures();
            MeshLoader::Mesh mesh = MeshLoader::loadPLY(entry.path().string());
            if (mesh.isValid()) {
              std::string name = entry.path().stem().string();
              currentMesh = MeshLoader::displayMesh(mesh, name);
            }
            clearSelection();
            deformationModeEnabled = false;
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

  if (ImGui::IsMouseClicked(1)) {
    if (!deformationModeEnabled) {
      clearSelection();
    }
    return;
  }
  if (ImGui::IsMouseClicked(0)) {
    glm::vec2 mpos{io.MousePos.x, io.MousePos.y};
    auto pr = polyscope::pickAtScreenCoords(mpos);
    if (pr.isHit && pr.structure == currentMesh) {
      auto mpr = currentMesh->interpretPickResult(pr);
      if (mpr.elementType == polyscope::MeshElement::VERTEX) {
        std::cout
          << "pick: screen=("
          << pr.screenCoords.x << "," << pr.screenCoords.y
          << "), depth=" << pr.depth
          << ", world=("
          << pr.position.x << "," << pr.position.y << "," << pr.position.z
          << "), vertex=" << mpr.index
          << "\n";

        if (!deformationModeEnabled) {
          Eigen::Vector3d P{pr.position.x, pr.position.y, pr.position.z};
          selectedPoints.push_back(P);

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
}

void Viewer::init() {
  polyscope::init();
  polyscope::options::usePrefsFile = false;
  polyscope::options::programName = "Interactive ARAP Viewer";
  polyscope::options::verbosity   = 1;

  polyscope::state::userCallback = []() {
    setupUI();
    vertexPickerCallback();
    if (deformationModeEnabled) {
      polyscope::view::setViewToCamera(lockedCameraParams);
    }
  };
}

void Viewer::show() {
  polyscope::show();
}

void startViewer() {
  Viewer viewer;
  viewer.init();
  viewer.show();
}

} // namespace Window
