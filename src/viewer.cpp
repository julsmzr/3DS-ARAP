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
bool                          autoScalingDisabled    = false;
Solver::ARAPSolver            solver;

// vertex dragging state
int                           draggedVertexIndex     = -1;

// ARAP solving mode
static bool                   realTimeSolving        = false;

// drag‐state
static bool                        isDragging     = false;
static glm::vec2                   dragStartScreen;
static Eigen::Vector3d             dragPlanePoint;
static glm::vec3                   dragPlaneNormal;
static std::vector<Eigen::Vector3d> dragSamples;
static polyscope::CurveNetwork*    dragPath       = nullptr;

// Sampling constants
static constexpr float        TARGET_FPS             = 30.0f;
static constexpr float        INITIAL_SAMPLE_RATE    = 60.0f;
static constexpr float        INITIAL_SOLVE_TIME     = 0.016f;  // 16ms initial estimate
static constexpr float        MIN_SAMPLE_RATE        = 1.0f;
static constexpr float        MAX_SAMPLE_RATE        = 120.0f;
static constexpr float        EMA_ALPHA              = 0.2f;    // Exponential moving average factor

// Sampling state variables
static float                  targetFrameTime        = 1.0f / TARGET_FPS;
static float                  currentSampleRate      = INITIAL_SAMPLE_RATE;
static float                  avgSolveTime           = INITIAL_SOLVE_TIME;
static double                 currentSampleInterval  = 1.0 / INITIAL_SAMPLE_RATE;
static auto lastSampleTime = std::chrono::steady_clock::now();

// Disable Polyscope's automatic view adjustments
void disableAutoScaling() {
  if (!autoScalingDisabled) {
    // Disable auto-centering and scaling
    polyscope::options::automaticallyComputeSceneExtents = false;
    
    autoScalingDisabled = true;
    std::cout << "[View] Automatic view adjustments disabled\n";
  }
}

// Restore Polyscope's automatic view adjustments
void restoreAutoScaling() {
  if (autoScalingDisabled) {
    // Re-enable automatic centering and scaling
    polyscope::options::automaticallyComputeSceneExtents = true;
    
    autoScalingDisabled = false;
    std::cout << "[View] Automatic view adjustments restored\n";
  }
}

static void resetPerformanceMetrics() {
    currentSampleRate = INITIAL_SAMPLE_RATE;
    avgSolveTime = INITIAL_SOLVE_TIME;
    currentSampleInterval = 1.0 / INITIAL_SAMPLE_RATE;
    lastSampleTime = std::chrono::steady_clock::now();
}

void updateSampleRate() {
    if (realTimeSolving) {
        float adaptedRate = TARGET_FPS * (targetFrameTime / std::max(avgSolveTime, 0.001f));
        currentSampleRate = std::clamp(adaptedRate, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE);        
        currentSampleInterval = 1.0 / currentSampleRate;
    }
}

void updateMeshVisualization() {
    if (currentMesh && solver.hasMesh()) {
        if (deformationModeEnabled) {
            disableAutoScaling();
        }
        
        currentMesh->updateVertexPositions(solver.getVertices());
        
        if (deformationModeEnabled) {
            polyscope::view::setViewToCamera(lockedCameraParams);
        }
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
    if (ImGui::Button("Clear Selection")) {
      clearSelection();
      if (deformationModeEnabled) {
        // If in deformation mode, exit it since all constraints are cleared
        deformationModeEnabled = false;
        restoreAutoScaling();
        std::cout << "[Info] Deformation mode DISABLED (constraints cleared)\n";
      }
    }
    ImGui::SameLine();
    const char* modeLabel = deformationModeEnabled
      ? "Disable Deformation Mode"
      : "Enable Deformation Mode";
    if (ImGui::Button(modeLabel)) {
      if (!deformationModeEnabled) {
        lockedCameraParams     = polyscope::view::getCameraParametersForCurrentView();
        deformationModeEnabled = true;
        disableAutoScaling();
        updateConstraints();
        std::cout << "[Info] Deformation mode ENABLED\n";
      } else {
        deformationModeEnabled = false;
        isDragging            = false;
        restoreAutoScaling();
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
    
    // Toggle between real-time and on-demand solving
    if (ImGui::Checkbox("Real-time Solving", &realTimeSolving)) {
      if (realTimeSolving) {
        std::cout << "[Info] Switched to real-time ARAP solving\n";
        // Clear any dragged path when switching to real-time
        if (dragPath) {
          polyscope::removeStructure(dragPath->name);
          dragPath = nullptr;
        }
        updateSampleRate();
      } else {
        std::cout << "[Info] Switched to on-demand ARAP solving\n";
        resetPerformanceMetrics();
      }
    }
    
    // Sample rate controls (only show in real-time mode)
    if (realTimeSolving) {
      ImGui::Separator();
      ImGui::Text("Performance Stats:");
      ImGui::Text("Current Sample Rate: %.1f fps", currentSampleRate);
      ImGui::Text("Avg Solve Time: %.1f ms", avgSolveTime * 1000.0f);
    }
    
    // ARAP solve button (disabled in real-time mode)
    if (realTimeSolving) {
      ImGui::BeginDisabled();
    }
    if (ImGui::Button("Solve ARAP")) {
      solver.solveARAP();
      updateMeshVisualization();
      // Clear dragged path after solving
      if (dragPath) {
        polyscope::removeStructure(dragPath->name);
        dragPath = nullptr;
      }
    }
    if (realTimeSolving) {
      ImGui::EndDisabled();
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
            restoreAutoScaling();
            
            auto M = MeshLoader::loadPLY(e.path().string());
            if (M.isValid()) {
              currentMesh = MeshLoader::displayMesh(M, e.path().stem().string());
              polyscope::view::resetCameraToHomeView();
              
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
    // Ensure auto-scaling is disabled and lock camera
    disableAutoScaling();
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
      if (dt >= currentSampleInterval && inside) {
        auto wp = Solver::screenToWorld(mpos, dragPlanePoint, dragPlaneNormal);
        std::cout << "[Deform] drag vertex=" << draggedVertexIndex
                  << " screen=(" << mpos.x << "," << mpos.y
                  << ") world=(" << wp.x() << "," << wp.y() << "," << wp.z() << ")\n";
        lastSampleTime = now;

        // Update constraint position for dragged vertex
        if (draggedVertexIndex >= 0 && solver.hasMesh()) {
          // Set the dragged vertex as a constraint
          std::vector<int> constraintIndices = {draggedVertexIndex};
          std::vector<Eigen::Vector3d> constraintPositions = {wp};
          
          // Add existing selected vertices as fixed constraints
          for (size_t i = 0; i < selectedVertexIndices.size(); ++i) {
            int idx = selectedVertexIndices[i];
            if (idx != draggedVertexIndex) {
              constraintIndices.push_back(idx);
              constraintPositions.push_back(selectedPoints[i]);
            }
          }
          
          // Update the constraints
          solver.setConstraints(constraintIndices, constraintPositions);
          
          if (realTimeSolving) {
            // Real-time mode: solve ARAP immediately and measure performance
            auto solveStart = std::chrono::steady_clock::now();
            solver.solveARAP();
            auto solveEnd = std::chrono::steady_clock::now();
            
            float currentSolveTime = std::chrono::duration<float>(solveEnd - solveStart).count();
            avgSolveTime = (1.0f - EMA_ALPHA) * avgSolveTime + EMA_ALPHA * currentSolveTime;
            
            updateSampleRate();
          }
          
          updateMeshVisualization();
        }

        dragSamples.push_back(wp);
        
        // On-demand mode: show dragged path
        if (!realTimeSolving) {
          if (dragPath) {
            polyscope::removeStructure(dragPath->name);
            dragPath = nullptr;
          }
          int n = int(dragSamples.size());
          if (n >= 2) {
            Eigen::MatrixXd nodes(n,3);
            for (int i = 0; i < n; i++) nodes.row(i) = dragSamples[i];
            Eigen::MatrixXi edges(n-1,2);
            for (int i = 0; i < n-1; i++) edges.row(i) << i, i+1;
            dragPath = polyscope::registerCurveNetwork("drag path", nodes, edges);
            dragPath->setRadius(0.001);
            dragPath->setColor({0.0f, 0.0f, 1.0f});
          }
        }
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
  
  // Initialize performance metrics
  resetPerformanceMetrics();
  
  polyscope::state::userCallback  = []() {
    if (deformationModeEnabled) {
      disableAutoScaling();
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