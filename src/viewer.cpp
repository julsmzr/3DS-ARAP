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
#include <thread>
#include <atomic>
#include <mutex>
#include <glm/glm.hpp>

namespace fs = std::filesystem;
namespace Window {

// globals
polyscope::SurfaceMesh*       currentMesh            = nullptr;
std::string                   currentMeshPath        = "";
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

// animation mode state
bool                          animationModeEnabled   = false;
std::vector<Eigen::MatrixXd>  precomputedMeshes;
std::atomic<bool>             isPrecomputing{false};
std::atomic<int>              precomputationProgress{0};
std::atomic<bool>             precomputationComplete{false};
bool                          isPlaying              = false;
int                           currentFrame            = 0;
std::thread                   precomputationThread;
std::mutex                    meshDataMutex;

// UI state for solver configuration
int                           selectedArapImplementation = 0;  // 0=Paper, 1=Ceres, 2=IGT
int                           selectedCeresSolver = 0;         // 0=Cholesky, 1=Sparse Schur, 2=CGNR
int                           selectedPaperSolver = 0;         // 0=Cholesky, 1=LDLT
int                           iterations = 5;                  // Number of ARAP iterations

// Status message
std::string                   statusMessage = "Ready";

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

// Forward declarations
void precomputeAnimation(const std::vector<Eigen::Vector3d>& pathSamples, 
                        int draggedVertex, 
                        const std::vector<int>& fixedIndices,
                        const std::vector<Eigen::Vector3d>& fixedPositions);
void playAnimation();
void cleanupAnimation();
void loadMesh(const std::string& meshPath);

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
  
  // Clear animation state
  if (animationModeEnabled) {
    animationModeEnabled = false;
    isPlaying = false;
    currentFrame = 0;
    if (precomputationThread.joinable()) {
      isPrecomputing = false;
      precomputationThread.join();
    }
    {
      std::lock_guard<std::mutex> lock(meshDataMutex);
      precomputedMeshes.clear();
    }
    precomputationComplete = false;
    precomputationProgress = 0;
  }
  
  // Clear solver constraints
  if (solver.hasMesh()) {
    solver.setConstraints({}, {});
  }
  
  statusMessage = "Selection cleared";
  std::cout << "[Info] Selection cleared\n";
}

void loadMesh(const std::string& meshPath) {
  polyscope::removeAllStructures();
  clearSelection();

  deformationModeEnabled = false;
  restoreAutoScaling();
  
  auto M = MeshLoader::loadPLY(meshPath);
  if (M.isValid()) {
    fs::path path(meshPath);
    currentMesh = MeshLoader::displayMesh(M, path.stem().string());
    currentMeshPath = meshPath;
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
    
    // Initialize solver settings based on UI state
    solver.setArapImplementation(static_cast<Solver::ARAPImplementation>(selectedArapImplementation));
    solver.setSolverType(static_cast<Solver::SolverType>(selectedCeresSolver));
    solver.setPaperSolverType(static_cast<Solver::PaperSolverType>(selectedPaperSolver));
    solver.setNumberofIterations(iterations);
    
    statusMessage = "Mesh loaded: " + path.filename().string() + " (" + 
                   std::to_string(vertices.rows()) + " vertices, " + 
                   std::to_string(faces.rows()) + " faces)";
    std::cout << "[Info] Mesh loaded: " << path.filename().string() << std::endl;
  } else {
    statusMessage = "Failed to load mesh: " + fs::path(meshPath).filename().string();
    std::cout << "[Error] Failed to load mesh: " << meshPath << std::endl;
  }
}

void setupUI() {
  // =================================
  // MESH SELECTION GROUP
  // =================================
  ImGui::Text("Mesh Selection");
  ImGui::Separator();
  if (ImGui::Button("Select Mesh...")) ImGui::OpenPopup("Select Mesh");
  
  ImGui::SameLine();
  if (currentMesh && !currentMeshPath.empty()) {
    if (ImGui::Button("Reset Mesh")) {
      loadMesh(currentMeshPath);
      statusMessage = "Mesh reset to original state";
    }
  }

  // =================================
  // ARAP CONFIGURATION GROUP
  // =================================
  if (currentMesh) {
    ImGui::Text("ARAP Configuration");
    ImGui::Separator();
    if (ImGui::Button("Clear Selection")) {
      clearSelection();
      if (deformationModeEnabled) {
        // If in deformation mode, exit it since all constraints are cleared
        deformationModeEnabled = false;
        restoreAutoScaling();
        statusMessage = "Deformation mode DISABLED (constraints cleared)";
      } else {
        statusMessage = "Selection cleared";
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
          statusMessage = "Deformation mode ENABLED";
        } else {
          deformationModeEnabled = false;
          isDragging            = false;
          restoreAutoScaling();
          statusMessage = "Deformation mode DISABLED";
        }
      }
      
      ImGui::Separator();
      
      // Real-time solving toggle (disabled in animation mode)
      if (animationModeEnabled) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Checkbox("Real-time Solving", &realTimeSolving)) {
        if (realTimeSolving) {
          statusMessage = "Switched to real-time ARAP solving";
          // Clear any dragged path when switching to real-time
          if (dragPath) {
            polyscope::removeStructure(dragPath->name);
            dragPath = nullptr;
          }
          updateSampleRate();
        } else {
          statusMessage = "Switched to on-demand ARAP solving";
          resetPerformanceMetrics();
        }
      }
      if (animationModeEnabled) {
        ImGui::EndDisabled();
      }
      
      // Performance stats (only show in real-time mode)
      if (realTimeSolving) {
        ImGui::Indent();
        ImGui::Text("Current Sample Rate: %.1f fps", currentSampleRate);
        ImGui::Text("Avg Solve Time: %.1f ms", avgSolveTime * 1000.0f);
        ImGui::Unindent();
      }
      
      // Animation mode toggle (only available when real-time solving is disabled)
      if (realTimeSolving) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Checkbox("Animation Mode", &animationModeEnabled)) {
        if (animationModeEnabled) {
          statusMessage = "Animation mode ENABLED";
          isPlaying = false;
          currentFrame = 0;
          precomputationComplete = false;
          precomputationProgress = 0;
        } else {
          statusMessage = "Animation mode DISABLED";
          isPlaying = false;
          currentFrame = 0;
          // Stop precomputation if running
          if (precomputationThread.joinable()) {
            isPrecomputing = false;
            precomputationThread.join();
          }
          {
            std::lock_guard<std::mutex> lock(meshDataMutex);
            precomputedMeshes.clear();
          }
          precomputationComplete = false;
          precomputationProgress = 0;
        }
      }
      if (realTimeSolving) {
        ImGui::EndDisabled();
      }
      
      // Animation mode controls
      if (animationModeEnabled) {
        ImGui::Indent();
        
        if (isPrecomputing) {
          // Show loading progress
          ImGui::Text("Precomputing frames...");
          float progress = static_cast<float>(precomputationProgress.load()) / 50.0f;
          ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), 
                            (std::to_string(precomputationProgress.load()) + "/50").c_str());
          
          if (ImGui::Button("Cancel Precomputation")) {
            isPrecomputing = false;
            if (precomputationThread.joinable()) {
              precomputationThread.join();
            }
            statusMessage = "Animation precomputation cancelled";
          }
        } else if (precomputationComplete) {
          // Show play controls
          ImGui::Text("Animation ready (%d frames)", static_cast<int>(precomputedMeshes.size()));
          
          if (!isPlaying) {
            if (ImGui::Button("Play Animation")) {
              isPlaying = true;
              currentFrame = 0;
              statusMessage = "Playing animation";
            }
          } else {
            if (ImGui::Button("Stop Animation")) {
              isPlaying = false;
              currentFrame = 0;
              // Reset to original mesh
              updateMeshVisualization();
              statusMessage = "Animation stopped";
            }
            ImGui::SameLine();
            ImGui::Text("Frame: %d/%d", currentFrame, static_cast<int>(precomputedMeshes.size()));
          }
        } else {
          ImGui::Text("Draw an edge in deformation mode to generate animation");
        }
        
        ImGui::Unindent();
      }
    }
  

  // =================================
  // SOLVER CONFIGURATION GROUP
  // =================================
  if (solver.hasMesh()) {
    ImGui::Text("Solver Configuration");
    ImGui::Separator();
    float dropdownWidth = 150.0f;
      
      const char* arapItems[] = { "Paper ARAP", "Ceres ARAP", "IGT ARAP" };
      
      ImGui::SetNextItemWidth(dropdownWidth);
      if (ImGui::Combo("##ARAP", &selectedArapImplementation, arapItems, IM_ARRAYSIZE(arapItems))) {
          solver.setArapImplementation(static_cast<Solver::ARAPImplementation>(selectedArapImplementation));
          statusMessage = "ARAP implementation changed to " + std::string(arapItems[selectedArapImplementation]);
      }
      
      ImGui::SameLine();
      ImGui::SetNextItemWidth(dropdownWidth);

      if (selectedArapImplementation == 1) { // Ceres ARAP
          const char* ceresItems[] = { "Sparse Normal Cholesky", "Sparse Schur", "CGNR" };
          if (ImGui::Combo("##Solver", &selectedCeresSolver, ceresItems, IM_ARRAYSIZE(ceresItems))) {
              solver.setSolverType(static_cast<Solver::SolverType>(selectedCeresSolver));
              statusMessage = "Ceres solver changed to " + std::string(ceresItems[selectedCeresSolver]);
          }
      } else { // Paper ARAP or IGT ARAP
          const char* paperItems[] = { "Cholesky", "LDLT" };
          if (ImGui::Combo("##Solver", &selectedPaperSolver, paperItems, IM_ARRAYSIZE(paperItems))) {
              solver.setPaperSolverType(static_cast<Solver::PaperSolverType>(selectedPaperSolver));
              statusMessage = "Paper solver changed to " + std::string(paperItems[selectedPaperSolver]);
          }
      }    

      ImGui::SameLine();
      ImGui::SetNextItemWidth(dropdownWidth);

      if (ImGui::SliderInt("##Iterations", &iterations, 1, 50, "Iter: %d")) {
          solver.setNumberofIterations(iterations);
          statusMessage = "Iterations set to " + std::to_string(iterations);
      }

      ImGui::NewLine();
      if (realTimeSolving || animationModeEnabled) {
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
        statusMessage = "ARAP solved with " + std::to_string(iterations) + " iterations";
      }
      if (realTimeSolving || animationModeEnabled) {
        ImGui::EndDisabled();
      }
    }
  

  // =================================
  // STATUS MESSAGE
  // =================================
  ImGui::Separator();
  ImGui::Text("Status: %s", statusMessage.c_str());

  if (ImGui::BeginPopup("Select Mesh")) {
    const std::string dataDir = "Data";
    if (fs::is_directory(dataDir)) {

      std::vector<fs::directory_entry> plyFiles;
      for (auto& e : fs::directory_iterator(dataDir)) {
        if (e.path().extension() == ".ply") plyFiles.push_back(e);
      }
      std::sort(plyFiles.begin(), plyFiles.end(), [](const auto& a, const auto& b) {
        return a.path().filename().string() < b.path().filename().string();
      });
      
      for (auto& e : plyFiles) {
        if (e.path().extension() == ".ply") {
          auto fn = e.path().filename().string();
          if (ImGui::Selectable(fn.c_str())) {
            loadMesh(e.path().string());
            statusMessage = "Loaded mesh: " + fn;
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
          
          if (realTimeSolving && !animationModeEnabled) {
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
        
        // On-demand mode or animation mode: show dragged path
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
      
      // In animation mode, start precomputation when drag ends
      if (animationModeEnabled && !dragSamples.empty() && draggedVertexIndex >= 0) {
        std::cout << "[Animation] Starting precomputation with " << dragSamples.size() << " samples" << std::endl;
        
        // Stop any existing precomputation
        if (precomputationThread.joinable()) {
          isPrecomputing = false;
          precomputationThread.join();
        }
        
        // Collect fixed constraints (selected vertices excluding the dragged one)
        std::vector<int> fixedIndices;
        std::vector<Eigen::Vector3d> fixedPositions;
        for (size_t i = 0; i < selectedVertexIndices.size(); ++i) {
          if (selectedVertexIndices[i] != draggedVertexIndex) {
            fixedIndices.push_back(selectedVertexIndices[i]);
            fixedPositions.push_back(selectedPoints[i]);
          }
        }
        
        // Start precomputation in background thread
        precomputationThread = std::thread([dragSamples = dragSamples, 
                                          draggedVertex = draggedVertexIndex,
                                          fixedIndices = std::move(fixedIndices),
                                          fixedPositions = std::move(fixedPositions)]() {
          precomputeAnimation(dragSamples, draggedVertex, fixedIndices, fixedPositions);
        });
      }
      
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

void cleanupAnimation() {
  // Stop precomputation if running
  if (precomputationThread.joinable()) {
    isPrecomputing = false;
    precomputationThread.join();
  }
  
  // Clear animation state
  animationModeEnabled = false;
  isPlaying = false;
  currentFrame = 0;
  {
    std::lock_guard<std::mutex> lock(meshDataMutex);
    precomputedMeshes.clear();
  }
  precomputationComplete = false;
  precomputationProgress = 0;
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
    
    // Handle animation playback
    if (animationModeEnabled && isPlaying) {
      playAnimation();
    }
    
    setupUI();
    vertexPickerCallback();
  };
}

void Viewer::show() {
  polyscope::show();
  // Cleanup animation resources when viewer shuts down
  cleanupAnimation();
}

void startViewer() {
  Viewer v;
  v.init();
  v.show();
}

// Precompute animation frames
void precomputeAnimation(const std::vector<Eigen::Vector3d>& pathSamples, 
                        int draggedVertex, 
                        const std::vector<int>& fixedIndices,
                        const std::vector<Eigen::Vector3d>& fixedPositions) {
    if (pathSamples.size() < 2) return;
    
    const int numFrames = 50;  // Sample 50 times along the edge
    isPrecomputing = true;
    precomputationProgress = 0;
    precomputationComplete = false;
    
    // Clear previous data
    {
        std::lock_guard<std::mutex> lock(meshDataMutex);
        precomputedMeshes.clear();
        precomputedMeshes.reserve(numFrames);
    }
    
    // Create a copy of the solver for background computation
    Solver::ARAPSolver backgroundSolver;
    backgroundSolver.setMesh(solver.getVertices(), solver.getFaces());
    
    std::cout << "[Animation] Starting precomputation of " << numFrames << " frames..." << std::endl;
    
    for (int frame = 0; frame < numFrames && isPrecomputing; ++frame) {
        // Interpolate along the path
        float t = static_cast<float>(frame) / (numFrames - 1);
        Eigen::Vector3d interpolatedPos;
        
        if (pathSamples.size() == 2) {
            // Linear interpolation between start and end
            interpolatedPos = (1.0f - t) * pathSamples[0] + t * pathSamples[1];
        } else {
            // Multi-segment interpolation
            float segmentLength = static_cast<float>(pathSamples.size() - 1);
            float scaledT = t * segmentLength;
            int segmentIndex = std::min(static_cast<int>(scaledT), static_cast<int>(pathSamples.size()) - 2);
            float localT = scaledT - segmentIndex;
            
            interpolatedPos = (1.0f - localT) * pathSamples[segmentIndex] + 
                            localT * pathSamples[segmentIndex + 1];
        }
        
        // Set up constraints for this frame
        std::vector<int> constraintIndices = {draggedVertex};
        std::vector<Eigen::Vector3d> constraintPositions = {interpolatedPos};
        
        // Add fixed constraints
        for (size_t i = 0; i < fixedIndices.size(); ++i) {
            constraintIndices.push_back(fixedIndices[i]);
            constraintPositions.push_back(fixedPositions[i]);
        }
        
        backgroundSolver.setConstraints(constraintIndices, constraintPositions);
        backgroundSolver.solveARAP();
        
        // Store the result
        {
            std::lock_guard<std::mutex> lock(meshDataMutex);
            precomputedMeshes.push_back(backgroundSolver.getVertices());
        }
        
        precomputationProgress = frame + 1;
        
        // Small delay to prevent UI freezing
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (isPrecomputing) {
        precomputationComplete = true;
        std::cout << "[Animation] Precomputation completed!" << std::endl;
    } else {
        std::cout << "[Animation] Precomputation cancelled." << std::endl;
    }
    
    isPrecomputing = false;
}

void playAnimation() {
    if (!precomputationComplete || precomputedMeshes.empty()) return;
    
    static auto lastFrameTime = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<float>(now - lastFrameTime).count();
    
    // Play at 30 FPS
    const float frameInterval = 1.0f / 30.0f;
    
    if (elapsed >= frameInterval) {
        std::lock_guard<std::mutex> lock(meshDataMutex);
        
        if (currentFrame < static_cast<int>(precomputedMeshes.size())) {
            // Update mesh with current frame
            currentMesh->updateVertexPositions(precomputedMeshes[currentFrame]);
            currentFrame++;
            lastFrameTime = now;
        } else {
            // Animation finished, loop back to start
            currentFrame = 0;
        }
    }
}

} // namespace Window