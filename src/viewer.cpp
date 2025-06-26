#include "viewer.h"
#include "load_mesh.h"
#include <polyscope/polyscope.h>
#include <imgui.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace Window {

void loadMeshFromFile(const std::string& filepath) {
    std::cout << "Loading mesh from: " << filepath << std::endl;
    
    // Clear any existing meshes
    polyscope::removeAllStructures();
    
    MeshLoader::Mesh mesh = MeshLoader::loadPLY(filepath);
    
    if (mesh.isValid()) {
        // Extract filename for display name
        std::string meshName = fs::path(filepath).stem().string();
        MeshLoader::displayMesh(mesh, meshName);
        std::cout << "Successfully loaded and displayed mesh: " << meshName << std::endl;
    } else {
        std::cerr << "Failed to load mesh from: " << filepath << std::endl;
    }
}

void setupUI() {
    if (ImGui::Button("Select Mesh...")) {
        ImGui::OpenPopup("Select Mesh");
    }
    
    if (ImGui::BeginPopup("Select Mesh")) {
        std::string dataDir = "Data";
        if (fs::exists(dataDir) && fs::is_directory(dataDir)) {
            for (const auto& entry : fs::directory_iterator(dataDir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".ply") {
                    std::string filename = entry.path().filename().string();
                    if (ImGui::Selectable(filename.c_str())) {
                        loadMeshFromFile(entry.path().string());
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

void Viewer::init() {
    polyscope::init();
    polyscope::options::usePrefsFile = false;
    
    // Configure Polyscope options to match frontend.cpp
    polyscope::options::programName = "Interactive ARAP Viewer";
    polyscope::options::verbosity = 1;
    
    polyscope::state::userCallback = []() {         
        setupUI();
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
