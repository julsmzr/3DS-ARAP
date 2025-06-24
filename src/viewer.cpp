#include "viewer.h"
#include <polyscope/polyscope.h>
#include <imgui.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace Window {

void setupUI() {
    if (ImGui::Button("Load Mesh")) {
        ImGui::OpenPopup("Select Mesh");
    }
    
    if (ImGui::BeginPopup("Select Mesh")) {
        std::string dataDir = "Data";
        if (fs::exists(dataDir) && fs::is_directory(dataDir)) {
            for (const auto& entry : fs::directory_iterator(dataDir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".ply") {
                    std::string filename = entry.path().filename().string();
                    if (ImGui::Selectable(filename.c_str())) {
                        // TODO: Implement proper mesh loading
                        std::cout << "Selected mesh: " << filename << std::endl;
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
    
    // Set smaller UI scale
    polyscope::state::userCallback = [](){ 
        ImGui::GetIO().FontGlobalScale = ViewerConfig::UI_SCALE_FACTOR;
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

