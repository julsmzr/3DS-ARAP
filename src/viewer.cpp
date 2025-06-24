#include "viewer.h"
#include <polyscope/polyscope.h>
#include "imgui.h"
#include <GLFW/glfw3.h>

namespace Window {

void Viewer::init() {
    polyscope::init();
    
    // Set smaller UI scale
    polyscope::state::userCallback = [](){ 
        ImGui::GetIO().FontGlobalScale = ViewerConfig::UI_SCALE_FACTOR;
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

