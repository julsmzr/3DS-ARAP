#pragma once
#include <string> 

namespace Window {
    
    class Viewer {
        public:
        void init();
        void show();
    };
    
    void startViewer();
    void loadMeshFromFile(const std::string& filepath);
    void setupUI();
    
} // namespace Window