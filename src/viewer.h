#pragma once

namespace ViewerConfig {
    constexpr float UI_SCALE_FACTOR = 0.6f;
}

namespace Window {
    
    class Viewer {
        public:
        void init();
        void show();
    };
    
    void startViewer();
    
} // namespace Window