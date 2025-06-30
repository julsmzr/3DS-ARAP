#include <iostream>
#include "viewer.h"

int main() {
#ifndef NDEBUG
    std::cout << "[DEBUG] Running in debug mode\n";
#endif
    std::cout << "Interactive ARAP\n\n";
    Window::startViewer();
    return 0;
}
