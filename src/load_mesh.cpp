#include "load_mesh.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <happly.h>

namespace MeshLoader {

Mesh loadPLY(const std::string& filepath) {
    Mesh mesh;
    
    try {
        // Use happly to load both ASCII and binary PLY files
        happly::PLYData plyIn(filepath);
        
        // Get vertex positions
        std::vector<double> vPos_x = plyIn.getElement("vertex").getProperty<double>("x");
        std::vector<double> vPos_y = plyIn.getElement("vertex").getProperty<double>("y");
        std::vector<double> vPos_z = plyIn.getElement("vertex").getProperty<double>("z");
        
        // Store vertices with scaling
        mesh.vertices.reserve(vPos_x.size());
        auto SCALE = 1000.0;
        for (size_t i = 0; i < vPos_x.size(); i++) {
            mesh.vertices.emplace_back(SCALE * vPos_x[i], SCALE * vPos_y[i], SCALE * vPos_z[i]);
        }
        
        // Get face indices
        try {
            auto faceIndices = plyIn.getElement("face").getListProperty<int>("vertex_indices");
            mesh.faces = faceIndices;
        } catch (...) {
            try {
                auto faceIndices = plyIn.getElement("face").getListProperty<size_t>("vertex_indices");
                mesh.faces.resize(faceIndices.size());
                for (size_t i = 0; i < faceIndices.size(); i++) {
                    mesh.faces[i].resize(faceIndices[i].size());
                    for (size_t j = 0; j < faceIndices[i].size(); j++) {
                        mesh.faces[i][j] = static_cast<int>(faceIndices[i][j]);
                    }
                }
            } catch (...) {
                std::cerr << "Warning: Could not load face indices" << std::endl;
            }
        }
        
        std::cout << "Loaded mesh with " << mesh.vertices.size() << " vertices and " 
                  << mesh.faces.size() << " faces" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading PLY file: " << e.what() << std::endl;
    }
    
    return mesh;
}

void displayMesh(const Mesh& mesh, const std::string& name) {
    if (!mesh.isValid()) {
        std::cerr << "Error: Cannot display invalid mesh" << std::endl;
        return;
    }
    
    // Convert vertices to matrix format for polyscope
    Eigen::MatrixXd vertices(mesh.vertices.size(), 3);
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        vertices.row(i) = mesh.vertices[i];
    }
    
    // Convert faces to matrix format (assuming triangular faces)
    // For non-triangular faces, we'll triangulate them
    std::vector<std::array<int, 3>> triangles;
    
    for (const auto& face : mesh.faces) {
        if (face.size() == 3) {
            // Triangle face
            triangles.push_back({face[0], face[1], face[2]});
        } else if (face.size() == 4) {
            // Quad face - split into two triangles
            triangles.push_back({face[0], face[1], face[2]});
            triangles.push_back({face[0], face[2], face[3]});
        } else if (face.size() > 4) {
            // Polygon - fan triangulation
            for (size_t i = 1; i < face.size() - 1; ++i) {
                triangles.push_back({face[0], face[i], face[i+1]});
            }
        }
    }
    
    Eigen::MatrixXi faces(triangles.size(), 3);
    for (size_t i = 0; i < triangles.size(); ++i) {
        faces.row(i) << triangles[i][0], triangles[i][1], triangles[i][2];
    }
    
    // Register and display the mesh
    auto psMesh = polyscope::registerSurfaceMesh(name, vertices, faces);
    psMesh->setEnabled(true);
    
    // Automatically fit the camera to show the mesh
    polyscope::view::resetCameraToHomeView();
    
    std::cout << "Displayed mesh '" << name << "' with " << triangles.size() << " triangles" << std::endl;
}

} // namespace MeshLoader