#include "load_mesh.h"
#include <happly.h>
#include <polyscope/polyscope.h>
#include <iostream>

namespace MeshLoader {

Mesh loadPLY(const std::string& filepath) {
  Mesh mesh;
  try {
    happly::PLYData plyIn(filepath);

    auto xs = plyIn.getElement("vertex").getProperty<double>("x");
    auto ys = plyIn.getElement("vertex").getProperty<double>("y");
    auto zs = plyIn.getElement("vertex").getProperty<double>("z");
    mesh.vertices.reserve(xs.size());
    const double SCALE = 1000.0;
    for (size_t i = 0; i < xs.size(); i++) {
      mesh.vertices.emplace_back(SCALE*xs[i], SCALE*ys[i], SCALE*zs[i]);
    }

    try {
      mesh.faces = plyIn.getElement("face").getListProperty<int>("vertex_indices");
    } catch (...) {
      // alternate size_t list
      auto fl = plyIn.getElement("face").getListProperty<size_t>("vertex_indices");
      mesh.faces.resize(fl.size());
      for (size_t i = 0; i < fl.size(); i++) {
        mesh.faces[i].assign(fl[i].begin(), fl[i].end());
      }
    }

    std::cout << "Loaded " << mesh.vertices.size() 
              << " verts, " << mesh.faces.size() << " faces\n";
  } catch (std::exception& e) {
    std::cerr << "PLY load error: " << e.what() << "\n";
  }
  return mesh;
}

void writePLY(const std::string& filepath, const Mesh& mesh) {
  try {
    if (!mesh.isValid()) {
      throw std::runtime_error("Mesh is not valid: missing vertices or faces");
    }

    happly::PLYData plyOut;

    // Scale down vertices (inverse of loading)
    const double SCALE = 1000.0;
    std::vector<double> xs, ys, zs;
    xs.reserve(mesh.vertices.size());
    ys.reserve(mesh.vertices.size());
    zs.reserve(mesh.vertices.size());

    for (const auto& v : mesh.vertices) {
      xs.push_back(v.x() / SCALE);
      ys.push_back(v.y() / SCALE);
      zs.push_back(v.z() / SCALE);
    }

    plyOut.addElement("vertex", mesh.vertices.size());
    plyOut.getElement("vertex").addProperty<double>("x", xs);
    plyOut.getElement("vertex").addProperty<double>("y", ys);
    plyOut.getElement("vertex").addProperty<double>("z", zs);

    // Faces
    plyOut.addElement("face", mesh.faces.size());
    plyOut.getElement("face").addListProperty<int>("vertex_indices", mesh.faces);

    // Write to file
    std::ofstream outFile(filepath, std::ios::binary);
    plyOut.write(outFile, happly::DataFormat::Binary);
    std::cout << "Saved mesh to " << filepath << "\n";

  } catch (const std::exception& e) {
    std::cerr << "PLY write error: " << e.what() << "\n";
  }
}

polyscope::SurfaceMesh* displayMesh(const Mesh& mesh, const std::string& name) {
  if (!mesh.isValid()) {
    std::cerr << "Invalid mesh\n";
    return nullptr;
  }

  Eigen::MatrixXd V(mesh.vertices.size(), 3);
  for (size_t i=0; i<mesh.vertices.size(); i++) 
    V.row(i) = mesh.vertices[i];

  std::vector<std::array<int,3>> T;
  for (auto& f : mesh.faces) {
    if (f.size()==3) {
      T.push_back({f[0],f[1],f[2]});
    } else if (f.size()==4) {
      T.push_back({f[0],f[1],f[2]});
      T.push_back({f[0],f[2],f[3]});
    } else {
      for (size_t i=1; i+1<f.size(); i++) {
        T.push_back({f[0], f[i], f[i+1]});
      }
    }
  }

  Eigen::MatrixXi F(T.size(), 3);
  for (size_t i=0; i<T.size(); i++) 
    F.row(i) = Eigen::Vector3i{T[i][0], T[i][1], T[i][2]};

  auto ps = polyscope::registerSurfaceMesh(name, V, F);
  ps->setEnabled(true);
  ps->setEdgeWidth(1.0);
  ps->setEdgeColor({0,0,0});
  ps->setSelectionMode(polyscope::MeshSelectionMode::VerticesOnly);
  polyscope::view::resetCameraToHomeView();

  return ps;
}

} // namespace MeshLoader
