#pragma once

#include <vector>
#include <set>
#include <unordered_map>
#include <utility>
#include <functional>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "load_mesh.h"

struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const noexcept {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1); 
    }
};

class Solver {
public:
    Solver(MeshLoader::Mesh mesh);

    Solver() {}

    void init(MeshLoader::Mesh mesh);

    void setAnchors(const std::vector<unsigned>& anchorIndices, const std::vector<Eigen::Vector3d>& anchorPositions);

    void clearAnchors(){
        m_anchorIndices.clear();
        m_anchorPositions.clear();
    }

    void setDragIndex(unsigned dragPosition);

    void deform(int iterations, const Eigen::Vector3d& dragEndPosition);

    static Eigen::Vector3d screenToWorld( const glm::vec2& screenCoords, const Eigen::Vector3d& planePoint, const glm::vec3& planeNormal) {

            // Get viewport size
            size_t w = polyscope::view::windowWidth;
            size_t h = polyscope::view::windowHeight;

            // 1) Build NDC coords
            float x_ndc = (screenCoords.x / float(w)) * 2.0f - 1.0f;
            float y_ndc = 1.0f - (screenCoords.y / float(h)) * 2.0f;
            glm::vec4 clip{x_ndc, y_ndc, -1.0f, 1.0f};

            // 2) Invert the projection matrix
            glm::mat4 invProj = glm::inverse(
                polyscope::view::getCameraPerspectiveMatrix()
            );
            glm::vec4 camSpace = invProj * clip;
            camSpace /= camSpace.w;

            // 3) Invert the view matrix
            glm::mat4 invView = glm::inverse(
                polyscope::view::getCameraViewMatrix()
            );
            glm::vec4 worldDir4 = invView * glm::vec4(camSpace.x, camSpace.y, camSpace.z, 0.0f);
            glm::vec3 rayDir = glm::normalize(glm::vec3(worldDir4));

            // 4) Ray‐plane intersection
            glm::vec3 origin = polyscope::view::getCameraWorldPosition();  
            glm::vec3 planeP{ float(planePoint.x()), float(planePoint.y()), float(planePoint.z()) };
            float denom = glm::dot(planeNormal, rayDir);
            if (std::fabs(denom) < 1e-6f) {
                return planePoint;
            }
            float t = glm::dot(planeP - origin, planeNormal) / denom;
            glm::vec3 W = origin + rayDir * t;
            return Eigen::Vector3d{ W.x, W.y, W.z };
    }

private:
    void buildAdjacency();
    void computeCotangentWeights();

    MeshLoader::Mesh m_mesh;
    size_t n;

    //Anchors, i.e. vertices that are fixed
    std::vector<unsigned> m_anchorIndices;
    std::vector<Eigen::Vector3d> m_anchorPositions;

    //index of vertex that is dragged
    unsigned m_dragIndex;

    //if index is a constraint or not
    std::vector<bool> m_isHandle;

    std::vector<std::set<unsigned>> m_vertexNeighbors;

    std::unordered_map<std::pair<unsigned, unsigned>, float, PairHash> m_cotangentWeights;
};