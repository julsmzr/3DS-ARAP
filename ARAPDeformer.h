#pragma once

#include <vector>
#include <set>
#include <unordered_map>
#include <utility>
#include <functional>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "SimpleMesh.h"

struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const noexcept {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1); 
    }
};

class ARAPDeformer {
public:
    ARAPDeformer(SimpleMesh& mesh);

    void setHandles(const std::vector<unsigned>& handleIndices,
                    const std::vector<Eigen::Vector3f>& handlePositions);

    void deform(int iterations);

private:
    void buildAdjacency();
    void computeCotangentWeights();
    int getHandleIndex(unsigned vertexIndex) const;

    SimpleMesh& m_mesh;
    int n;

    std::vector<unsigned> m_handleIndices;
    std::vector<Eigen::Vector3f> m_handlePositions;
    std::vector<bool> m_isHandle;

    std::vector<std::set<unsigned>> m_vertexNeighbors;

    std::unordered_map<std::pair<unsigned, unsigned>, float, PairHash> m_cotangentWeights;

    void precomputeLaplacian();
    void computeRotations();
    void solvePositions();
};

