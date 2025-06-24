#include "ARAPDeformer.h"
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <unordered_map>
#include <set>

// Helper typedefs
using SparseMatrixf = Eigen::SparseMatrix<float>;
using Tripletf = Eigen::Triplet<float>;
using namespace Eigen;

ARAPDeformer::ARAPDeformer(SimpleMesh& mesh)
    : m_mesh(mesh) {
    buildAdjacency();
    computeCotangentWeights();
}

void ARAPDeformer::setHandles(const std::vector<unsigned>& handleIndices, const std::vector<Vector3f>& handlePositions) {
    m_handleIndices = handleIndices;
    m_handlePositions = handlePositions;

    m_isHandle.assign(m_mesh.getVertices().size(), false);
    for (unsigned idx : handleIndices)
        m_isHandle[idx] = true;
}

void ARAPDeformer::deform(int iterations) {
    auto& vertices = m_mesh.getVertices();
    const size_t n = vertices.size();

    // Precompute original positions
    std::vector<Vector3f> originalPositions(n);
    for (size_t i = 0; i < n; ++i)
        originalPositions[i] = vertices[i].position.head<3>();

    // Initialize deformed positions
    std::vector<Vector3f> deformedPositions = originalPositions;

    std::vector<Matrix3f> rotations(n, Matrix3f::Identity());

    for (int iter = 0; iter < iterations; ++iter) {
        // Local Step: compute rotations
        for (size_t i = 0; i < n; ++i) {
            if (m_isHandle[i]) continue;

            Matrix3f S = Matrix3f::Zero();
            for (const auto& j : m_vertexNeighbors[i]) {
                float w = m_cotangentWeights[{i, j}];

                Vector3f pi = originalPositions[i];
                Vector3f pj = originalPositions[j];
                Vector3f qi = deformedPositions[i];
                Vector3f qj = deformedPositions[j];

                S += w *  (pi - pj) * (qi - qj).transpose();
            }

            JacobiSVD<Matrix3f> svd(S, ComputeFullU | ComputeFullV);
            Matrix3f U = svd.matrixU();
            Matrix3f V = svd.matrixV();

            Matrix3f R = U * V.transpose();
            if (R.determinant() < 0) {
                Matrix3f I = Matrix3f::Identity();
                I(2, 2) = -1;
                R = U * I * V.transpose();
            }

            rotations[i] = R;
        }

        // Global Step: solve linear system
        SparseMatrixf L(n, n);
        std::vector<Tripletf> triplets;
        VectorXf bx = VectorXf::Zero(n);
        VectorXf by = VectorXf::Zero(n);
        VectorXf bz = VectorXf::Zero(n);

        for (size_t i = 0; i < n; ++i) {
            if (m_isHandle[i]) {
                triplets.emplace_back(i, i, 1.0f);
                bx(i) = m_handlePositions[getHandleIndex(i)].x();
                by(i) = m_handlePositions[getHandleIndex(i)].y();
                bz(i) = m_handlePositions[getHandleIndex(i)].z();
                continue;
            }

            float weightSum = 0.0f;
            Vector3f b = Vector3f::Zero();
            for (const auto& j : m_vertexNeighbors[i]) {
                float w = m_cotangentWeights[{i, j}];
                weightSum += w;

                triplets.emplace_back(i, j, -w);

                Vector3f pij = originalPositions[i] - originalPositions[j];
                Vector3f Rij = 0.5f * (rotations[i] + rotations[j]) * pij;
                b += w * Rij;
            }

            triplets.emplace_back(i, i, weightSum);
            bx(i) = b.x();
            by(i) = b.y();
            bz(i) = b.z();
        }

        L.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::SimplicialLDLT<SparseMatrixf> solver;
        solver.compute(L);
        if (solver.info() != Success) {
            std::cerr << "Decomposition failed" << std::endl;
            return;
        }

        VectorXf x = solver.solve(bx);
        VectorXf y = solver.solve(by);
        VectorXf z = solver.solve(bz);

        for (size_t i = 0; i < n; ++i) {
            deformedPositions[i] = Vector3f(x(i), y(i), z(i));
        }
    }

    // Update mesh positions
    for (size_t i = 0; i < n; ++i) {
        vertices[i].position.head<3>() = deformedPositions[i];
    }
}

void ARAPDeformer::buildAdjacency() {
    const auto& triangles = m_mesh.getTriangles();
    const auto& vertices = m_mesh.getVertices();
    m_vertexNeighbors.clear();
    m_vertexNeighbors.resize(vertices.size());

    for (const auto& tri : triangles) {
        m_vertexNeighbors[tri.idx0].insert(tri.idx1);
        m_vertexNeighbors[tri.idx0].insert(tri.idx2);
        m_vertexNeighbors[tri.idx1].insert(tri.idx0);
        m_vertexNeighbors[tri.idx1].insert(tri.idx2);
        m_vertexNeighbors[tri.idx2].insert(tri.idx0);
        m_vertexNeighbors[tri.idx2].insert(tri.idx1);
    }
}

void ARAPDeformer::computeCotangentWeights() {
    const auto& triangles = m_mesh.getTriangles();
    const auto& vertices = m_mesh.getVertices();
    m_cotangentWeights.clear();

    auto cotangent = [](const Vector3f& a, const Vector3f& b) -> float {
        float cosTheta = a.dot(b);
        float sinTheta = a.cross(b).norm();
        return cosTheta / sinTheta;
    };

    std::map<std::pair<unsigned, unsigned>, float> rawWeights;
    for (const auto& tri : triangles) {
        unsigned i0 = tri.idx0, i1 = tri.idx1, i2 = tri.idx2;

        Vector3f v0 = vertices[i0].position.head<3>();
        Vector3f v1 = vertices[i1].position.head<3>();
        Vector3f v2 = vertices[i2].position.head<3>();

        float cotAlpha = cotangent(v1 - v0, v2 - v0);
        float cotBeta  = cotangent(v2 - v1, v0 - v1);
        float cotGamma = cotangent(v0 - v2, v1 - v2);

        rawWeights[{i1, i2}] += 0.5*cotAlpha;
        rawWeights[{i2, i1}] += 0.5*cotAlpha;

        rawWeights[{i0, i2}] += 0.5*cotBeta;
        rawWeights[{i2, i0}] += 0.5*cotBeta;

        rawWeights[{i0, i1}] += 0.5*cotGamma;
        rawWeights[{i1, i0}] += 0.5*cotGamma;
    }

    for (const auto& [edge, w] : rawWeights) {
    m_cotangentWeights[edge] = w;
}
}

int ARAPDeformer::getHandleIndex(unsigned vertexIndex) const {
    for (size_t i = 0; i < m_handleIndices.size(); ++i)
        if (m_handleIndices[i] == vertexIndex)
            return static_cast<int>(i);
    return -1;
}
