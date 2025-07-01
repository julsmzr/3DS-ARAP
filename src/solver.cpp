#include "solver.h"
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <unordered_map>
#include <set>
#include <chrono>

// Helper typedefs
using SparseMatrixd = Eigen::SparseMatrix<double>;
using Tripletd = Eigen::Triplet<double>;
using namespace Eigen;
using namespace MeshLoader;

Solver::Solver(Mesh mesh){
    init(mesh);
}

void Solver::init(Mesh mesh){
    m_mesh = mesh;
    n = mesh.vertices.size();

    std::cout << "Loaded " << n << " vertices and " << m_mesh.faces.size() << " faces\n";
    buildAdjacency();
    computeCotangentWeights();
}

void Solver::setAnchors(const std::vector<unsigned>& anchorIndices, const std::vector<Vector3d>& anchorPositions) {
    // m_anchorPositions = anchorPositions; //Commented this out and tried to use m_mesh instead because for some reason m_mesh.vertices[idx] =/= anchorPositions for the same idx
    m_anchorIndices = anchorIndices;

    m_anchorPositions.clear();
    m_anchorPositions.reserve(m_anchorIndices.size());

    for (unsigned idx : m_anchorIndices) {
        m_anchorPositions.push_back(m_mesh.vertices[idx]);
    }
}

void Solver::setDragIndex(unsigned dragIndex) {
    m_dragIndex = dragIndex;
}

void Solver::deform(int iterations, const Eigen::Vector3d& dragEndPosition) {

    std::vector<Eigen::Vector3d> m_handlePositions = m_anchorPositions;
    m_handlePositions.push_back(dragEndPosition);

    std::vector<unsigned>  m_handleIndices = m_anchorIndices;
    m_handleIndices.push_back(m_dragIndex);

    m_isHandle.assign(n, false);

    std::unordered_set<unsigned> handleSet(m_handleIndices.begin(), m_handleIndices.end());

    for (unsigned i = 0; i < n; ++i) {
        m_isHandle[i] = (handleSet.find(i) != handleSet.end());
    }
    size_t trueCount = std::count(m_isHandle.begin(), m_isHandle.end(), true);
    std::cout << "Number of true entries in m_isHandle: " << trueCount << std::endl;

    // Print anchors
    std::cout << "Anchor Indices, Positions, and Mesh Vertices:\n";
    for (size_t i = 0; i < m_anchorIndices.size(); ++i) {
        unsigned index = m_anchorIndices[i];
        const Eigen::Vector3d& anchorPos = m_anchorPositions[i];
        const Eigen::Vector3d& meshVertex = m_mesh.vertices[index];

        std::cout << "  Index: " << index
                << " -> Anchor Position: " << anchorPos.transpose()
                << " | Mesh Vertex: " << meshVertex.transpose() << '\n';
    }

    // Print drag
    std::cout << "Drag Index and Position:\n";
    std::cout << "  Index: " << m_dragIndex
            << " -> Position: " << dragEndPosition.transpose() << '\n';


    auto getHandleIndex = [m_handleIndices](unsigned vertexIndex) -> int {
        auto it = std::find(m_handleIndices.begin(), m_handleIndices.end(), vertexIndex);
        return (it != m_handleIndices.end()) ? static_cast<int>(it - m_handleIndices.begin()) : -1;
    };

    // Precompute original positions
    std::vector<Vector3d> originalPositions = m_mesh.vertices;

    // Initialize deformed positions
    std::vector<Vector3d> deformedPositions = originalPositions;

    std::vector<Matrix3d> rotations(n, Matrix3d::Identity());

    for (int iter = 0; iter < iterations; ++iter) {
        std::cout << "Iteration " << iter + 1 << "/" << iterations << " started.\n";
        // Local Step: compute rotations
        auto t1 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) {
            if (m_isHandle[i]) continue;

            Matrix3d S = Matrix3d::Zero();
            for (const auto& j : m_vertexNeighbors[i]) {
                float w = m_cotangentWeights[{i, j}];

                Vector3d pi = originalPositions[i];
                Vector3d pj = originalPositions[j];
                Vector3d qi = deformedPositions[i];
                Vector3d qj = deformedPositions[j];

                S += w *  (pi - pj) * (qi - qj).transpose();
            }

            JacobiSVD<Matrix3d> svd(S, ComputeFullU | ComputeFullV);
            Matrix3d U = svd.matrixU();
            Matrix3d V = svd.matrixV();

            Matrix3d R = U * V.transpose();
            if (R.determinant() < 0) {
                Matrix3d I = Matrix3d::Identity();
                I(2, 2) = -1;
                R = U * I * V.transpose();
            }

            rotations[i] = R;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> localStepDuration = t2 - t1;
        std::cout << "  Rotation step took " << localStepDuration.count() << " seconds.\n";

        // Global Step: solve linear system
        auto t3 = std::chrono::high_resolution_clock::now();
        SparseMatrixd L(n, n);
        std::vector<Tripletd> triplets;
        VectorXd bx = VectorXd::Zero(n);
        VectorXd by = VectorXd::Zero(n);
        VectorXd bz = VectorXd::Zero(n);

        for (size_t i = 0; i < n; ++i) {
            if (m_isHandle[i]) {
                triplets.emplace_back(i, i, 1.0f);
                bx(i) = m_handlePositions[getHandleIndex(i)].x();
                by(i) = m_handlePositions[getHandleIndex(i)].y();
                bz(i) = m_handlePositions[getHandleIndex(i)].z();
                continue;
            }

            float weightSum = 0.0f;
            Vector3d b = Vector3d::Zero();
            for (const auto& j : m_vertexNeighbors[i]) {
                float w = m_cotangentWeights[{i, j}];
                weightSum += w;

                triplets.emplace_back(i, j, -w);

                Vector3d pij = originalPositions[i] - originalPositions[j];
                Vector3d Rij = 0.5f * (rotations[i] + rotations[j]) * pij;
                b += w * Rij;
            }

            triplets.emplace_back(i, i, weightSum);
            bx(i) = b.x();
            by(i) = b.y();
            bz(i) = b.z();
        }

        L.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::SimplicialLDLT<SparseMatrixd> solver;
        solver.compute(L);
        if (solver.info() != Success) {
            std::cerr << "Decomposition failed" << std::endl;
            return;
        }

        VectorXd x = solver.solve(bx);
        VectorXd y = solver.solve(by);
        VectorXd z = solver.solve(bz);

         auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> globalStepDuration = t4 - t3;
        std::cout << "  Position step (solve) took " << globalStepDuration.count() << " seconds.\n";

        for (size_t i = 0; i < n; ++i) {
            deformedPositions[i] = Vector3d(x(i), y(i), z(i));
        }
    }

    // Update mesh positions
    for (size_t i = 0; i < n; ++i) {
        m_mesh.vertices[i] = deformedPositions[i];
    }
}

void Solver::buildAdjacency() {
    m_vertexNeighbors.clear();
    m_vertexNeighbors.resize(n);

    for (const auto& face : m_mesh.faces) {
        m_vertexNeighbors[face[0]].insert(face[1]);
        m_vertexNeighbors[face[0]].insert(face[2]);
        m_vertexNeighbors[face[1]].insert(face[0]);
        m_vertexNeighbors[face[1]].insert(face[2]);
        m_vertexNeighbors[face[2]].insert(face[0]);
        m_vertexNeighbors[face[2]].insert(face[1]);
    }
}

void Solver::computeCotangentWeights() {
    m_cotangentWeights.clear();

    auto cotangent = [](const Vector3d& a, const Vector3d& b) -> float {
        float cosTheta = a.dot(b);
        float sinTheta = a.cross(b).norm();
        return cosTheta / sinTheta;
    };

    std::map<std::pair<unsigned, unsigned>, float> rawWeights;
    for (const auto& face : m_mesh.faces) {
        unsigned i0 = face[0], i1 = face[1], i2 = face[2];

        Vector3d v0 = m_mesh.vertices[i0];
        Vector3d v1 = m_mesh.vertices[i1];
        Vector3d v2 = m_mesh.vertices[i2];

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
