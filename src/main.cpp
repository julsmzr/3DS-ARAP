#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <iostream>
#include "viewer.h"
#include "load_mesh.h"

// Equation 3
float energy_at_vertex(
    int i,
    const std::vector<Eigen::Vector3f>& p,
    const std::vector<Eigen::Vector3f>& p_prime,
    const Eigen::Matrix3f& R_i,
    const std::vector<int>& neighbors,
    const std::unordered_map<int, float>& weights
) {
    float energy = 0.0f;

    for (int j : neighbors) {
        Eigen::Vector3f e_ij     = p[i] - p[j];
        Eigen::Vector3f e_ij_def = p_prime[i] - p_prime[j];
        Eigen::Vector3f diff     = e_ij_def - R_i * e_ij;

        float w_ij = weights.at(j);
        energy += w_ij * diff.squaredNorm();
    }

    return energy;
}

// Equation 6
Eigen::Matrix3f compute_optimal_rotation(
    int i,
    const std::vector<Eigen::Vector3f>& p,
    const std::vector<Eigen::Vector3f>& p_prime,
    const std::vector<int>& neighbors,
    const std::unordered_map<int, float>& weights
) {
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();

    for (int j : neighbors) {
        float w_ij = weights.at(j);

        Eigen::Vector3f e_ij     = p[i] - p[j];           // original edge
        Eigen::Vector3f e_ij_def = p_prime[i] - p_prime[j]; // deformed edge

        S += w_ij * (e_ij * e_ij_def.transpose());
    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    Eigen::Matrix3f R = V * U.transpose();

    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    return R;
}

// Equation 9
void build_laplacian_and_rhs(
    const std::vector<Eigen::Vector3f>& p,
    const std::vector<Eigen::Matrix3f>& R,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<std::unordered_map<int, float>>& weights,
    Eigen::SparseMatrix<float>& L,
    Eigen::MatrixXf& b
) {
    const int n = p.size();
    std::vector<Eigen::Triplet<float>> triplets;
    b = Eigen::MatrixXf::Zero(n, 3);

    for (int i = 0; i < n; ++i) {
        float weight_sum = 0.0f;

        for (int j : neighbors[i]) {
            float w_ij = weights[i].at(j);
            weight_sum += w_ij;

            triplets.emplace_back(i, j, -w_ij);

            Eigen::Vector3f p_diff = p[i] - p[j];
            Eigen::Matrix3f R_avg = 0.5f * (R[i] + R[j]);
            b.row(i) += w_ij * (R_avg * p_diff).transpose();
        }

        triplets.emplace_back(i, i, weight_sum);
    }

    L.resize(n, n);
    L.setFromTriplets(triplets.begin(), triplets.end());
}

void arap_solve(
    const std::vector<Eigen::Vector3f>& p,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<std::unordered_map<int, float>>& weights,
    const std::unordered_map<int, Eigen::Vector3f>& constraints,
    std::vector<Eigen::Vector3f>& p_prime,
    int iterations = 5
) {
    const int n = p.size();
    p_prime = p;

    std::vector<Eigen::Matrix3f> R(n);

    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < n; ++i) {
            R[i] = compute_optimal_rotation(i, p, p_prime, neighbors[i], weights[i]);
        }

        Eigen::SparseMatrix<float> L;
        Eigen::MatrixXf b;
        build_laplacian_and_rhs(p, R, neighbors, weights, L, b);

        for (const auto& [idx, pos] : constraints) {
            L.coeffRef(idx, idx) += 1e10f;
            b.row(idx) = 1e10f * pos.transpose();
        }

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
        solver.compute(L);

        if (solver.info() != Eigen::Success) {
            std::cerr << "L factorization failed.\n";
            return;
        }

        Eigen::MatrixXf p_prime_mat = solver.solve(b);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solve failed.\n";
            return;
        }

        for (int i = 0; i < n; ++i) {
            p_prime[i] = p_prime_mat.row(i).transpose();
        }
    }
}

void compute_cotangent_weights(
    const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<std::vector<int>>& faces,
    std::vector<std::vector<int>>& neighbors,
    std::vector<std::unordered_map<int, float>>& weights
) {
    const int n = vertices.size();
    neighbors.resize(n);
    weights.resize(n);

    for (const auto& f : faces) {
        if (f.size() < 3) continue;

        for (int i = 0; i < f.size(); ++i) {
            int i0 = f[i];
            int i1 = f[(i + 1) % f.size()];
            int i2 = f[(i + 2) % f.size()];

            const auto& v0 = vertices[i0];
            const auto& v1 = vertices[i1];
            const auto& v2 = vertices[i2];

            Eigen::Vector3f e0 = v1 - v0;
            Eigen::Vector3f e1 = v2 - v0;

            float cotangent = e0.dot(e1) / (e0.cross(e1)).norm();

            weights[i1][i2] += cotangent;
            weights[i2][i1] += cotangent;

            neighbors[i1].push_back(i2);
            neighbors[i2].push_back(i1);
        }
    }

    // Remove duplicate neighbor entries
    for (auto& nbrs : neighbors) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
}

void extractARAPDataFromMesh(
    const MeshLoader::Mesh& mesh,
    std::vector<Eigen::Vector3f>& p
) {
    p.clear();
    for (const auto& v : mesh.vertices) {
        p.emplace_back(v.x(), v.y(), v.z());
    }

    std::vector<std::vector<int>> faces;
    for (const auto& f : mesh.faces) {
        std::vector<int> faceVec(f.begin(), f.end());
        faces.push_back(faceVec);
    }
}

void interactive_arap() {
    std::cout << "Interactive ARAP\n" << std::endl;
    
    if (!Window::currentMesh) {
        std::cerr << "no mesh loaded.\n";
        return;
    }

    const auto& mesh = Window::currentMesh->data;

    std::vector<Eigen::Vector3f> p;
    for (const auto& v : mesh.vertices) {
        p.emplace_back(v.x(), v.y(), v.z());
    }

    std::vector<std::vector<int>> faces;
    for (const auto& f : mesh.faces) {
        std::vector<int> faceVec(f.begin(), f.end());
        faces.push_back(faceVec);
    }

    std::vector<std::vector<int>> neighbors;
    std::vector<std::unordered_map<int, float>> weights;

    extractARAPDataFromMesh(mesh, p);
    std::cout << "Extracted ARAP data from mesh" << std::endl;
    compute_cotangent_weights(p, faces, neighbors, weights);
    std::cout << "Computed cotn weights" << std::endl;

    std::unordered_map<int, Eigen::Vector3f> constraints;
    constraints[0] = p[0];
    if (p.size() > 1) {
        constraints[1] = p[1] + Eigen::Vector3f(20.0f, 15.0f, 34.0f);
        int movable = 100;
        constraints[movable] = p[movable] + Eigen::Vector3f(20.0f, 80.0f, 77.0f);
    }

    std::vector<Eigen::Vector3f> p_prime;

    std::cout << "Solving arap" << std::endl;
    arap_solve(p, neighbors, weights, constraints, p_prime, 3);
    std::cout << "Solved arap" << std::endl;

    std::cout << "Deformed positions:\n";
    for (const auto& v : p_prime) {
        // std::cout << v.transpose() << "\n";
    }

    std::vector<glm::vec3> updatedVerts;
    for (const auto& v : p_prime) {
        updatedVerts.emplace_back(v.x(), v.y(), v.z());
    }
    Window::currentMesh->view->updateVertexPositions(updatedVerts);
    std::cout << "Updated (deformed) mesh.\n";
}

int main() {
#ifndef NDEBUG
    std::cout << "[DEBUG] Running in debug mode\n";
#endif
    std::cout << "Interactive ARAP\n\n";
    Window::startViewer();
    return 0;
}
