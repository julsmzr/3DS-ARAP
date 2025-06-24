#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <iostream>

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

int main() {
    std::cout << "Interactive ARAP\n" << std::endl;
    return 0;
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