#include "solver.h"
#include <polyscope/view.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <igl/arap.h>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>

Eigen::Vector3d Solver::screenToWorld(
    const glm::vec2& screenCoords,
    const Eigen::Vector3d& planePoint,
    const glm::vec3& planeNormal) {

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

// ARAPSolver implementation
namespace Solver {

void ARAPSolver::setMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces) {
    vertices_ = vertices;
    faces_ = faces;
    constraintIndices_.clear();
    constraintPositions_.clear();
    weightsComputed_ = false;
    
    std::cout << "[Solver] Mesh loaded: " << vertices_.rows() << " vertices, " 
              << faces_.rows() << " faces" << std::endl;
}

void ARAPSolver::setConstraints(const std::vector<int>& constraintIndices, 
                               const std::vector<Eigen::Vector3d>& constraintPositions) {
    if (constraintIndices.size() != constraintPositions.size()) {
        std::cerr << "[Solver] Error: constraint indices and positions size mismatch" << std::endl;
        return;
    }
    
    constraintIndices_ = constraintIndices;
    constraintPositions_ = constraintPositions;
    
    std::cout << "[Solver] Constraints set: " << constraintIndices_.size() << " vertices" << std::endl;
}

void ARAPSolver::updateVertex(int vertexIndex, const Eigen::Vector3d& newPosition) {
    if (vertexIndex < 0 || vertexIndex >= vertices_.rows()) {
        std::cerr << "[Solver] Error: vertex index " << vertexIndex << " out of range" << std::endl;
        return;
    }
    
    vertices_.row(vertexIndex) = newPosition.transpose();
}

void ARAPSolver::computeNeighbours() {
    const int n = vertices_.rows();
    neighbors_.clear();
    neighbors_.resize(n);

    // Process each face to build neighbors
    for (int f = 0; f < faces_.rows(); ++f) {
        int v0 = faces_(f, 0);
        int v1 = faces_(f, 1);
        int v2 = faces_(f, 2);

        // Add neighbors   
        neighbors_[v0].insert(v1);
        neighbors_[v0].insert(v2);
        neighbors_[v1].insert(v0);
        neighbors_[v1].insert(v2);
        neighbors_[v2].insert(v0);
        neighbors_[v2].insert(v1);

    }  
}

void ARAPSolver::computeCotangentWeights() {
    const int n = vertices_.rows();
    neighbors_.clear();
    weights_.clear();
    neighbors_.resize(n);
    weights_.resize(n);

    // Process each face to build neighbors and compute cotangent weights
    for (int f = 0; f < faces_.rows(); ++f) {
        int v0 = faces_(f, 0);
        int v1 = faces_(f, 1);
        int v2 = faces_(f, 2);

        Eigen::Vector3f p0 = vertices_.row(v0).cast<float>();
        Eigen::Vector3f p1 = vertices_.row(v1).cast<float>();
        Eigen::Vector3f p2 = vertices_.row(v2).cast<float>();

        // Compute cotangent weights for each edge
        // For edge (v1,v2), use angle at v0
        Eigen::Vector3f e01 = p1 - p0;
        Eigen::Vector3f e02 = p2 - p0;
        float cot0 = e01.dot(e02) / e01.cross(e02).norm();
        cot0 = std::max(cot0, 1e-6f);  // Avoid numerical issues

        // For edge (v0,v2), use angle at v1  
        Eigen::Vector3f e10 = p0 - p1;
        Eigen::Vector3f e12 = p2 - p1;
        float cot1 = e10.dot(e12) / e10.cross(e12).norm();
        cot1 = std::max(cot1, 1e-6f);

        // For edge (v0,v1), use angle at v2
        Eigen::Vector3f e20 = p0 - p2;
        Eigen::Vector3f e21 = p1 - p2;
        float cot2 = e20.dot(e21) / e20.cross(e21).norm();
        cot2 = std::max(cot2, 1e-6f);

        // Add cotangent weights (each edge gets contribution from opposite angle)
        weights_[v1][v2] += cot0 / 2;
        weights_[v2][v1] += cot0 / 2;
        weights_[v0][v2] += cot1 / 2;
        weights_[v2][v0] += cot1 / 2;
        weights_[v0][v1] += cot2 / 2;
        weights_[v1][v0] += cot2 / 2;

        // Add neighbors   
        neighbors_[v0].insert(v1);
        neighbors_[v0].insert(v2);
        neighbors_[v1].insert(v0);
        neighbors_[v1].insert(v2);
        neighbors_[v2].insert(v0);
        neighbors_[v2].insert(v1);

    }  
    weightsComputed_ = true;
}

Eigen::Matrix3f ARAPSolver::computeOptimalRotation(int i, const std::vector<Eigen::Vector3f>& p, 
                                                  const std::vector<Eigen::Vector3f>& p_prime) {
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();

    if (neighbors_[i].empty()) {
        // Return identity for isolated vertices
        return Eigen::Matrix3f::Identity();
    }

    // Compute covariance matrix S
    for (int j : neighbors_[i]) {
        if (weights_[i].count(j) == 0) {
            // Skip if weight not found (shouldn't happen if weights are properly computed)
            continue;
        }
        
        float w_ij = weights_[i].at(j);
        // Skip negligible weights to avoid numerical issues
        if (std::abs(w_ij) < 1e-10f) continue;
        
        Eigen::Vector3f e_ij = p[i] - p[j];           // original edge
        Eigen::Vector3f e_ij_def = p_prime[i] - p_prime[j]; // deformed edge
        
        S += w_ij * (e_ij * e_ij_def.transpose());
    }
    
    // Add small regularization for numerical stability
    S += Eigen::Matrix3f::Identity() * 1e-6f;

    // Compute optimal rotation using SVD
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    
    // Handle SVD failure (extremely rare)
    if (svd.singularValues().minCoeff() < 1e-15f) {
        std::cerr << "[Solver] Warning: SVD has very small singular value" << std::endl;
    }

    // Compute rotation matrix
    Eigen::Matrix3f R = V * U.transpose();

    // Ensure proper rotation (det = 1)
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    return R;
}

void ARAPSolver::buildLaplacianAndRHS(const std::vector<Eigen::Vector3f>& p, 
                                     const std::vector<Eigen::Matrix3f>& R,
                                     Eigen::SparseMatrix<float>& L, Eigen::MatrixXf& b) {
    const int n = p.size();
    std::vector<Eigen::Triplet<float>> triplets;
    b = Eigen::MatrixXf::Zero(n, 3);
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float weight_sum = 0.0f;

        for (int j : neighbors_[i]) {
            float w_ij = weights_[i].at(j);
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

void ARAPSolver::solveARAP() {
    switch (arapImplementation) {
        case PAPER_ARAP:
            solveARAPPaper();
            break;
        case CERES_ARAP:
            solveARAPCeres();
            break;
        case IGL_ARAP:
            // IGL implementation (mocked for now, calls paper ARAP) TODO
            solveARAPIgl();
            break;
        default:
            solveARAPPaper();
            break;
    }
}

void ARAPSolver::solveARAPIgl() {
    std::cout << "[Solver] Using IGL ARAP implementation" << std::endl;
    
    if (!hasMesh()) {
        std::cout << "[Solver] No mesh loaded for ARAP solve" << std::endl;
        return;
    }
    
    // Skip if no constraints (nothing to deform)
    if (constraintIndices_.empty()) {
        std::cout << "[Solver] No constraints set for ARAP solve, skipping" << std::endl;
        return;
    }
    
    size_t num_of_constraints = constraintIndices_.size();
    Eigen::MatrixXd V = vertices_;
    Eigen::MatrixXd U = V;
    Eigen::MatrixXi F = faces_;
    Eigen::VectorXi b(num_of_constraints);
    Eigen::MatrixXd bc(num_of_constraints,3);
    
    for(int i = 0; i < num_of_constraints; i++) {
        b(i) = constraintIndices_[i];
        bc.row(i) = constraintPositions_[i];
    }

    igl::ARAPData arap_data;
    arap_data.max_iter = numberOfIterations;
    std::cout << "Vector b:\n" << b << "\n\n";
    std::cout << "Matrix bc:\n" << bc << "\n";
    igl::arap_precomputation(V, F, V.cols(), b, arap_data);
    igl::arap_solve(bc, arap_data, U);

    vertices_ = U;
}

void ARAPSolver::solveARAPPaper() {
    if (!hasMesh()) {
        std::cout << "[Solver] No mesh loaded for ARAP solve" << std::endl;
        return;
    }
    
    // Skip if no constraints (nothing to deform)
    if (constraintIndices_.empty()) {
        std::cout << "[Solver] No constraints set for ARAP solve, skipping" << std::endl;
        return;
    }

    // Compute weights if not done yet
    if (!weightsComputed_) {
        computeCotangentWeights();
    }

    const int n = vertices_.rows();
    
    // Convert to float vectors for ARAP computation
    std::vector<Eigen::Vector3f> p(n), p_prime(n);
    for (int i = 0; i < n; ++i) {
        p[i] = vertices_.row(i).cast<float>();
        p_prime[i] = p[i]; // Initialize deformed positions
    }

    // Prepare constraints map
    std::unordered_map<int, Eigen::Vector3f> constraints;
    for (size_t i = 0; i < constraintIndices_.size(); ++i) {
        constraints[constraintIndices_[i]] = constraintPositions_[i].cast<float>();
    }

    // ARAP iterations (local-global alternation)
    std::vector<Eigen::Matrix3f> R(n);
    const int iterations = numberOfIterations;  // Increased iterations for better convergence

    for (int iter = 0; iter < iterations; ++iter) {
        // Local step: compute optimal rotations
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            R[i] = computeOptimalRotation(i, p, p_prime);
        }

        // Global step: solve linear system
        Eigen::SparseMatrix<float> L;
        Eigen::MatrixXf b;
        buildLaplacianAndRHS(p, R, L, b);

        // Apply constraints (hard constraints via penalty method)
        const float penalty = 1e6f;  // Reduced penalty to avoid numerical issues
        for (const auto& [idx, pos] : constraints) {
            // Add penalty term to diagonal
            L.coeffRef(idx, idx) += penalty;
            // Set RHS to penalty * constraint position
            b.row(idx) += penalty * pos.transpose();
        }

        // Add small regularization to diagonal for numerical stability
        for (int i = 0; i < n; ++i) {
            L.coeffRef(i, i) += 1e-8f;
        }

        // Solve system using selected solver type
        Eigen::MatrixXf p_prime_mat;
        if (paperSolverType == PAPER_CHOLESKY) {
            // Use Cholesky solver (as specified in ARAP paper)
            Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> solver;
            solver.compute(L);
            
            if (solver.info() != Eigen::Success) {
                std::cerr << "[Solver] Cholesky factorization failed" << std::endl;
                return;
            }
            
            p_prime_mat = solver.solve(b);
            if (solver.info() != Eigen::Success) {
                std::cerr << "[Solver] Cholesky solve failed" << std::endl;
                return;
            }
        } else if(paperSolverType == PAPER_LDLT) {
            // Use LDLT solver
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
            solver.compute(L);
            
            if (solver.info() != Eigen::Success) {
                std::cerr << "[Solver] LDLT factorization failed" << std::endl;
                return;
            }
            
            p_prime_mat = solver.solve(b);
            if (solver.info() != Eigen::Success) {
                std::cerr << "[Solver] LDLT solve failed" << std::endl;
                return;
            }
        } else {
            // Use LU solver
            Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
            solver.compute(L);
            
            if (solver.info() != Eigen::Success) {
                std::cerr << "[Solver] LU factorization failed" << std::endl;
                return;
            }
            
            p_prime_mat = solver.solve(b);
            if (solver.info() != Eigen::Success) {
                std::cerr << "[Solver] LU solve failed" << std::endl;
                return;
            }
        }

        // Update positions
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            p_prime[i] = p_prime_mat.row(i).transpose();
        }
    }

    // Update mesh vertices with deformed positions
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        vertices_.row(i) = p_prime[i].cast<double>();
    }

    std::cout << "[Solver] ARAP solve completed (" << iterations << " iterations)" << std::endl;
}
void ARAPSolver::solveARAPCeres() {

    const int n = vertices_.rows();
    if (!hasMesh()) {
        std::cout << "[Solver] No mesh loaded for ARAP solve" << std::endl;
        return;
    }
    
    // Skip if no constraints (nothing to deform)
    if (constraintIndices_.empty()) {
        std::cout << "[Solver] No constraints set for ARAP solve, skipping" << std::endl;
        return;
    }

    // Compute weights if not done yet
    if (!weightsComputed_) {
        computeNeighbours();
    }

    ceres::Problem problem;

    std::vector<std::array<double, 3>> p_prime(n), angle(n);
    
    for (auto& arr : angle) {
        arr.fill(0.0);
    }

    
    for(size_t i = 0; i < constraintIndices_.size(); ++i) {
        auto constraintFunction = CeresSolver::EqualityConstraint::create(constraintPositions_[i], 5.0);
        problem.AddResidualBlock(constraintFunction, nullptr, (double*) &p_prime[constraintIndices_[i]]);
    }
    
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d p_i = vertices_.row(i).transpose();
        p_prime[i][0] = p_i(0);
        p_prime[i][1] = p_i(1);
        p_prime[i][2] = p_i(2);

        if (neighbors_[i].empty()) {
            continue;
        }
    
        for (int j : neighbors_[i]) {
            
            Eigen::Vector3d p_j = vertices_.row(j).transpose();
            
            auto constraintFunction = CeresSolver::EnergyCostFunction::create(p_i, p_j, 1.0);
            problem.AddResidualBlock(constraintFunction, nullptr, (double*) &p_prime[i], (double*) &p_prime[j], (double*) &angle[i]);
        }
    }

    ceres::Solver::Options options;
	options.max_num_iterations = numberOfIterations;
    
	options.linear_solver_type = getSolverType();
    options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
    std::cout<<"Start solving" << std::endl;
	ceres::Solve(options, &problem, &summary);
    std::cout<<"End solving" << std::endl;
    
    std::cout << summary.BriefReport() << std::endl;

    // Update mesh vertices with deformed positions
    //#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        vertices_.row(i)(0) = p_prime[i][0];
        vertices_.row(i)(1) = p_prime[i][1];
        vertices_.row(i)(2) = p_prime[i][2];
    }

}

} // namespace Solver
