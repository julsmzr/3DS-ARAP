#include <gtest/gtest.h>
#include "solver.h"
#include "load_mesh.h"
#include <iostream>

// Fixture for rigidity tests
class ARAP : public ::testing::Test {
protected:
    MeshLoader::Mesh mesh;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    Solver::ARAPSolver solver;

    void SetUp() override {
        mesh.vertices = {
            { 0.0, 0.0, 0.0, },
            { 1.0, 0.0, 0.0, },
            { 0.0, 1.0, 0.0, },
            { 0.0, 0.0, 1.0, },
        };
        mesh.faces = {
            { 0,1,2, },
            { 0,1,3, },
            { 0,2,3, },
            { 1,2,3, },
        };

        vertices = Eigen::MatrixXd(mesh.vertices.size(), 3);
        for (size_t i = 0; i < mesh.vertices.size(); i++) {
            vertices.row(i) = mesh.vertices[i];
        }
        faces = Eigen::MatrixXi(mesh.faces.size(), 3);
        for (size_t i = 0; i < mesh.faces.size(); i++) {
            faces.row(i) = Eigen::Vector3i{mesh.faces[i][0], mesh.faces[i][1], mesh.faces[i][2]};
        }
        solver.setMesh(vertices, faces);
    }
};

double computeEdgeLengthRMS(
    const Eigen::MatrixXd& V_init,
    const Eigen::MatrixXd& V_deformed,
    const std::vector<std::vector<int>>& faces)
{
    using Edge = std::pair<int, int>;
    std::set<Edge> edge_set;

    // Step 1: Extract unique edges from triangle faces
    for (const auto& face: faces) {
        for (int j = 0; j < 3; ++j) {
            int a = face[j];
            int b = face[(j + 1) % 3];
            if (a > b) std::swap(a, b);
            edge_set.insert({a, b});
        }
    }

    // Step 2: Compute squared differences of edge lengths
    double sum_squared_diff = 0.0;
    for (const auto& edge : edge_set) {
        Eigen::Vector3d vi_init = V_init.row(edge.first);
        Eigen::Vector3d vj_init = V_init.row(edge.second);
        double len_init = (vi_init - vj_init).norm();

        Eigen::Vector3d vi_def = V_deformed.row(edge.first);
        Eigen::Vector3d vj_def = V_deformed.row(edge.second);
        double len_def = (vi_def - vj_def).norm();

        double diff = len_def - len_init;
        sum_squared_diff += diff * diff;
    }

    double rms = std::sqrt(sum_squared_diff / edge_set.size());
    return rms;
}


TEST_F(ARAP, Stationary) {
    // set stationary constraint
    std::vector<int> constraintIndices = { 0 };
    std::vector<Eigen::Vector3d> constraintPoints = { 
       { 0,0,0, } 
    };

    solver.setConstraints(constraintIndices, constraintPoints);
    solver.solveARAP();

    auto deformedVertices = solver.getVertices();
    auto deformedFaces = solver.getFaces();

    EXPECT_EQ(vertices.rows(), deformedVertices.rows());
    EXPECT_EQ(vertices.cols(), deformedVertices.cols());
    EXPECT_TRUE(vertices.isApprox(deformedVertices, 1e-7));
}

TEST_F(ARAP, Rigid) {
    // define constraint for a rigid transformation
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 4.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Vector3d t(0.5, 0.0, 0.0);

    Eigen::MatrixXd expectedVertices(mesh.faces.size(), 3);
    std::vector<int> constraintIndices;
    std::vector<Eigen::Vector3d> constraintPoints;
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        auto v_dash = R * mesh.vertices[i] + t;
        constraintPoints.emplace_back( v_dash );
        constraintIndices.emplace_back(i);
        expectedVertices.row(i) = v_dash;
    }

    solver.setConstraints(constraintIndices, constraintPoints);
    solver.solveARAP();

    auto deformedVertices = solver.getVertices();
    auto deformedFaces = solver.getFaces();

    EXPECT_TRUE(expectedVertices.isApprox(deformedVertices, 1e-7));
}

TEST_F(ARAP, NonRigid) {
    // define a non rigid constraint
    std::vector<int> constraintIndices;
    std::vector<Eigen::Vector3d> constraintPoints;
    constraintIndices.emplace_back(3);
    constraintPoints.emplace_back(Eigen::Vector3d{0.0, 0.0, 1.1});

    // define heuristic expected results
    Eigen::MatrixXd expectedVertices(4, 3);
    expectedVertices <<
        0.0, 0.0, 0.1,        
        1.0, 0.0, 0.1,        
        0.0, 1.0, 0.1,        
        0.0, 0.0, 1.1;        
    
    solver.setConstraints(constraintIndices, constraintPoints);
    solver.solveARAP();

    auto deformedVertices = solver.getVertices();
    auto deformedFaces = solver.getFaces();

    EXPECT_TRUE(expectedVertices.isApprox(deformedVertices, 1e-6));

    // check global rigidity, calculate edge RMS
    auto rms = computeEdgeLengthRMS(vertices, deformedVertices, mesh.faces);
    EXPECT_LT(rms, 1e-5);

#ifdef SHOW_CALCULATION
    // show calculation details
    std::cout << "RMS:" << rms << std::endl;
    std::cout << "Expected:" 
        << std::endl 
        << expectedVertices
        << std::endl;

    std::cout << "Deformed:" 
        << std::endl 
        << deformedVertices
        << std::endl;
#endif
}
