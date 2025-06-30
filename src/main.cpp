#include <vector>
#include "Eigen.h"
#include "SimpleMesh.h"

std::set<unsigned int> getConnectedVertexIndices(unsigned int vertexIndex, const SimpleMesh &mesh)
{
  const auto &cells = mesh.getCells();
  return cells[vertexIndex].neighborVertexIndices;
}

std::vector<Vector4f> getConnectedVertices(const std::set<unsigned int> &neighborIndices, const SimpleMesh &mesh)
{
  const auto &vertices = mesh.getVertices();

  std::vector<Vector4f> neighbors;
  for (const auto neighborIndex : neighborIndices)
  {
    neighbors.push_back(vertices[neighborIndex].position);
  }
  return neighbors;
}

MatrixXf computeR(Matrix3f &S)
{
  JacobiSVD<Matrix3f> svd(S, ComputeFullU | ComputeFullV);
  Matrix3f U = svd.matrixU();
  Matrix3f V = svd.matrixV();

  Matrix3f R = V * U.transpose();

  auto determinant = R.determinant();
  if (determinant < 0)
  {
    // change sign of the column of Ui corresponding to the smallest eigenvalue
    U.col(2) = -1 * U.col(2);
    R = V * U.transpose();
  }
  return R;
}

void calulateARAPStep(SimpleMesh &mesh, SimpleMesh &deformedMesh, const std::map<unsigned int, Vertex> &constraints)
{
  const auto &vertices = mesh.getVertices();
  auto &deformedVertices = deformedMesh.getVertices();
  auto size = vertices.size();
  std::vector<Matrix3f> rotations;
  rotations.reserve(size);

  for (unsigned int vertexIndex = 0; vertexIndex < size; vertexIndex++)
  {
    auto const &v = vertices[vertexIndex];
    auto const &v_def = deformedVertices[vertexIndex].position;

    auto connectedIndices = getConnectedVertexIndices(vertexIndex, mesh);

    Matrix3f S;
    for (unsigned int neighborIndex : connectedIndices)
    {
      Edge edge(vertexIndex, neighborIndex);
      const auto weight = mesh.getWeight(edge);
      const auto &neighbor = vertices[neighborIndex].position;
      const auto &neighbor_def = deformedVertices[neighborIndex].position;
      S += weight * (v.position.head(3) - neighbor.head(3)) * (v_def.head(3) - neighbor_def.head(3)).transpose();
    }

    rotations[vertexIndex] = computeR(S);
  }

  // Now we have the rotations for each vertex, we can compute the new positions
  // VectorXf b(size * 3);
  MatrixXf b = MatrixXf::Zero(size, 3);
  for (unsigned int vertexIndex = 0; vertexIndex < size; vertexIndex++)
  {
    auto const &v = vertices[vertexIndex];
    auto connectedIndices = getConnectedVertexIndices(vertexIndex, mesh);
    for (unsigned int neighborIndex : connectedIndices)
    {
      const auto neighbor = vertices[neighborIndex].position;
      Edge edge(vertexIndex, neighborIndex);
      const auto weight = mesh.getWeight(edge);
      Vector3f bv = weight * (rotations[vertexIndex] + rotations[neighborIndex]) * (v.position.head(3) - neighbor.head(3));
      // b.segment(vertexIndex*3, 3) += bv;
      b.block(vertexIndex, 0, 1, 3) += bv.transpose();
    }
  }

  SparseMatrix<float> L = mesh.getLaplacian();

  // apply constraints to L
  for (const auto &[index, vertex] : constraints)
  {
    for (SparseMatrix<float>::InnerIterator it(L, index); it; ++it)
    {
      it.valueRef() = 0; // Set all entries in the row to 0
    }
    // Set the diagonal entry to 1
    L.coeffRef(index, index) = 1;
    // Set the b vector to the position of the constrained vertex
    b.row(index) = vertex.position.head(3);
  }

#ifndef USE_LLT
  Eigen::SimplicialCholesky<SparseMatrix<float>> chol(L); // performs a Cholesky factorization of A
  MatrixXf x = chol.solve(b);
  auto info = chol.info();
  if (info != Eigen::Success)
  {
    std::cerr << "Solving failed: " << (int)info << std::endl;
    return;
  }
#else
  SimplicialLLT<SparseMatrix<float>> solver;
  solver.compute(L);
  auto info = solver.info();
  if (info != Eigen::Success)
  {
    std::cerr << "Decomposition failed: " << (int)info << std::endl;
    return;
  }
  MatrixXf x = solver.solve(b);
  info = solver.info();
  if (info != Eigen::Success)
  {
    std::cerr << "Solving failed: " << (int)info << std::endl;
    return;
  }
#endif
  // update the deformed mesh with the new positions
  for (unsigned int vertexIndex = 0; vertexIndex < size; vertexIndex++)
  {
    auto &deformedVertex = deformedVertices[vertexIndex].position;
    deformedVertex.segment(0, 3) = x.row(vertexIndex); // Set the new position
  }
}

void calulateARAP(SimpleMesh &mesh, const std::map<unsigned int, Vertex> &constraints, int iterations = 10)
{
  // create a copy of the mesh to deform
  SimpleMesh deformedMesh = mesh;
  for (int i = 0; i < iterations; i++)
  {
    calulateARAPStep(mesh, deformedMesh, constraints);
  }
  // update the mesh with the deformed positions
  for (unsigned int vertexIndex = 0; vertexIndex < deformedMesh.getVertices().size(); vertexIndex++)
  {
    mesh.getVertices()[vertexIndex].position = deformedMesh.getVertices()[vertexIndex].position;
  }
}

int main()
{
  SimpleMesh mesh;
#ifdef LOAD_MESH
  const std::string filename = std::string("cactus_small.off");

  if (!mesh.loadMesh(filename))
  {
    std::cout << "Mesh file wasn't read successfully at location: " << filename << std::endl;
    return -1;
  }
#else
  mesh = SimpleMesh::tetrahedron();
#endif

  mesh.buildCells();
  mesh.calculateWeights();
  mesh.calculateLaplacian();

  // Setup constraints
  std::map<unsigned int, Vertex> constraints;
  constraints[0] = mesh.getVertices()[0]; // Fix the first vertex
  constraints[1] = mesh.getVertices()[1]; // Simulate movement of the second vertex
  constraints[1].position << 1.2, 0.2, 0.1, 1.0;

  mesh.writeMesh("original_mesh.off");
  calulateARAP(mesh, constraints, 10);
  // Output the deformed mesh
  mesh.writeMesh("deformed_mesh.off");

  return 0;
}