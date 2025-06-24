#include <iostream>
#include <vector>
#include <Eigen/Core>
#include "SimpleMesh.h"
#include "ARAPDeformer.h"

int main() {
    std::cout << "Interactive ARAP\n" << std::endl;

    const std::string filenameSource = "../Data/bunny.off";

    SimpleMesh sourceMesh;
    if (!sourceMesh.loadMesh(filenameSource)) {
        std::cerr << "Mesh file wasn't read successfully." << std::endl;
        return -1;
    }

    ARAPDeformer deformer(sourceMesh);

	unsigned idx = 5;
    std::vector<unsigned> handleIndices = {idx};
    Eigen::Vector3f pos = (sourceMesh.getVertices()[idx].position.head<3>() + Eigen::Vector3f(0, -0.05f, 0)).eval();
	std::vector<Eigen::Vector3f> handlePositions;
	handlePositions.push_back(pos);


    deformer.setHandles(handleIndices, handlePositions);

    std::cout << "Deforming mesh..." << std::endl;
    deformer.deform(25);

    const std::string outputFilename = "../Data/bunny_deformed.off";
    if (sourceMesh.writeMesh(outputFilename)) {
        std::cout << "Saved deformed mesh to: " << outputFilename << std::endl;
    } else {
        std::cerr << "Failed to save deformed mesh." << std::endl;
    }

    return 0;
}
