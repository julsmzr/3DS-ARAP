#include <torch/torch.h>
#include <iostream>
#include <point_transformer_block.h>

int main() {
    // Set torch to CPU
    torch::Device device(torch::kCPU);

    // Dummy input data
    int64_t N = 5;            // number of points
    int64_t in_features = 16; // input feature dimension
    int64_t out_features = 16;
    int64_t position_dim = 3; // x,y,z

    // Random input features (N x in_features)
    auto x = torch::randn({N, in_features}, device);

    // Random positions (N x position_dim)
    auto p = torch::randn({N, position_dim}, device);

    // Create PointTransformerBlock instance
    PointTransformerBlock block(in_features, out_features, position_dim);
    block->to(device);

    // Run forward pass
    auto [output, positions] = block->forward(x, p);

    // Print output shapes
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Positions shape: " << positions.sizes() << std::endl;

    // Optionally print tensor data
    std::cout << "Output tensor: " << output << std::endl;

    return 0;
}