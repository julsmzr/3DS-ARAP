#pragma once
#include <torch/torch.h>

struct PositionEncoderImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    PositionEncoderImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim)
        : fc1(register_module("fc1", torch::nn::Linear(input_dim, hidden_dim))),
          fc2(register_module("fc2", torch::nn::Linear(hidden_dim, output_dim))) {}

    torch::Tensor forward(torch::Tensor delta_p) {
        // auto x = p_i - p_j;
        auto x = torch::relu(fc1->forward(delta_p));
        x = fc2->forward(x);
        return x;
    }
};
TORCH_MODULE(PositionEncoder);