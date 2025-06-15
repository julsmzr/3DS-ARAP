#pragma once
#include <torch/torch.h>

struct GammaMLPImpl : torch::nn::Module {
    torch::nn::Sequential mlp;

    GammaMLPImpl(int64_t in_dim, int64_t hidden_dim, int64_t out_dim) {
        mlp = register_module("mlp", torch::nn::Sequential(
            torch::nn::Linear(in_dim, hidden_dim),
            torch::nn::ReLU(),
            torch::nn::Linear(hidden_dim, out_dim)
        ));
    }

    torch::Tensor forward(torch::Tensor x) {
        return mlp->forward(x);
    }
};

TORCH_MODULE(GammaMLP); // creates GammaMLP = std::shared_ptr<GammaMLPImpl>