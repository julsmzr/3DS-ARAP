#pragma once
#include <torch/torch.h>
#include "gamma_mlp.h"
#include "position_encoder.h"

struct PointTransformerLayerImpl : torch::nn::Module {
    GammaMLP gamma_mlp;
    torch::nn::Linear phi{nullptr}, psi{nullptr}, alpha{nullptr};
    PositionEncoder position_encoder;
    torch::nn::Softmax rho;

    PointTransformerLayerImpl(int64_t feature_dim, int64_t position_dim)
        : gamma_mlp(feature_dim, feature_dim, feature_dim),
          phi(register_module("phi", torch::nn::Linear(feature_dim, feature_dim))),
          psi(register_module("psi", torch::nn::Linear(feature_dim, feature_dim))),
          alpha(register_module("alpha", torch::nn::Linear(feature_dim, feature_dim))),
          position_encoder(position_dim, feature_dim, feature_dim),
          rho(torch::nn::Softmax(torch::nn::SoftmaxOptions(-1))) {}

    torch::Tensor forward(torch::Tensor x, torch::Tensor p);
};

TORCH_MODULE(PointTransformerLayer);  // Makes shared_ptr alias