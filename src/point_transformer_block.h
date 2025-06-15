#pragma once
#include <torch/torch.h>
#include "point_transformer_layer.h"

struct PointTransformerBlockImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    PointTransformerLayer point_transformer_layer;

    PointTransformerBlockImpl(int64_t in_features, int64_t out_features, int64_t position_dim)
        : fc1(register_module("fc1", torch::nn::Linear(in_features, out_features))),
          fc2(register_module("fc2", torch::nn::Linear(out_features, out_features))),
          point_transformer_layer(out_features, position_dim) {}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor p) {
        auto residual = x.clone();

        x = fc1->forward(x);
        x = point_transformer_layer->forward(residual, p);
        x = fc2->forward(x);

        x += residual;

        return {x, p};
    }
};
TORCH_MODULE(PointTransformerBlock);  // Alias: PointTransformerBlock = shared_ptr<PointTransformerBlockImpl>