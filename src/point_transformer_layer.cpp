#include "point_transformer_layer.h"

torch::Tensor PointTransformerLayerImpl::forward(torch::Tensor x, torch::Tensor p) {
    int64_t N = x.size(0);  // batch of N points

    // delta_p: (N, N, pos_dim)
    auto delta_p = p.unsqueeze(1) - p.unsqueeze(0);
    auto delta_p_flat = delta_p.reshape({-1, delta_p.size(-1)});

    auto position_encoding = position_encoder->forward(delta_p_flat); // (N*N, F)
    position_encoding = position_encoding.reshape({N, N, -1});       // (N, N, F)

    auto phi_x = phi->forward(x).unsqueeze(1);  // (N, 1, F)
    auto psi_x = psi->forward(x).unsqueeze(0);  // (1, N, F)

    auto gamma_input = phi_x - psi_x + position_encoding;  // (N, N, F)
    auto gamma_output = gamma_mlp->forward(gamma_input);    // (N, N, F)
    auto attn = rho(gamma_output);                         // (N, N, F)

    auto alpha_x = alpha->forward(x).unsqueeze(0);         // (1, N, F)
    auto feat_branch = alpha_x + position_encoding;        // (N, N, F)

    auto y = torch::sum(attn * feat_branch, 1);            // (N, F)
    return y;
}