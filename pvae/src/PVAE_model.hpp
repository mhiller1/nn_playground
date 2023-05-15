#pragma once

#include <torch/torch.h>

struct PVAEOutput{
    torch::Tensor reconstruction;
    torch::Tensor mu;
    torch::Tensor log_var;
};

class PVAEImpl : public torch::nn::Module {

    torch::nn::Sequential enc_1;

    torch::nn::Sequential dec_1;
    torch::nn::Sequential dec_2;

    torch::nn::Linear enc_mu;
    torch::nn::Linear enc_log;

public:
    PVAEImpl():
        enc_1(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 3, {5, 5}).padding({2, 2}).stride({2, 1}).bias(false)), // 512*21 -> 256*21*2
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(3)),
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, {5, 5}).padding({2, 2}).stride({2, 1}).bias(false)), // 256*21*2 -> 128*21*2
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(3)),
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, {5, 5}).padding({2, 2}).stride({2, 1}).bias(false)),  // 128*21*2 -> 64*21*2
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(3)),
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, {5, 5}).padding({2, 2}).stride({2, 1}).bias(false)),  // 64*21*2 -> 32*21*2
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(3)),
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::Flatten(torch::nn::FlattenOptions()),
            torch::nn::Linear(32*21*3, 32*21*3),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(32*21*3)),
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true))
            ),

        enc_mu(32*21*3, 32*32),
        enc_log(32*21*3, 32*32),

        dec_1(
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(32*32)),
            torch::nn::Linear(32*32, 32*21*3),
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(32*21*3)),
            torch::nn::Linear(32*21*3, 32*21*3)
            ),

        dec_2(
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(3, 3, {6, 5}).padding({2, 2}).stride({2, 1}).bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(3)),
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(3, 3, {6, 5}).padding({2, 2}).stride({2, 1}).bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(3)),
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(3, 3, {6, 5}).padding({2, 2}).stride({2, 1}).bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(3)),
            torch::nn::ELU(torch::nn::ELUOptions().inplace(true)),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(3, 1, {6, 5}).padding({2, 2}).stride({2, 1}).bias(false)),
            torch::nn::Sigmoid()
            )
    {
        register_module("enc_1", enc_1);
        register_module("enc_mu", enc_mu);
        register_module("enc_log", enc_log);
        register_module("dec_1", dec_1);
        register_module("dec_2", dec_2);
    }

    torch::Tensor reparameterize(const torch::Tensor& mu, const torch::Tensor& log_var) {
        if (is_training()) {
            auto std = log_var.div(2).exp_();
            auto eps = torch::randn_like(std);
            return eps.mul(std).add_(mu);
        }
        return mu;
    }

    std::pair<torch::Tensor, torch::Tensor> encode(const torch::Tensor& x) {
        auto a = enc_1->forward(x.reshape({-1, 1, 512, 21}));
        return {enc_mu->forward(a), enc_log->forward(a)};
    }

    torch::Tensor decode(const torch::Tensor& x) {
        return dec_2->forward(dec_1->forward(x).reshape({-1, 3, 32, 21}));
    }

    PVAEOutput forward(const torch::Tensor& x) {
        //return decode(encode(x));
        auto encode_output = encode(x);
        auto mu = encode_output.first;
        auto log_var = encode_output.second;
        auto z = reparameterize(mu, log_var);
        auto x_reconstructed = decode(z);
        return {x_reconstructed, mu, log_var};
    }
};

TORCH_MODULE(PVAE);
