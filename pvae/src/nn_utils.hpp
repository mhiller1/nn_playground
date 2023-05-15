#pragma once

#include <mutex>
#include <string>

#include <torch/torch.h>

#include "protein_seq.hpp"
#include "common_utils.hpp"

void change_lr(torch::optim::Optimizer& opt, const double& lr){
    for (auto &group : opt.param_groups()){
        if(group.has_options()) ((torch::optim::OptimizerOptions&)(group.options())).set_lr(lr);
    }
}

torch::Tensor seq_to_tensor(const std::string& seq){
    auto ten = torch::zeros(seq.size(), at::TensorOptions(torch::kInt64));
    auto ptr = ten.data_ptr<int64_t>();
    for(const auto& i : seq){
        *ptr = aa_to_idx(i);
        ++ptr;
    }
    return torch::one_hot(ten, aminoacids.size()).toType(torch::kFloat);
}

std::string tensor_to_seq(const torch::Tensor& ten){
    std::string out(ten.sizes()[0], ' ');
    {
        auto am = torch::argmax(ten, 1);
        auto ptr = am.data_ptr<int64_t>();
        for(char& i : out){
            i = idx_to_aa(*ptr);
            ++ptr;
        }
    }
    return out;
}

template<size_t Seq_length, size_t Size = 10000, bool Allow_shorter = true>
class ProtGen : public torch::data::datasets::Dataset<ProtGen<Seq_length, Size, Allow_shorter>> {
public:
    torch::data::Example<> get(size_t index) override {
        auto out = seq_to_tensor(randseq(Allow_shorter ? randint(1, Seq_length) : Seq_length, Seq_length));
        return {out, out};
    }

    torch::optional<size_t> size() const override {
        return Size;
    }
};
