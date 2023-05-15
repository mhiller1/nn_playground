#include <string>

#include <iostream>
#include <ctime>
#include <csignal>

#include "cxxopts.hpp"

#include "nn_utils.hpp"
#include "protein_seq.hpp"
#include "PVAE_model.hpp"

void run(const cxxopts::ParseResult& opts){

    constexpr long long netlen = 512;

    auto use_gpu = opts["train"].as<bool>();
    auto cuda_available = torch::cuda::is_available();

    if(use_gpu && !cuda_available){
        std::cout << "CUDA usage requested but not available" << std::endl;
    }

    torch::Device device((cuda_available && use_gpu) ? torch::kCUDA : torch::kCPU);
    PVAE model;

    if(opts.count("load")){
        auto opt_load = opts["load"].as<std::string>();
        torch::serialize::InputArchive archive;
        archive.load_from(opt_load);
        model->load(archive);
    }

    if(opts["train"].as<bool>()){
        const int64_t batch_size = opts["batch"].as<size_t>();
        const size_t num_epochs = opts["epochs"].as<size_t>();

        const double max_learning_rate = (1.0 / std::pow(10, opts["maxlr"].as<size_t>()));
        const double min_learning_rate = (1.0 / std::pow(10, opts["minlr"].as<size_t>()));

        auto ms = ProtGen<netlen>();
        auto dataset = ms.map(torch::data::transforms::Stack<>());
        auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset, batch_size);
        auto num_samples = dataset.size().value();

        model->to(device);

        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(max_learning_rate));
        //torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(max_learning_rate));

        for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
            torch::Tensor examples;
            size_t batch_index = 0;

            change_lr(optimizer, min_learning_rate + ((max_learning_rate-min_learning_rate) * ((epoch+1)/num_epochs)));

            model->train();

            for (auto& batch : *dataloader) {
                // Transfer examples to device
                examples = batch.data.reshape({-1, 1, netlen, 21}).to(device);

                // Forward pass
                auto output = model->forward(examples);

                // Compute reconstruction loss and kl divergence
                // For KL divergence, see Appendix B in VAE paper https://arxiv.org/pdf/1312.6114.pdf
                auto reconstruction_loss = torch::nn::functional::binary_cross_entropy(output.reconstruction,
                                           examples, torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(torch::kSum));
                /*auto reconstruction_loss = torch::nn::functional::mse_loss(output.reconstruction,
                                             examples, torch::nn::functional::MSELossFuncOptions().reduction(torch::kSum));*/
                auto kl_divergence = -0.5 * torch::sum(1 + output.log_var - output.mu.pow(2) - output.log_var.exp());

                // Backward pass and optimize
                const auto loss = reconstruction_loss + kl_divergence;
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "],\t Step [" << batch_index + 1 << "/"
                          << num_samples / batch_size << "],\t Reconstruction loss: "
                          << reconstruction_loss.item<float>() / batch.data.size(0)
                          << ",\t KL-divergence: " << kl_divergence.item<float>() / batch.data.size(0)
                          << std::endl;

                ++batch_index;
            }

            model->eval();
            torch::NoGradGuard no_grad;
        }
    }else{
        model->eval();
    }

    if(opts.count("save")){
        torch::serialize::OutputArchive output_archive;
        model->save(output_archive);
        output_archive.save_to(opts["save"].as<std::string>());
    }

    if(opts.count("test")){
        auto tests = opts["test"].as<size_t>();
        //std::ofstream test("test.txt", std::ofstream::out | std::ofstream::binary);
        for(size_t i = 0; i < tests; ++i){
            auto seq = randseq(randint(128, netlen), netlen);
            auto ten = seq_to_tensor(seq).reshape({1, netlen, 21}).to(device);
            std::cout << seq << "\n" << tensor_to_seq(model->forward(ten).reconstruction.reshape({netlen, 21})) << "\n" << std::endl;
        }
    }
}

int main(int argc, char* argv[]){

    std::srand(std::time(nullptr));

    auto sig_handler = [](int sig){
        throw std::runtime_error(std::string("Signal received: " + std::to_string(sig)));
    };

    std::signal(SIGABRT, sig_handler);
    std::signal(SIGFPE , sig_handler);
    std::signal(SIGILL , sig_handler);
    std::signal(SIGINT , sig_handler);
    std::signal(SIGSEGV, sig_handler);
    std::signal(SIGTERM, sig_handler);

    const cxxopts::ParseResult opts = [&](){
        cxxopts::Options options("ProtVAE multitool", "");

        options.add_options()
            ("h,help", "Print usage")
            ("l,load", "Load pretrained model", cxxopts::value<std::string>())
            ("s,save", "Save trained model", cxxopts::value<std::string>())
            ("t,train", "Train model", cxxopts::value<bool>()->default_value("false"))
            ("g,gpu", "Use gpu for training and inference", cxxopts::value<bool>()->default_value("false"))
            ("k,test", "Training epochs", cxxopts::value<size_t>()->default_value("10"))
            ("e,epochs", "Training epochs", cxxopts::value<size_t>()->default_value("10"))
            ("b,batch", "Batch size", cxxopts::value<size_t>()->default_value("500"))
            ("m,maxlr", "Max negative exponent of learning rate", cxxopts::value<size_t>()->default_value("3"))
            ("n,minlr", "Min negative exponent of learning rate", cxxopts::value<size_t>()->default_value("4"))
            ;

        const auto result = options.parse(argc, argv);

        if (result.count("help")){
            std::cout << options.help() << std::endl;
            std::exit(0);
        }

        return result;
    }();

    try{
        run(opts);
    }catch(const std::exception &e){
        std::cout << "Error:\n" << e.what() << std::endl;
    }

    return 0;
}
