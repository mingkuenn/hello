#ifndef MLP_H
#define MLP_H

#include <vector>
#include <Eigen/Dense>
#include "layers.h"
#include <filesystem>
#include <spdlog/spdlog.h>

class MLP
{
private:
    std::vector<nn::Layer *> layers;

public:
    void add_layer(nn::Layer *layer)
    {
        layers.push_back(layer);
    }

    /// @brief Forward pass through the entire network
    /// @param input (input_size, batch_size)
    /// @return
    Eigen::MatrixXf forward(const Eigen::MatrixXf &input)
    {
        Eigen::MatrixXf output = input;
        for (auto &layer : layers)
        {
            output = layer->forward(output);
        }
        return output;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf &loss_grad);

    /// @brief Get all weights and biases, implemented for linear layers only
    /// @return Vector of weights and biases, with weights followed by biases alternatively
    std::vector<Eigen::MatrixXf> get_params() const
    {
        std::vector<Eigen::MatrixXf> weights_list;
        std::vector<Eigen::MatrixXf> biases_list;
        for (const auto &layer : layers)
        {
            if (auto linear_layer = dynamic_cast<nn::LinearLayer *>(layer))
            {
                weights_list.push_back(linear_layer->get_weights());
                biases_list.push_back(linear_layer->get_biases());
            }
        }

        // Combine weights and biases into a single vector
        std::vector<Eigen::MatrixXf> params;
        for (size_t i = 0; i < weights_list.size(); ++i)
        {
            params.push_back(weights_list[i]);
            params.push_back(biases_list[i]);
        }
        return params;
    }

    void set_params(const std::vector<Eigen::MatrixXf> &params)
    {
        size_t index = 0;
        for (auto &layer : layers)
        {
            if (auto linear_layer = dynamic_cast<nn::LinearLayer *>(layer))
            {
                if (index + 1 < params.size())
                {
                    linear_layer->set_parameters(params[index], params[index + 1]);
                    index += 2; // Move to the next pair of weights and biases
                }
                else
                {
                    spdlog::error("Not enough parameters provided to set_params");
                    return;
                }
            }
        }
    }

    std::vector<nn::Layer *> get_layers() const
    {
        return layers;
    }

    void save_model(const std::filesystem::path &file_path);
    void load_model(const std::filesystem::path &file_path);
};

#endif // MLP_H