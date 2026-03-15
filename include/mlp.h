#ifndef MLP_H
#define MLP_H

#include <vector>
#include <Eigen/Dense>
#include "layers.h"

class MLP
{
private:
    std::vector<nn::Layer *> layers;

public:
    void add_layer(nn::Layer *layer)
    {
        layers.push_back(layer);
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input)
    {
        Eigen::MatrixXf output = input;
        for (auto &layer : layers)
        {
            output = layer->forward(output);
        }
        return output;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf &loss_grad)
    {
        Eigen::MatrixXf grad = loss_grad;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it)->backward(grad);
        }
        return grad;
    }

    void save_model(const std::string &file_path);
    void load_model(const std::string &file_path);
};

#endif // MLP_H