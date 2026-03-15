#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <Eigen/Dense>

namespace nn
{
    class Layer
    {
    public:
        virtual ~Layer() = default;
        virtual Eigen::MatrixXf forward(const Eigen::MatrixXf &input) = 0;
        virtual Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output) { return grad_output; }
    };

    class LinearLayer : public Layer
    {
    private:
        int input_size;
        int output_size;
        Eigen::MatrixXf weights;
        Eigen::MatrixXf biases;
        Eigen::MatrixXf cached_input;
        float learning_rate = 0.01f;

    public:
        LinearLayer(int in_size, int out_size) : input_size(in_size), output_size(out_size)
        {
            weights = Eigen::MatrixXf::Random(output_size, input_size);
            biases = Eigen::MatrixXf::Random(output_size, 1);
        }
        Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;
        Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output) override;
    };

    class ReluLayer : public Layer
    {
    public:
        Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override
        {
            return input.cwiseMax(0.0f); // Apply ReLU activation
        }
        Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output) override
        {
            // Gradient of ReLU: 1 for input > 0, else 0
            return grad_output.cwiseGreater(0.0f).select(grad_output, 0.0f);
        }
    };

    class SigmoidLayer : public Layer
    {
    public:
        Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override
        {
            return 1.0f / (1.0f + (-input.array()).exp()); // Apply Sigmoid activation
        }
        Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output) override
        {
            // Gradient of Sigmoid: sigmoid(x) * (1 - sigmoid(x))
            auto sigmoid = forward(grad_output);
            return grad_output.cwiseProduct(sigmoid.cwiseProduct(Eigen::MatrixXf::Ones(sigmoid.rows(), sigmoid.cols()) - sigmoid));
        }
    };

}

#endif // LAYERS_H
