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

    /// @brief A fully connected linear layer, currently only this layer stores weights
    class LinearLayer : public Layer
    {
    private:
        int input_size;
        int output_size;
        // (output_size, input_size) for weights, (output_size,) for biases
        Eigen::MatrixXf weights;
        Eigen::VectorXf biases;
        Eigen::MatrixXf cached_input;
        float learning_rate;

    public:
        LinearLayer(int in_size, int out_size, float lr) : input_size(in_size), output_size(out_size), learning_rate(lr)
        {
            weights = Eigen::MatrixXf::Random(output_size, input_size);
            biases = Eigen::VectorXf::Random(output_size);
        }
        Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;
        /// @brief Backward pass, weights and bias updated
        /// @param grad_output
        /// @return
        Eigen::MatrixXf backward(const Eigen::MatrixXf &grad_output) override;
        Eigen::MatrixXf get_weights() const { return weights; }
        Eigen::VectorXf get_biases() const { return biases; }
        void set_parameters(const Eigen::MatrixXf &new_weights, const Eigen::VectorXf &new_biases)
        {
            weights = new_weights;
            biases = new_biases;
        }
        int get_input_size() const { return input_size; }
        int get_output_size() const { return output_size; }
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
