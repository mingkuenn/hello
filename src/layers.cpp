#include "layers.h"

/// @brief Forward pass of linear layer, computes W*x + b
/// @param input dimension: (batch_size, input_size)
/// @return
Eigen::MatrixXf nn::LinearLayer::forward(const Eigen::MatrixXf &input)
{
    cached_input = input; // Cache input for backward pass
    // Broadcast bias to each row
    Eigen::MatrixXf output = (input * weights.transpose()) + biases.transpose().replicate(input.rows(), 1);

    return output;
}

Eigen::MatrixXf nn::LinearLayer::backward(const Eigen::MatrixXf &grad_output)
{
    // Use cached_input to compute gradients
    Eigen::MatrixXf grad_input = grad_output * weights;                    // Gradient w.r.t. input
    Eigen::MatrixXf grad_weights = grad_output.transpose() * cached_input; // Gradient w.r.t. weights
    Eigen::VectorXf grad_biases = grad_output.colwise().sum();             // Gradient w.r.t. biases

    // Update weights and biases
    weights -= learning_rate * grad_weights;
    biases -= learning_rate * grad_biases;

    return grad_input;
}