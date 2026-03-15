#include "layers.h"

Eigen::MatrixXf nn::LinearLayer::forward(const Eigen::MatrixXf &input)
{
    cached_input = input;                               // Cache input for backward pass
    return (weights * input).colwise() + biases.col(0); // Linear transformation
}

Eigen::MatrixXf nn::LinearLayer::backward(const Eigen::MatrixXf &grad_output)
{
    // Compute gradients for weights and biases
    Eigen::MatrixXf grad_weights = grad_output * cached_input.transpose();
    Eigen::MatrixXf grad_biases = grad_output.rowwise().sum();

    // Compute gradient for input
    Eigen::MatrixXf grad_input = weights.transpose() * grad_output;

    // Update weights and biases (gradient descent)
    weights -= learning_rate * grad_weights;
    biases -= learning_rate * grad_biases.colwise().replicate(1);

    return grad_input;
}