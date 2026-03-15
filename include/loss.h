#include <Eigen/Dense>

namespace Loss
{
    class _BaseLoss
    {
    public:
        virtual Eigen::VectorXf forward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) = 0;
        virtual Eigen::MatrixXf backward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) = 0;
    };

    struct MSELoss : public _BaseLoss
    {
    public:
        /// @brief Input the predicted output (outdim, batchsize) and the target output (outdim, batchsize), return the MSE loss for each sample in the batch
        /// @param predictions
        /// @param targets
        /// @return
        Eigen::VectorXf forward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) override
        {
            return (predictions - targets).rowwise().squaredNorm();
        };

        Eigen::MatrixXf backward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) override
        {
            return 2.0f * (predictions - targets);
        };
    };

    struct CrossEntropyLoss : public _BaseLoss
    {
    public:
        Eigen::VectorXf forward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) override
        {
            // Add a small epsilon to prevent log(0)
            const float epsilon = 1e-12f;
            Eigen::MatrixXf clipped_preds = predictions.array().max(epsilon).min(1.0f - epsilon);
            return -(targets.array() * clipped_preds.array().log()).rowwise().sum();
        };

        Eigen::MatrixXf backward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) override
        {
            // Add a small epsilon to prevent division by zero
            const float epsilon = 1e-12f;
            Eigen::MatrixXf clipped_preds = predictions.array().max(epsilon).min(1.0f - epsilon);
            return (clipped_preds - targets);
        };

        Eigen::MatrixXf backward_with_logits(const Eigen::MatrixXf &logits, const Eigen::MatrixXf &targets)
        {
            // Subtract row max for numerical stability
            Eigen::VectorXf row_max = logits.rowwise().maxCoeff();
            Eigen::MatrixXf shifted_logits = logits.colwise() - row_max;
            Eigen::MatrixXf exp_logits = shifted_logits.array().exp();
            Eigen::VectorXf row_sum = exp_logits.rowwise().sum();
            Eigen::MatrixXf softmax = exp_logits.array().colwise() / row_sum.array();
            return (softmax - targets) / logits.rows();
        };
    };

    struct SigmoidCrossEntropyLoss : public _BaseLoss
    {
    public:
        /// @brief Sigmoid + Cross-entropy loss combined for numerical stability
        /// Forward computes sigmoid(predictions) then cross-entropy
        Eigen::VectorXf forward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) override
        {
            const float epsilon = 1e-12f;
            Eigen::MatrixXf sigmoid_preds = 1.0f / (1.0f + (-predictions.array()).exp());
            Eigen::MatrixXf clipped_preds = sigmoid_preds.array().max(epsilon).min(1.0f - epsilon);
            return -(targets.array() * clipped_preds.array().log() + (1.0f - targets.array()) * (1.0f - clipped_preds.array()).log()).rowwise().sum();
        };

        /// @brief Numerically stable backward: sigmoid(predictions) - targets
        /// This combines sigmoid and cross-entropy gradients for stability
        Eigen::MatrixXf backward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) override
        {
            Eigen::MatrixXf sigmoid_preds = 1.0f / (1.0f + (-predictions.array()).exp());
            return sigmoid_preds - targets;
        };
    };

    struct SoftmaxCrossEntropyLoss : public _BaseLoss
    {
    public:
        /// @brief Softmax + Cross-entropy loss for multi-class classification
        /// Forward: softmax(logits) then cross-entropy with targets
        Eigen::VectorXf forward(const Eigen::MatrixXf &logits, const Eigen::MatrixXf &targets) override
        {
            // Subtract row max for numerical stability
            Eigen::VectorXf row_max = logits.rowwise().maxCoeff();
            Eigen::MatrixXf shifted_logits = logits.colwise() - row_max;
            Eigen::MatrixXf exp_logits = shifted_logits.array().exp();
            Eigen::VectorXf row_sum = exp_logits.rowwise().sum();
            Eigen::MatrixXf softmax = exp_logits.array().colwise() / row_sum.array();

            // Cross-entropy loss: -sum(targets * log(softmax))
            const float epsilon = 1e-12f;
            Eigen::MatrixXf clipped_softmax = softmax.array().max(epsilon).min(1.0f - epsilon);
            return -(targets.array() * clipped_softmax.array().log()).rowwise().sum();
        };

        /// @brief Numerically stable backward: softmax(logits) - targets
        Eigen::MatrixXf backward(const Eigen::MatrixXf &logits, const Eigen::MatrixXf &targets) override
        {
            // Subtract row max for numerical stability
            Eigen::VectorXf row_max = logits.rowwise().maxCoeff();
            Eigen::MatrixXf shifted_logits = logits.colwise() - row_max;
            Eigen::MatrixXf exp_logits = shifted_logits.array().exp();
            Eigen::VectorXf row_sum = exp_logits.rowwise().sum();
            Eigen::MatrixXf softmax = exp_logits.array().colwise() / row_sum.array();

            return softmax - targets;
        };
    };
};
