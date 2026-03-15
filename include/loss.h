#include <Eigen/Dense>

namespace loss
{
    struct MSELoss
    {
    public:
        /// @brief Input the predicted output (outdim, batchsize) and the target output (outdim, batchsize), return the MSE loss for each sample in the batch
        /// @param predictions
        /// @param targets
        /// @return
        static Eigen::VectorXf forward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets)
        {
            return (predictions - targets).colwise().squaredNorm();
        };

        static Eigen::VectorXf backward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets)
        {
            return 2.0f * (predictions - targets) / predictions.size();
        };
    };

    struct CrossEntropyLoss
    {
    public:
        static Eigen::VectorXf forward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets)
        {
            // Add a small epsilon to prevent log(0)
            const float epsilon = 1e-12f;
            Eigen::MatrixXf clipped_preds = predictions.array().max(epsilon).min(1.0f - epsilon);
            return -(targets.array() * clipped_preds.array().log()).colwise().sum();
        };

        static Eigen::VectorXf backward(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets)
        {
            // Add a small epsilon to prevent division by zero
            const float epsilon = 1e-12f;
            Eigen::MatrixXf clipped_preds = predictions.array().max(epsilon).min(1.0f - epsilon);
            return (clipped_preds - targets) / predictions.size();
        };
    };
};