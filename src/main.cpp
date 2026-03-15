#include "mnist_loader.h"
#include <filesystem>
#include "mlp.h"
#include "loss.h"
#include <random>

namespace fs = std::filesystem;

namespace nn
{
    constexpr float learning_rate = 0.01f;
    constexpr float tolerance = 1e-4f;
}

int main()
{
    fs::path base_path = fs::current_path();
    fs::path train_images_path = base_path / "archive" / "train-images-idx3-ubyte" / "train-images-idx3-ubyte";
    fs::path train_labels_path = base_path / "archive" / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte";
    fs::path test_images_path = base_path / "archive" / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte";
    fs::path test_labels_path = base_path / "archive" / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte";

    MNISTLoader train_data = MNISTLoader(train_images_path, train_labels_path);
    MNISTLoader test_data = MNISTLoader(test_images_path, test_labels_path);

    MLP mlp = MLP();
    mlp.add_layer(new nn::LinearLayer(28 * 28, 128));
    mlp.add_layer(new nn::ReluLayer());
    mlp.add_layer(new nn::LinearLayer(128, 10));
    mlp.add_layer(new nn::SigmoidLayer());

    // Random sample initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, train_data.get_num_images() - 1);

    // Draw sample to perform SGD minibatch
    const int batch_size = 64;
    int dimention = train_data.get_num_rows() * train_data.get_num_cols();
    Eigen::MatrixXf input(dimention, batch_size);
    Eigen::MatrixXf target = Eigen::MatrixXf::Zero(10, batch_size); // One-hot encoding for labels
    for (int i = 0; i < batch_size; ++i)
    {
        // Get random index
        int idx = distribution(gen);
        auto image_data = train_data.get_image_at(idx);
        for (int j = 0; j < dimention; ++j)
        {
            input(j, i) = static_cast<float>(image_data[j]) / 255.0f; // Normalize pixel values
        }
        // Set the corresponding label in the target matrix
        int label = train_data.get_label_at(idx);
        target(label, i) = 1.0f; // One-hot encoding
    }

    // Forward
    float previous_loss = std::numeric_limits<float>::max();
    vector<float> loss_history;
    int iteration = 0;
    while (true)
    {
        Eigen::MatrixXf output = mlp.forward(input);
        Eigen::VectorXf loss = loss::CrossEntropyLoss::forward(output, target);
        // Get the average loss for the batch
        float average_loss = loss.mean();
        loss_history.push_back(average_loss);
        previous_loss = average_loss;

        if (std::abs(previous_loss - average_loss) < nn::tolerance)
        {
            break;
        }

        if (iteration % 1000 == 0)
        {
            cout << "Iteration: " << iteration << ", Loss: " << average_loss << endl;
        }

        if (iteration++ > 10000)
        {
            cout << "Reached maximum iterations. Stopping training." << endl;
            break;
        }

        // Update weights
        Eigen::MatrixXf loss_grad = loss::CrossEntropyLoss::backward(output, target);
        mlp.backward(loss_grad);
    }

    // Save the trained model parameters
    mlp.save_parameters("model_parameters.txt");
}
