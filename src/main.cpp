#include "mnist_loader.h"
#include <filesystem>
#include "mlp.h"
#include "loss.h"
#include <random>
#include <iostream>
#include <spdlog/spdlog.h>
#include <H5Cpp.h>

namespace fs = std::filesystem;

struct ModelConfig
{
    int batch_size = 64;
    int class_count = 10;
    float learning_rate = 0.001f;
    float tolerance = 1e-2f;
    int input_dimension = 28 * 28;
    int max_iterations = 10000;

    // setters
    void set_batch_size(int new_batch_size) { batch_size = new_batch_size; }
    void set_class_count(int new_class_count) { class_count = new_class_count; }
    void set_learning_rate(float new_learning_rate) { learning_rate = new_learning_rate; }
    void set_tolerance(float new_tolerance) { tolerance = new_tolerance; }
    void set_input_dimension(int new_input_dimension) { input_dimension = new_input_dimension; }
    void set_max_iterations(int new_max_iterations) { max_iterations = new_max_iterations; }
};

struct TrainingMetadata
{
    int iteration = 0;
    float previous_loss = std::numeric_limits<float>::max();
    std::vector<float> loss_history;

    // setters
    void set_iteration(int new_iteration) { iteration = new_iteration; }
    void set_previous_loss(float new_previous_loss) { previous_loss = new_previous_loss; }
    void add_loss_history(float new_loss) { loss_history.push_back(new_loss); }
};

class MNISTClassifier
{
private:
    MLP model;
    Loss::_BaseLoss *loss_function;
    ModelConfig config = ModelConfig();
    TrainingMetadata training_metadata;

public:
    MNISTClassifier(MLP model, Loss::_BaseLoss *loss_function) : model(model), loss_function(loss_function) {}
    void train(MNISTLoader &train_data, int batch_size, int max_iterations);
    void evaluate(MNISTLoader &test_data);
    void save_model(const std::filesystem::path &file_path);
    void load_model(const std::filesystem::path &file_path);

    void edit_config(const ModelConfig &new_config) { config = new_config; }
};

void MNISTClassifier::train(MNISTLoader &train_data, int batch_size, int max_iterations)
{
    // Random sample initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, train_data.get_num_images() - 1);

    // Convert image to Eigen map
    vector<uint8_t> images = train_data.get_images();
    // Construct map with **ROW MAJOR** order and unaligned access
    // keep it uint8_t here first
    // Size: (num_images, input_dimension) = (60000, 784)
    int num_images = train_data.get_num_images();
    Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AlignmentType::Unaligned> image_map(images.data(), num_images, config.input_dimension);
    // Convert to float and normalize to [0, 1]
    Eigen::MatrixXf Eigeninput = image_map.cast<float>() / 255.0f;

    // Data initialization, buffers for input and target
    Eigen::MatrixXf input(batch_size, config.input_dimension);                      // Each row is an input sample
    Eigen::MatrixXf target = Eigen::MatrixXf::Zero(batch_size, config.class_count); // One-hot encoding for labels

    // Training loop
    spdlog::info("Starting training loop...");
    float previous_loss = std::numeric_limits<float>::max();
    vector<float> loss_history;
    int iteration = 0;
    while (true)
    {
        // Randomly sample a batch of data
        // 1. Randomly select indices for the batch
        vector<int> batch_indices(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            // For safety, ensure index within bound first.
            int generated_index = distribution(gen);
            assert(generated_index >= 0 && generated_index < train_data.get_num_images());
            batch_indices[i] = generated_index;
        }

        // 2. Prepare input and target matrices for the batch
        target.setZero(); // Clear target for this batch
        for (int i = 0; i < batch_size; ++i)
        {
            int index = batch_indices[i];
            input.row(i) = Eigeninput.row(index); // Get the image as a row vector
            int label = train_data.get_label_at(index);
            target(i, label) = 1.0f; // One-hot encoding
        }

        // Forward pass
        Eigen::MatrixXf output = model.forward(input);
        Eigen::VectorXf loss = loss_function->forward(output, target);

        // Get the average loss for the batch
        float average_loss = loss.mean();
        loss_history.push_back(average_loss);

        if (std::abs(previous_loss - average_loss) < config.tolerance)
        {
            spdlog::info("Convergence reached at iteration {}. Stopping training.", iteration);
            spdlog::info("Final Loss: {}", average_loss);
            break;
        }

        if (iteration % 100 == 0)
        {
            spdlog::info("Iteration: {}, Loss: {}", iteration, average_loss);
        }

        if (iteration >= config.max_iterations)
        {
            spdlog::info("Reached maximum iterations. Stopping training.");
            break;
        }

        // Update training metadata
        previous_loss = average_loss;
        iteration++;

        // Update weights
        Eigen::MatrixXf loss_grad = loss_function->backward(output, target);
        model.backward(loss_grad);
    }
};

void MNISTClassifier::evaluate(MNISTLoader &test_data)
{
    spdlog::info("Starting evaluation on test data...");

    // Get the matrix form of test images
    vector<uint8_t> images = test_data.get_images();
    Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AlignmentType::Unaligned> image_map(images.data(), test_data.get_num_images(), config.input_dimension);
    Eigen::MatrixXf Eigeninput = image_map.cast<float>() / 255.0f;
    vector<uint8_t> labels = test_data.get_labels();

    // Forward pass (returned logits, not probabilities)
    Eigen::MatrixXf output = model.forward(Eigeninput);

    // Apply softmax to match the loss function's transformation during training
    Eigen::VectorXf row_max = output.rowwise().maxCoeff();
    Eigen::MatrixXf shifted_output = output.colwise() - row_max;
    Eigen::MatrixXf exp_output = shifted_output.array().exp();
    Eigen::VectorXf row_sum = exp_output.rowwise().sum();
    Eigen::MatrixXf softmax_output = exp_output.array().colwise() / row_sum.array();

    // Convert to class predictions - softmax_output shape is (num_samples, num_classes)
    Eigen::VectorXi predicted_labels(softmax_output.rows());
    for (int i = 0; i < softmax_output.rows(); ++i)
    {
        Eigen::VectorXf::Index max_idx;
        softmax_output.row(i).maxCoeff(&max_idx);
        predicted_labels(i) = max_idx;
    }

    // Calculate accuracy
    int correct_predictions = 0;
    for (int i = 0; i < test_data.get_num_images(); ++i)
    {
        if (predicted_labels(i) == labels[i])
        {
            correct_predictions++;
        }
    }
    float accuracy = static_cast<float>(correct_predictions) / static_cast<float>(test_data.get_num_images());
    spdlog::info("Test Accuracy: {}", accuracy);
}

/// @brief Save models incluing layers and metadata. Not included history
/// @param file_path
void MNISTClassifier::save_model(const std::filesystem::path &file_path)
{
    // Flat architecture:
    // Root for metadata, then for each layer add a group with its param in attr and param as data
    H5::H5File file;
    try
    {
        file = H5::H5File(file_path.string(), H5F_ACC_TRUNC);
    }
    catch (const H5::FileIException &e)
    {
        spdlog::error("Failed to create file: {}", e.getCDetailMsg());
        return;
    }

    // Get layers ready
    std::vector<nn::Layer *> layers = model.get_layers();
    int layer_count = 0;
    for (const auto &layer : layers)
    {
        if (dynamic_cast<nn::LinearLayer *>(layer))
        {
            layer_count++;
        }
    }

    // Create a group for the model parameters
    H5::Group root = file.createGroup("/");
    H5::DataSpace attr_dataspace = H5::DataSpace(H5S_SCALAR);
    H5::Attribute layer_attr = root.createAttribute("num_layers", H5::PredType::NATIVE_INT, attr_dataspace);
    layer_attr.write(H5::PredType::NATIVE_INT, &layer_count);

    // Iteratively handle each layer
    layer_count = 0;
    for (const auto &layer : layers)
    {
        if (auto linear_layer = dynamic_cast<nn::LinearLayer *>(layer))
        {
            layer_count++;
            string layer_name = "layer" + std::to_string(layer_count);
            H5::Group layer_group = root.createGroup(layer_name);

            // Save attributes
            int input_size = linear_layer->get_input_size();
            int output_size = linear_layer->get_output_size();

            H5::Attribute input_attr = layer_group.createAttribute("input_size", H5::PredType::NATIVE_INT, attr_dataspace);
            input_attr.write(H5::PredType::NATIVE_INT, &input_size);
            H5::Attribute output_attr = layer_group.createAttribute("output_size", H5::PredType::NATIVE_INT, attr_dataspace);
            output_attr.write(H5::PredType::NATIVE_INT, &output_size);

            // Save weights and biases
            // Specify the dimensions as in-out dimensions
            Eigen::MatrixXf weights = linear_layer->get_weights();
            Eigen::MatrixXf biases = linear_layer->get_biases();
            hsize_t weight_dim[2] = {hsize_t(input_size), hsize_t(output_size)};
            hsize_t bias_dim[1] = {hsize_t(output_size)};
            H5::DataSpace weight_dataspace(2, weight_dim);
            H5::DataSpace bias_dataspace(1, bias_dim);
            H5::DataSet weight_dataset = layer_group.createDataSet("weights", H5::PredType::NATIVE_FLOAT, weight_dataspace);
            weight_dataset.write(weights.data(), H5::PredType::NATIVE_FLOAT);
            H5::DataSet bias_dataset = layer_group.createDataSet("biases", H5::PredType::NATIVE_FLOAT, bias_dataspace);
            bias_dataset.write(biases.data(), H5::PredType::NATIVE_FLOAT);
        }
    }

    spdlog::info("Model saved successfully to {}", file_path.string());
}

void MNISTClassifier::load_model(const std::filesystem::path &file_path)
{
    H5::H5File file;
    try
    {
        file = H5::H5File(file_path.string(), H5F_ACC_RDONLY);
    }
    catch (const H5::FileIException &e)
    {
        spdlog::error("Failed to open file: {}", e.getCDetailMsg());
        return;
    }

    // Open the model group
    H5::Group root = file.openGroup("/");
    // Check number of layers
    H5::Attribute layer_attr = root.openAttribute("num_layers");
    int h5_layer_count;
    layer_attr.read(H5::PredType::NATIVE_INT, &h5_layer_count);

    std::vector<nn::Layer *> layers = model.get_layers();
    int layer_count = 0;
    for (auto &layer : layers)
    {
        if (auto linear_layer = dynamic_cast<nn::LinearLayer *>(layer))
        {
            layer_count++;
            string layer_name = "layer" + std::to_string(layer_count);
            H5::Group layer_group = root.openGroup(layer_name);

            // Read attributes
            H5::Attribute input_attr = layer_group.openAttribute("input_size");
            H5::Attribute output_attr = layer_group.openAttribute("output_size");
            int input_size, output_size;
            input_attr.read(H5::PredType::NATIVE_INT, &input_size);
            output_attr.read(H5::PredType::NATIVE_INT, &output_size);

            // Read weights and biases
            H5::DataSet weight_dataset = layer_group.openDataSet("weights");
            H5::DataSet bias_dataset = layer_group.openDataSet("biases");
            Eigen::MatrixXf weights(input_size, output_size);
            Eigen::MatrixXf biases(output_size, 1);
            weight_dataset.read(weights.data(), H5::PredType::NATIVE_FLOAT);
            bias_dataset.read(biases.data(), H5::PredType::NATIVE_FLOAT);

            // Check layer dimensions
            if (linear_layer->get_input_size() != input_size || linear_layer->get_output_size() != output_size)
            {
                spdlog::error("Layer dimension mismatch for {}: expected input size {}, output size {}, but got input size {}, output size {}",
                              layer_name, linear_layer->get_input_size(), linear_layer->get_output_size(), input_size, output_size);
                continue;
            }

            // Set the parameters in the model
            linear_layer->set_parameters(weights, biases);
        }
    }

    // Check if all layers were loaded
    if (layer_count != h5_layer_count)
    {
        spdlog::error("Layer count mismatch: expected {}, but got {}", h5_layer_count, layer_count);
    };

    spdlog::info("Model loaded successfully from {}", file_path.string());
}

int main()
{
    // std::cout << "--- PROCESS STARTED ---" << std::endl;

    spdlog::info("Starting training...");
    fs::path base_path = fs::current_path();
    fs::path train_images_path = base_path / "archive" / "train-images-idx3-ubyte" / "train-images-idx3-ubyte";
    fs::path train_labels_path = base_path / "archive" / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte";
    fs::path test_images_path = base_path / "archive" / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte";
    fs::path test_labels_path = base_path / "archive" / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte";

    MNISTLoader train_data = MNISTLoader(train_images_path, train_labels_path);
    MNISTLoader test_data = MNISTLoader(test_images_path, test_labels_path);
    spdlog::info("Data loaded successfully. Number of training samples: {}, Number of test samples: {}",
                 train_data.get_num_images(), test_data.get_num_images());

    MLP mlp = MLP();
    mlp.add_layer(new nn::LinearLayer(28 * 28, 256, 0.01f));
    mlp.add_layer(new nn::ReluLayer());
    mlp.add_layer(new nn::LinearLayer(256, 256, 0.01f));
    mlp.add_layer(new nn::ReluLayer());
    mlp.add_layer(new nn::LinearLayer(256, 10, 0.01f));

    Loss::SoftmaxCrossEntropyLoss loss_function = Loss::SoftmaxCrossEntropyLoss();
    MNISTClassifier classifier = MNISTClassifier(mlp, &loss_function);
    classifier.train(train_data, 64, 10000);
    classifier.evaluate(test_data);

    // Save the trained model parameters
    std::filesystem::path model_save_path = base_path / "mlp_model.h5";
    mlp.save_model(model_save_path);
}