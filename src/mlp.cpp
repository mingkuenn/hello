#include "mlp.h"
#include <fstream>
#include <H5Cpp.h>
#include "layers.h"
#include <spdlog/spdlog.h>
#include <filesystem>

Eigen::MatrixXf MLP::backward(const Eigen::MatrixXf &loss_grad)
{
    Eigen::MatrixXf grad = loss_grad;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        // Backpropagate through each layer in reverse order
        // Weights are updated within each layer's backward function

        grad = (*it)->backward(grad);
    }
    return grad;
}

void MLP::save_model(const std::filesystem::path &file_path)
{
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

    // Create group for the model
    H5::Group model_group = file.createGroup("/model");

    // Save each layer's parameters
    int layer_index = 0;
    for (auto &layer : layers)
    {
        if (auto linear_layer = dynamic_cast<nn::LinearLayer *>(layer))
        {
            // Save weights
            Eigen::MatrixXf weights = linear_layer->get_weights();
            hsize_t dims[2] = {static_cast<hsize_t>(weights.rows()), static_cast<hsize_t>(weights.cols())};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = model_group.createDataSet("layer" + std::to_string(layer_index) + "_weights",
                                                            H5::PredType::NATIVE_FLOAT, dataspace);
            dataset.write(weights.data(), H5::PredType::NATIVE_FLOAT);

            // Save biases
            Eigen::MatrixXf biases = linear_layer->get_biases();
            hsize_t bias_dims[2] = {static_cast<hsize_t>(biases.rows()), static_cast<hsize_t>(biases.cols())};
            H5::DataSpace bias_dataspace(2, bias_dims);
            H5::DataSet bias_dataset = model_group.createDataSet("layer" + std::to_string(layer_index) + "_biases",
                                                                 H5::PredType::NATIVE_FLOAT, bias_dataspace);
            bias_dataset.write(biases.data(), H5::PredType::NATIVE_FLOAT);
        }
        layer_index++;
    }

    file.close();
}

void MLP::load_model(const std::filesystem::path &file_path)
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
    H5::Group model_group = file.openGroup("/model");

    // Load each layer's parameters
    int layer_index = 0;
    for (auto &layer : layers)
    {
        if (auto linear_layer = dynamic_cast<nn::LinearLayer *>(layer))
        {
            // Load weights
            H5::DataSet dataset = model_group.openDataSet("layer" + std::to_string(layer_index) + "_weights");
            H5::DataSpace dataspace = dataset.getSpace();
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims);
            Eigen::MatrixXf weights(dims[0], dims[1]);
            dataset.read(weights.data(), H5::PredType::NATIVE_FLOAT);
            linear_layer->get_weights() = weights;

            // Load biases
            H5::DataSet bias_dataset = model_group.openDataSet("layer" + std::to_string(layer_index) + "_biases");
            H5::DataSpace bias_dataspace = bias_dataset.getSpace();
            hsize_t bias_dims[2];
            bias_dataspace.getSimpleExtentDims(bias_dims);
            Eigen::MatrixXf biases(bias_dims[0], bias_dims[1]);
            bias_dataset.read(biases.data(), H5::PredType::NATIVE_FLOAT);
            linear_layer->get_biases() = biases;
        }
        layer_index++;
    }

    file.close();
}