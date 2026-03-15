#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <filesystem>
#include <span>

namespace fs = std::filesystem;
using namespace std;

class MNISTLoader
{
    /// @brief A class to load MNIST dataset, require one pair of image and label
private:
    // Metadata for one pair of image and label
    int num_images;
    int num_rows;
    int num_cols;

    // Storing the images and labels in a large vector
    // use rows and column to calculate the bias
    vector<uint8_t> images;
    vector<uint8_t> labels;

public:
    /// @brief constructor, default load the data, else throw error
    /// @param image_file_path
    /// @param label_file_path
    MNISTLoader(const fs::path &image_file_path, const fs::path &label_file_path)
    {
        load_data(image_file_path, label_file_path);
    };
    void load_data(const fs::path &image_file_path, const fs::path &label_file_path);
    void save_sample_image(int index, const fs::path &output_path);

    // getters
    int get_num_images() const { return num_images; }
    int get_num_rows() const { return num_rows; }
    int get_num_cols() const { return num_cols; }
    const vector<uint8_t> &get_images() const { return images; }
    const vector<uint8_t> &get_labels() const { return labels; }

    // Get the label and image at a specific index
    uint8_t get_label_at(int index) const { return labels[index]; }

    /// @brief Get the image at a specific index as a span (non-owning view)
    span<const uint8_t> get_image_at(int index);
};

#endif // MNIST_LOADER_H