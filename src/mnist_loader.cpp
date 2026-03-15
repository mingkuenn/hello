#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <span>

#include "mnist_loader.h"

namespace fs = std::filesystem;

const fs::path _BASE_DIR = fs::current_path().string();
const fs::path test_data_path = _BASE_DIR / "archive" / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte";
const fs::path test_label_path = _BASE_DIR / "archive" / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte";

using namespace std;

// To decode the file, we need to handle endianess
uint32_t reverse_uint32(uint32_t num)
{
    return ((num >> 24) & 0x000000FF) |
           ((num >> 8) & 0x0000FF00) |
           ((num << 8) & 0x00FF0000) |
           ((num << 24) & 0xFF000000);
}

void MNISTLoader::load_data(const fs::path &image_file_path, const fs::path &label_file_path)
{
    /* Read image and label seperately, compare their correctness */
    const int image_magic_number = 2051;
    const int label_magic_number = 2049;
    vector<uint8_t> images;
    vector<uint8_t> labels;
    ifstream file(image_file_path, ios::binary);
    uint32_t num_images, num_rows, num_cols, num_labels, magic_number;

    if (!file.is_open())
    {
        cerr << "Failed to open file: " << image_file_path << endl;
        throw runtime_error("Failed to open image file.");
    }

    // Read image first
    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char *>(&num_cols), sizeof(num_cols));

    magic_number = reverse_uint32(magic_number);
    num_images = reverse_uint32(num_images);
    num_rows = reverse_uint32(num_rows);
    num_cols = reverse_uint32(num_cols);

    if (magic_number != image_magic_number)
    {
        cerr << "Invalid magic number in image file: " << magic_number << endl;
        throw runtime_error("Invalid magic number in image file.");
    }

    // Save with flat vector
    images.resize(num_images * num_rows * num_cols);
    file.read(reinterpret_cast<char *>(images.data()), num_images * num_rows * num_cols);

    file.close();

    // Now read the label file
    ifstream label_file(label_file_path, ios::binary);
    if (!label_file.is_open())
    {
        cerr << "Failed to open file: " << label_file_path << endl;
        throw runtime_error("Failed to open label file.");
    }

    label_file.read(reinterpret_cast<char *>(&magic_number), sizeof(label_magic_number));
    label_file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

    magic_number = reverse_uint32(magic_number);
    num_labels = reverse_uint32(num_labels);

    if (magic_number != label_magic_number)
    {
        cerr << "Invalid magic number in label file: " << magic_number << endl;
        throw runtime_error("Invalid magic number in label file.");
    }

    labels.resize(num_labels);
    label_file.read(reinterpret_cast<char *>(labels.data()), num_labels);

    label_file.close();

    // Check if the number of images matches the number of labels
    if (num_images != num_labels)
    {
        cerr << "Number of images (" << num_images << ") does not match number of labels (" << num_labels << ")." << endl;
        throw runtime_error("Number of images does not match number of labels.");
    }

    // Save to the class members
    this->num_images = num_images;
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->images = move(images);
    this->labels = move(labels);
}

void MNISTLoader::save_sample_image(int index, const fs::path &output_path)
{
    /* Input a sample pair of image and label */

    // Handle convertion to PGM in memory
    span<const uint8_t> image_data = get_image_at(index);
    vector<uint8_t> pgm_data(image_data.begin(), image_data.end());

    ofstream output_file(output_path, ios::binary);
    if (!output_file.is_open())
    {
        cerr << "Failed to open output file: " << output_path << endl;
        throw runtime_error("Failed to open output file.");
    }

    // Write PGM header
    output_file << "P5\n"
                << num_cols << " " << num_rows << "\n255\n";
    // Write image data
    output_file.write(reinterpret_cast<char *>(pgm_data.data()), pgm_data.size());
}

span<const uint8_t> MNISTLoader::get_image_at(int index)
{
    size_t offset = static_cast<size_t>(index) * num_rows * num_cols;
    return span<const uint8_t>(&images[offset], num_rows * num_cols);
}

// int main() {
//     try {
//         // Load the dataset
//         MNISTLoader dataset(test_data_path.string(), test_label_path.string());

//         // Random sample int, save image
//         srand(time(0));
//         int sample_idx = rand() % dataset.get_num_images();

//         // Get the corresponding sample image and label
//         uint8_t sample_label = dataset.get_label_at(sample_idx);
//         span<const uint8_t> sample_image = dataset.get_image_at(sample_idx);

//         // Save the sample image as PGM
//         fs::path output_path = "sample_image.pgm";
//         dataset.save_sample_image(sample_idx, output_path);
//         cout << "Sample image saved to: " << output_path << " with label: " << static_cast<int>(sample_label) << endl;
//     } catch (const exception& e) {
//         cerr << "Error: " << e.what() << endl;
//         return 1;
//     }

//     return 0;
// }