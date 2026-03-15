#include "mlp.h"
#include <fstream>

void MLP::save_model(const std::string &file_path)
{
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file for saving model.");
    }
}