#include <include/model.h>
#include <include/exceptions.h>

SaintCore::Models::LinearModel::~LinearModel() {
    return;
}

SaintCore::Tensor SaintCore::Models::LinearModel::forward(const Tensor &input) {
    // input_dim - (1, in_channels)
    // weights - (in_channels, out_channels)
    // bias - (1, out_channels)
    // output_dim - (1, out_channels)
    if (input.get_cols() != 1 || input.get_rows() != in_channels) {
        throw SizeMismatchException(
            "LinearModel_forward: Input dimension in mismatch: expected (1, " + std::to_string(in_channels) + "), got ("
            + std::to_string(input.get_rows()) + ", " + std::to_string(input.get_cols()) + ")");
    }
    return input * weights + bias;
}
