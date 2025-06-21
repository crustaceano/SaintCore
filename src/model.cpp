#include <include/model.h>


SaintCore::Models::LinearModel::~LinearModel() {
    return;
}

SaintCore::Tensor SaintCore::Models::LinearModel::forward(const Tensor &input) {
    // input_dim - (1, in_channels)
    // weights - (in_channels, out_channels)
    // bias - (1, out_channels)
    // output_dim - (1, out_channels)
    return input * weights + bias;
}


