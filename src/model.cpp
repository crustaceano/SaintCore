#include <include/model.h>
#include <include/exceptions.h>
#include <include/tensor.h>


using namespace SaintCore;
using namespace SaintCore::Models;
LinearModel::~LinearModel() = default;

Tensor LinearModel::forward(const std::vector<Tensor> &inputs) {
    // input_dim - (1, in_channels)
    // weights - (in_channels, out_channels)
    // bias - (1, out_channels)
    // output_dim - (1, out_channels)
    Tensor input = inputs[0];
    if (input.get_cols() != in_channels) {
        throw SizeMismatchException(
            "LinearModel_forward: Input dimension in mismatch: expected (batch_size " + std::to_string(in_channels) + "), got ("
            + std::to_string(input.get_rows()) + ", " + std::to_string(input.get_cols()) + ")");
    }
    return input * weights + bias;
}


std::vector<Tensor*> LinearModel::get_parameters() const {
    return {const_cast<Tensor*>(&weights), const_cast<Tensor*>(&bias)};
}


void LinearModel::update_parameters(std::vector<Tensor> &new_params) {
    weights = new_params[0];
    bias = new_params[1];
}



Tensor LinearModel::getGrad(const std::vector<Tensor> &inputs) const {
    Tensor input = inputs[0];
    if (input.get_cols() != in_channels) {
        throw SizeMismatchException(
            "LinearModel_getGrad: Input dimension mismatch: expected (batch_size, " + std::to_string(in_channels) + "), got (" +
            std::to_string(input.get_rows()) + ", " + std::to_string(input.get_cols()) + ")");
    }

    return weights.transposed(); // (out_channels, in_channels) → (in_channels, out_channels)
}

Tensor LinearModel::propagateGrad(const std::vector<Tensor> &input, Tensor &grad) {
    Tensor cur_grads = getGrad(input);
    return grad * cur_grads;
}

std::vector<Tensor> LinearModel::grad_from_trainable(const std::vector<Tensor> &inputs, Tensor &grad) {
    Tensor input = inputs[0];
    return {input.transposed() * grad, Functions::sum(grad, 0)};
}

std::vector<Tensor> LinearModel::getTrainParams_grad(const Tensor& input) const {
    return {input.transposed(), get_E(1)};
}