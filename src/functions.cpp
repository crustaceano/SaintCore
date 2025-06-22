#include <include/functions.h>
#include <cmath>
#include <exceptions.h>


SaintCore::Tensor SaintCore::Functions::exp(const Tensor &input) {
    Tensor output(input.get_rows(), input.get_cols());
    for (int i = 0; i < input.get_rows(); ++i) {
        for (int j = 0; j < input.get_cols(); ++j) {
            output[i][j] = std::exp(input[i][j]);
        }
    }
    return output;
}

SaintCore::Tensor SaintCore::Functions::sum(const Tensor &input, int axis) {
    if (axis == -1) {
        Tensor output(input.get_rows(), 1);
        for (int i = 0; i < input.get_rows(); ++i) {
            floatT sum = 0;
            for (int j = 0; j < input.get_cols(); ++j) {
                sum += input[i][j];
            }
            output[i][0] = sum;
        }
        return output;
    }
    if (axis == 0 || axis == -2) {
        Tensor output(1, input.get_cols());
        for (int j = 0; j < input.get_cols(); ++j) {
            floatT sum = 0;
            for (int i = 0; i < input.get_rows(); ++i) {
                sum += input[i][j];
            }
            output[0][j] = sum;
        }
        return output;
    }
    throw InvalidArgumentException("Invalid axis for sum operation. Use -1 for row-wise sum or 0 for column-wise sum.");
}

SaintCore::Tensor SaintCore::Functions::softmax(const Tensor &input) {
    int batch_size = input.get_rows();
    Tensor exp_input = exp(input); // (batch_size, num_classes)
    Tensor sums = sum(exp_input, -1); // (batch_size, 1)

    Tensor result(batch_size, input.get_cols());

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input.get_cols(); ++j) {
            result[i][j] = exp_input[i][j] / sums[i][0];
        }
    }

    return result;
}
