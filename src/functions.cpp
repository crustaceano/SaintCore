#include <include/functions.h>
#include <cmath>

SaintCore::Tensor SaintCore::Functions::softmax(const SaintCore::Tensor& input) {
    Tensor output(input.get_rows(), input.get_cols());
    float sum = 0.0f;

    for (int i = 0; i < input.get_rows(); ++i) {
        for (int j = 0; j < input.get_cols(); ++j) {
            output[i][j] = std::exp(input[i][j]);
            sum += output[i][j];
        }
    }

    for (int i = 0; i < output.get_rows(); ++i) {
        for (int j = 0; j < output.get_cols(); ++j) {
            output[i][j] /= sum;
        }
    }

    return output;
}