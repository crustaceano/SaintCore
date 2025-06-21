//
// Created by axmed on 21.06.2025.
//

#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include "tensor.h"

namespace SaintCore {
    namespace Functions {
        Tensor softmax(const Tensor& input);
        Tensor relu(const Tensor& output);
        Tensor cross_entropy(const Tensor& input, const Tensor& target);
    }
}

#endif //FUNCTIONS_H
