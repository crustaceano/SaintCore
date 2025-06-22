#include <include/functions.h>
#include <include/model.h>
#include <include/exceptions.h>
#include <include/tensor.h>
#include <iostream>


int main() {
    SaintCore::Tensor input(1, 2);
    std::cout << "input_vector\n" << input << '\n';
    SaintCore::Models::LinearModel model(2, 3);
    std::cout << "model weights" << model.get_weights() << '\n';
    SaintCore::Tensor output = model.forward(input);
    std::cout << "Output Tensor: \n" << output << '\n';
}