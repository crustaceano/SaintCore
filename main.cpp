#include <include/functions.h>
#include <include/model.h>
#include <include/exceptions.h>
#include <include/tensor.h>
#include <iostream>


int main() {
    SaintCore::Tensor input(3, 2);
    SaintCore::Models::LinearModel model(2, 3);
    SaintCore::Tensor output = model.forward(input);
    std::cout << "Output Tensor: \n" << output << '\n';
}