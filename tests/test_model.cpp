#include <gtest/gtest.h>
#include <include/model.h>
#include <include/tensor.h>
#include <include/exceptions.h>

TEST(test_model, test_linear_model_forward) {
    using namespace SaintCore::Models;
    using namespace SaintCore;

    LinearModel model(3, 2);
    Tensor input({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

    std::cout << "Weights: \n" << model.get_weights() << '\n';
    std::cout << "Bias: \n" << model.get_bias() << '\n';
    std::cout << "input: \n" << input << '\n';
    Tensor output = model.forward(input);

    std::cout << "output: \n" << output << '\n';


    EXPECT_EQ(output.get_rows(), input.get_rows());
    EXPECT_EQ(output.get_cols(), 2);
}