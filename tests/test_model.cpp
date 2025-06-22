#include <gtest/gtest.h>
#include <include/model.h>
#include <include/tensor.h>
#include <include/exceptions.h>

TEST(test_model, test_linear_model_forward) {
    using namespace SaintCore::Models;
    using namespace SaintCore;

    LinearModel model(3, 5);
    Tensor input({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});

    // Forward pass
    Tensor output = model.forward(input);

    EXPECT_EQ(output.get_rows(), input.get_rows());
    EXPECT_EQ(output.get_cols(), 5);
}