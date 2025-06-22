#include <gtest/gtest.h>
#include <cmath>
#include <include/functions.h>
#include <include/exceptions.h>
#include <include/tensor.h>

TEST(test_exp_function, test_exp_working) {
    using namespace SaintCore;
    EXPECT_EQ(Functions::exp(Tensor({{1, 2, 3}})), Tensor({{static_cast<float>(std::exp(1)), static_cast<float>(std::exp(2)), static_cast<float>(std::exp(3))}}));
    EXPECT_EQ(Functions::exp(Tensor({{1}, {2}, {3}})), Tensor({{static_cast<float>(std::exp(1))}, {static_cast<float>(std::exp(2))}, {static_cast<float>(std::exp(3))}}));
}

TEST(test_sum, test_sum_dimensions) {
    using namespace SaintCore;
    Tensor input({{1, 2, 3}, {4, 5, 6}});
    Tensor row_sum = Functions::sum(input, -1);
    EXPECT_EQ(row_sum.get_rows(), input.get_rows());
    EXPECT_EQ(row_sum.get_cols(), 1);

    Tensor col_sum = Functions::sum(input, 0);
    EXPECT_EQ(col_sum.get_rows(), 1);
    EXPECT_EQ(col_sum.get_cols(), input.get_cols());
}

TEST(test_sum, test_sum_is_correct) {
    using namespace SaintCore;
    Tensor input({{1, 2, 3, 4, 5}, {4, 5, 6, 7, 8}});
    Tensor row_sum = Functions::sum(input, -1);
    EXPECT_EQ(row_sum[0][0], 15);
    EXPECT_EQ(row_sum[1][0], 30);

    Tensor col_sum = Functions::sum(input, 0);
    EXPECT_EQ(col_sum, Tensor({{5, 7, 9, 11, 13}}));
}

TEST(test_softmax, test_output_size) {
    using namespace SaintCore;
    Tensor input({{1, 2, 3}, {4, 5, 6}});
    Tensor output = Functions::softmax(input);
    EXPECT_EQ(output.get_rows(), input.get_rows());
}


TEST(test_softmax, testing_softmax_values) {
    using namespace SaintCore;
    Tensor input({{1, 2, 3}, {4, 5, 6}});
    Tensor output = Functions::softmax(input);

    for (int i = 0; i < output.get_rows(); ++i) {
        floatT sum = 0;
        for (int j = 0; j < output.get_cols(); ++j) {
            sum += output[i][j];
        }
        EXPECT_EQ(sum, 1.0f);
    }
}