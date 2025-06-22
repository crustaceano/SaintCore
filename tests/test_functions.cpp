#include <gtest/gtest.h>
#include <cmath>
#include <include/functions.h>
#include <include/exceptions.h>
#include <include/tensor.h>

TEST(test_exp_function, test_exp_working) {
    using namespace SaintCore;
    EXPECT_EQ(Functions::exp(Tensor(std::vector<std::vector<float>>{{1, 2, 3}})),
              Tensor({{static_cast<floatT>(std::exp(1)), static_cast<floatT>(std::exp(2)), static_cast<float>(std::exp(3
                  ))
                  }}));
    EXPECT_EQ(Functions::exp(Tensor(std::vector<std::vector<float>>{{1}, {2}, {3}})),
              Tensor({{static_cast<floatT>(std::exp(1))}, {static_cast<floatT>(std::exp(2))}, {static_cast<float>(std::
                  exp
                  (3))}}));
}

TEST(test_sum, test_sum_dimensions) {
    using namespace SaintCore;
    Tensor input(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}});
    Tensor row_sum = Functions::sum(input, -1);
    EXPECT_EQ(std::make_pair(row_sum.get_rows(), row_sum.get_cols()), std::make_pair(input.get_rows(), 1));

    Tensor col_sum = Functions::sum(input, 0);
    EXPECT_EQ(std::make_pair(col_sum.get_rows(), col_sum.get_cols()), std::make_pair(1, input.get_cols()));
}

TEST(test_sum, test_sum_is_correct) {
    using namespace SaintCore;
    Tensor input(std::vector<std::vector<float>>{{1, 2, 3, 4, 5}, {4, 5, 6, 7, 8}});
    Tensor row_sum = Functions::sum(input, -1);
    EXPECT_EQ(std::make_pair(row_sum.get_rows(), row_sum.get_cols()), std::make_pair(2, 1));
    EXPECT_EQ(row_sum, Tensor(std::vector<std::vector<float>>{
                  {15.0},
                  {30.0}
                  }));

    Tensor col_sum = Functions::sum(input, 0);
    EXPECT_EQ(std::make_pair(col_sum.get_rows(), col_sum.get_cols()), std::make_pair(1, 5));
    EXPECT_EQ(col_sum, Tensor(std::vector<std::vector<float>>{{5, 7, 9, 11, 13}}));
}

TEST(test_softmax, test_output_size) {
    using namespace SaintCore;
    Tensor input(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}});
    Tensor output = Functions::softmax(input);
    EXPECT_EQ(std::make_pair(output.get_rows(), output.get_cols()),
              std::make_pair(input.get_rows(), input.get_cols()));
}


TEST(test_softmax, testing_softmax_values) {
    using namespace SaintCore;
    Tensor input(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}});
    Tensor output = Functions::softmax(input);

    for (int i = 0; i < output.get_rows(); ++i) {
        floatT sum = 0;
        for (int j = 0; j < output.get_cols(); ++j) {
            sum += output.at(i, j);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5);
    }
}
