#include <gtest/gtest.h>
#include <cmath>
#include <include/functions.h>
#include <include/tensor.h>

TEST(test_exp_function, test_positive) {
    EXPECT_EQ(SaintCore::Functions::exp(SaintCore::Tensor({{1, 2, 3}})), SaintCore::Tensor({{static_cast<float>(std::exp(1)), static_cast<float>(std::exp(2)), static_cast<float>(std::exp(3))}}));
}