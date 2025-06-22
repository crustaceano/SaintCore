#include <gtest/gtest.h>

#include <include/tensor.h>

#include "exceptions.h"

TEST(Tensor, Base) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({ {1, 2, 3}, {4, 5, 6} }));
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[1][1], 5);
    EXPECT_THROW(Tensor({{1, 2}, {3}}), BaseException);
}


TEST(Tensor, Index) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[0][0], 1);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[0][1], 2);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[0][2], 3);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[1][0], 4);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[1][1], 5);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[1][2], 6);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}})[-1], BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}})[3], BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}})[2], BaseException);
}


TEST(Tensor, Equal) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[0][0], 1);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[0][1], 2);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[0][2], 3);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[1][0], 4);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[1][1], 5);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} })[1][2], 6);
}