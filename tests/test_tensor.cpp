#include <gtest/gtest.h>

#include <include/tensor.h>

#include "exceptions.h"

TEST(Tensor, Base) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({ {1, 2, 3}, {4, 5, 6} }));
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(1, 1), 5);
    EXPECT_THROW(Tensor({{1, 2}, {3}}), BaseException);
}


TEST(Tensor, Index) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(0, 0), 1);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(0, 1), 2);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(0, 2), 3);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(1, 0), 4);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(1, 1), 5);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(1, 2), 6);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(-1, 0), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(-1, -1), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(0, -1), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(-100, -100), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(-100, 0), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(0, -100), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(100, 0), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(0, 100), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(100, 100), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(2, 0), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(0, 3), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(2, 3), BaseException);
}


TEST(Tensor, Equal) {
    using namespace SaintCore;
    floatT eps = 5e-6;
    floatT s1 = 1 + eps, s2 = 2 + eps, s3 = 3 + eps, s4 = 4 + eps, s5 = 5 + eps, s6 = 6 + eps;
    EXPECT_NE(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({ {s1, s2, s3}, {s4, s5, s6} }));
    eps = 5e-7;
    s1 = 1 + eps, s2 = 2 + eps, s3 = 3 + eps, s4 = 4 + eps, s5 = 5 + eps, s6 = 6 + eps;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({ {s1, s2, s3}, {s4, s5, s6} }));
    EXPECT_NE(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({{1, 2, 3}}));
}


TEST(Tensor, Sum) {
    using namespace SaintCore;
    floatT eps = 5e-6;
    floatT s1 = 1 + eps, s2 = 2 + eps, s3 = 3 + eps, s4 = 4 + eps, s5 = 5 + eps, s6 = 6 + eps;
    EXPECT_NE(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({ {s1, s2, s3}, {s4, s5, s6} }));
    eps = 5e-7;
    s1 = 1 + eps, s2 = 2 + eps, s3 = 3 + eps, s4 = 4 + eps, s5 = 5 + eps, s6 = 6 + eps;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({ {s1, s2, s3}, {s4, s5, s6} }));
    EXPECT_NE(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({{1, 2, 3}}));
}